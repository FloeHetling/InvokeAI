import math
from typing import Any

import torch
from diffusers import ChromaTransformer2DModel

from invokeai.backend.chroma.numeric_diagnostics import (
    is_chroma_numeric_diagnostics_enabled,
    log_tensor_fingerprint,
)
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


class ChromaTransformerAdapter:
    """Adapt Diffusers' Chroma transformer to InvokeAI's rectified-flow denoiser interface."""

    def __init__(self, model: ChromaTransformer2DModel):
        self.model = model
        self._batched_cfg_negative_extension: RegionalPromptingExtension | None = None
        self._batched_cfg_scale: list[float] | None = None
        self._batched_cfg_disabled = False
        self._numeric_diagnostics_first_forward_logged = False

    def enable_batched_cfg(
        self,
        negative_extension: RegionalPromptingExtension,
        cfg_scale: list[float],
    ) -> None:
        """Combine positive and negative Chroma CFG branches into one transformer batch.

        The adapter owns this optimization so the shared FLUX denoiser can continue to see
        cfg_scale=1 and make one model call per step. If the larger batch runs out of memory,
        the adapter permanently falls back to the equivalent sequential two-pass path for
        the rest of this denoise.
        """
        self._batched_cfg_negative_extension = negative_extension
        self._batched_cfg_scale = cfg_scale

    def __call__(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor,
        timestep_index: int,
        total_num_timesteps: int,
        controlnet_double_block_residuals: Any,
        controlnet_single_block_residuals: Any,
        ip_adapter_extensions: list[Any],
        regional_prompting_extension: RegionalPromptingExtension,
    ) -> torch.Tensor:
        del y, guidance, total_num_timesteps
        if controlnet_double_block_residuals is not None or controlnet_single_block_residuals is not None:
            raise ValueError("Chroma ControlNet residuals are not supported")
        if ip_adapter_extensions:
            raise ValueError("Chroma IP-Adapter extensions are not supported")
        if regional_prompting_extension.restricted_attn_mask is not None:
            raise ValueError("Regional prompt masks are not supported by the Chroma transformer")

        cfg_scale = self._cfg_scale_for_step(timestep_index)
        negative_extension = self._batched_cfg_negative_extension
        if negative_extension is not None and not math.isclose(cfg_scale, 1.0):
            if negative_extension.restricted_attn_mask is not None:
                return self._run_sequential_cfg(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    timesteps=timesteps,
                    positive_extension=regional_prompting_extension,
                    negative_extension=negative_extension,
                    cfg_scale=cfg_scale,
                )

            if not self._batched_cfg_disabled and self._can_batch_cfg(
                img=img,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                positive_extension=regional_prompting_extension,
                negative_extension=negative_extension,
            ):
                try:
                    return self._run_batched_cfg(
                        img=img,
                        img_ids=img_ids,
                        txt=txt,
                        txt_ids=txt_ids,
                        timesteps=timesteps,
                        positive_extension=regional_prompting_extension,
                        negative_extension=negative_extension,
                        cfg_scale=cfg_scale,
                    )
                except torch.OutOfMemoryError:
                    # The optimization is opportunistic. A Chroma model may already be partially
                    # loaded on a low-VRAM device, so a batch-2 activation peak can exceed the
                    # configured working-memory reserve even though sequential CFG is viable.
                    self._batched_cfg_disabled = True
                    TorchDevice.empty_cache()
                    InvokeAILogger.get_logger(__name__).warning(
                        "Chroma batched CFG ran out of device memory; falling back to sequential guidance "
                        "for the rest of this denoise. Set sequential_guidance=true to skip the batched attempt."
                    )

            return self._run_sequential_cfg(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                timesteps=timesteps,
                positive_extension=regional_prompting_extension,
                negative_extension=negative_extension,
                cfg_scale=cfg_scale,
            )

        text_attention_mask = self._get_text_attention_mask(regional_prompting_extension, txt)
        return self._forward_model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            text_attention_mask=text_attention_mask,
        )

    def _cfg_scale_for_step(self, timestep_index: int) -> float:
        if not self._batched_cfg_scale:
            return 1.0
        return self._batched_cfg_scale[min(timestep_index, len(self._batched_cfg_scale) - 1)]

    @staticmethod
    def _normalize_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        # InvokeAI's shared FLUX conditioning/sampling utilities carry position IDs with a
        # leading batch dimension. Diffusers Chroma expects the batch-independent 2D form.
        return position_ids[0] if position_ids.ndim == 3 else position_ids

    @staticmethod
    def _get_text_attention_mask(
        regional_prompting_extension: RegionalPromptingExtension,
        txt: torch.Tensor,
    ) -> torch.Tensor:
        text_attention_mask = regional_prompting_extension.regional_text_conditioning.attention_mask
        if text_attention_mask is None:
            text_attention_mask = torch.ones(txt.shape[:2], dtype=torch.bool, device=txt.device)
        return text_attention_mask.to(device=txt.device, dtype=torch.bool)

    def predict_cfg_branches(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
        allow_batched: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return positive and negative Chroma predictions without combining CFG.

        CFG++ needs both branches after the transformer forward. When allowed, the branches
        share the same batch-2 optimization as classical Chroma CFG. A batched OOM disables
        further batched attempts on this adapter and falls back to two sequential forwards.
        """
        if positive_extension.restricted_attn_mask is not None or negative_extension.restricted_attn_mask is not None:
            raise ValueError("Regional prompt masks are not supported by the Chroma transformer")

        positive = positive_extension.regional_text_conditioning
        positive_txt = positive.t5_embeddings
        positive_txt_ids = positive.t5_txt_ids

        if allow_batched and not self._batched_cfg_disabled and self._can_batch_cfg(
            img=img,
            txt=positive_txt,
            txt_ids=positive_txt_ids,
            timesteps=timesteps,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
        ):
            try:
                return self._run_batched_cfg_branches(
                    img=img,
                    img_ids=img_ids,
                    txt=positive_txt,
                    txt_ids=positive_txt_ids,
                    timesteps=timesteps,
                    positive_extension=positive_extension,
                    negative_extension=negative_extension,
                )
            except torch.OutOfMemoryError:
                self._batched_cfg_disabled = True
                TorchDevice.empty_cache()
                InvokeAILogger.get_logger(__name__).warning(
                    "Chroma batched positive/negative forward ran out of device memory; falling back to sequential "
                    "guidance for the rest of this denoise. Set sequential_guidance=true to skip the batched attempt."
                )

        return self._run_sequential_cfg_branches(
            img=img,
            img_ids=img_ids,
            txt=positive_txt,
            txt_ids=positive_txt_ids,
            timesteps=timesteps,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
        )

    @staticmethod
    def _pad_sequence_to_length(tensor: torch.Tensor, target_length: int) -> torch.Tensor:
        """Right-pad the sequence dimension (dim=1) with zeros."""
        pad_length = target_length - tensor.shape[1]
        if pad_length < 0:
            raise ValueError("target_length must not be shorter than the input sequence")
        if pad_length == 0:
            return tensor
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_length
        padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, padding), dim=1)

    @staticmethod
    def _pad_position_ids_to_length(position_ids: torch.Tensor, target_length: int) -> torch.Tensor:
        """Right-pad normalized (sequence, axes) position IDs with zeros."""
        pad_length = target_length - position_ids.shape[0]
        if pad_length < 0:
            raise ValueError("target_length must not be shorter than the input position IDs")
        if pad_length == 0:
            return position_ids
        padding = torch.zeros(
            (pad_length, *position_ids.shape[1:]), dtype=position_ids.dtype, device=position_ids.device
        )
        return torch.cat((position_ids, padding), dim=0)

    def _prepare_batched_cfg_text(
        self,
        *,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Pad positive/negative text to a shared length for one transformer batch.

        Variable-length Chroma conditioning means the two CFG branches commonly have
        different sequence lengths. Padding is applied only at the transformer batch
        boundary; padded tokens are masked out, so sequential and batched guidance see
        the same effective conditioning.
        """
        negative = negative_extension.regional_text_conditioning
        negative_txt = negative.t5_embeddings

        if txt.ndim != 3 or negative_txt.ndim != 3:
            return None
        if txt.shape[0] != negative_txt.shape[0] or txt.shape[2:] != negative_txt.shape[2:]:
            return None

        positive_mask = self._get_text_attention_mask(positive_extension, txt)
        negative_mask = self._get_text_attention_mask(negative_extension, negative_txt)
        if positive_mask.shape != txt.shape[:2] or negative_mask.shape != negative_txt.shape[:2]:
            return None

        positive_txt_ids = self._normalize_position_ids(txt_ids)
        negative_txt_ids = self._normalize_position_ids(negative.t5_txt_ids)
        if positive_txt_ids.ndim != 2 or negative_txt_ids.ndim != 2:
            return None
        if positive_txt_ids.shape[0] != txt.shape[1] or negative_txt_ids.shape[0] != negative_txt.shape[1]:
            return None
        if positive_txt_ids.shape[1:] != negative_txt_ids.shape[1:]:
            return None

        target_length = max(txt.shape[1], negative_txt.shape[1])
        positive_txt = self._pad_sequence_to_length(txt, target_length)
        negative_txt = self._pad_sequence_to_length(negative_txt, target_length)
        positive_mask = self._pad_sequence_to_length(positive_mask, target_length)
        negative_mask = self._pad_sequence_to_length(negative_mask, target_length)
        positive_txt_ids = self._pad_position_ids_to_length(positive_txt_ids, target_length)
        negative_txt_ids = self._pad_position_ids_to_length(negative_txt_ids, target_length)

        # Diffusers Chroma accepts one batch-independent txt_ids tensor. Chroma text
        # position IDs are currently all zeros, so variable lengths remain compatible
        # after right-padding. Keep this check explicit in case that assumption changes.
        if not torch.equal(positive_txt_ids, negative_txt_ids):
            return None

        return positive_txt, negative_txt, positive_mask, negative_mask, positive_txt_ids

    def _can_batch_cfg(
        self,
        *,
        img: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
    ) -> bool:
        if img.shape[0] != txt.shape[0]:
            return False
        if timesteps.ndim == 0 or timesteps.shape[0] != img.shape[0]:
            return False
        return (
            self._prepare_batched_cfg_text(
                txt=txt,
                txt_ids=txt_ids,
                positive_extension=positive_extension,
                negative_extension=negative_extension,
            )
            is not None
        )

    def _run_batched_cfg(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
        cfg_scale: float,
    ) -> torch.Tensor:
        positive_pred, negative_pred = self._run_batched_cfg_branches(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
        )
        return negative_pred + cfg_scale * (positive_pred - negative_pred)

    def _run_batched_cfg_branches(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prepared_text = self._prepare_batched_cfg_text(
            txt=txt,
            txt_ids=txt_ids,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
        )
        if prepared_text is None:
            raise ValueError("Positive and negative Chroma conditioning are not compatible for batched CFG")
        positive_txt, negative_txt, positive_mask, negative_mask, batched_txt_ids = prepared_text

        batch_size = img.shape[0]
        batched_pred = self._forward_model(
            img=torch.cat((img, img), dim=0),
            img_ids=img_ids,
            txt=torch.cat((positive_txt, negative_txt), dim=0),
            txt_ids=batched_txt_ids,
            timesteps=torch.cat((timesteps, timesteps), dim=0),
            text_attention_mask=torch.cat((positive_mask, negative_mask), dim=0),
        )
        positive_pred = batched_pred[:batch_size]
        negative_pred = batched_pred[batch_size:]
        return positive_pred, negative_pred

    def _run_sequential_cfg(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
        cfg_scale: float,
    ) -> torch.Tensor:
        positive_pred, negative_pred = self._run_sequential_cfg_branches(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
        )
        return negative_pred + cfg_scale * (positive_pred - negative_pred)

    def _run_sequential_cfg_branches(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        positive_extension: RegionalPromptingExtension,
        negative_extension: RegionalPromptingExtension,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positive_pred = self._forward_model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            timesteps=timesteps,
            text_attention_mask=self._get_text_attention_mask(positive_extension, txt),
        )

        negative = negative_extension.regional_text_conditioning
        negative_txt = negative.t5_embeddings
        negative_pred = self._forward_model(
            img=img,
            img_ids=img_ids,
            txt=negative_txt,
            txt_ids=negative.t5_txt_ids,
            timesteps=timesteps,
            text_attention_mask=self._get_text_attention_mask(negative_extension, negative_txt),
        )
        return positive_pred, negative_pred

    def _forward_model(
        self,
        *,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        timesteps: torch.Tensor,
        text_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attention_mask = torch.cat(
            [
                text_attention_mask,
                torch.ones(img.shape[:2], dtype=torch.bool, device=img.device),
            ],
            dim=1,
        )

        txt_ids = self._normalize_position_ids(txt_ids)
        img_ids = self._normalize_position_ids(img_ids)

        log_first_forward = (
            is_chroma_numeric_diagnostics_enabled() and not self._numeric_diagnostics_first_forward_logged
        )
        if log_first_forward:
            log_tensor_fingerprint("transformer.forward0.img", img)
            log_tensor_fingerprint("transformer.forward0.txt", txt)
            log_tensor_fingerprint("transformer.forward0.img_ids", img_ids)
            log_tensor_fingerprint("transformer.forward0.txt_ids", txt_ids)
            log_tensor_fingerprint("transformer.forward0.timesteps", timesteps)
            log_tensor_fingerprint("transformer.forward0.text_attention_mask", text_attention_mask)
            log_tensor_fingerprint("transformer.forward0.attention_mask", attention_mask)

        prediction = self.model(
            hidden_states=img,
            encoder_hidden_states=txt,
            timestep=timesteps,
            img_ids=img_ids,
            txt_ids=txt_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]

        if log_first_forward:
            log_tensor_fingerprint("transformer.forward0.output", prediction)
            self._numeric_diagnostics_first_forward_logged = True

        return prediction
