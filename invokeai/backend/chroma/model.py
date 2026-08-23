import math
from typing import Any, cast

import torch
from diffusers import ChromaTransformer2DModel

from invokeai.backend.chroma.attention import (
    _chroma_cudnn_attention,
    _is_chroma_cudnn_backend_unavailable_error,
    _should_use_chroma_cudnn_attention,
)
from invokeai.backend.chroma.executor import InvokeAIChromaTransformerExecutor, _chroma_fp16_accumulation
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


class ChromaTransformerAdapter:
    """Adapt Diffusers' Chroma transformer to InvokeAI's rectified-flow denoiser interface."""

    def __init__(self, model: ChromaTransformer2DModel, *, model_input_dtype: torch.dtype | None = None):
        # Diffusers' generated type information omits Chroma's runtime modules and call operator.
        self.model = cast(Any, model)
        self._model_input_dtype = model_input_dtype
        self._executor = (
            InvokeAIChromaTransformerExecutor(model) if isinstance(model, ChromaTransformer2DModel) else None
        )
        self._batched_cfg_negative_extension: RegionalPromptingExtension | None = None
        self._batched_cfg_scale: list[float] | None = None
        self._batched_cfg_disabled = False
        self._cudnn_attention_disabled = False

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

        if (
            allow_batched
            and not self._batched_cfg_disabled
            and self._can_batch_cfg(
                img=img,
                txt=positive_txt,
                txt_ids=positive_txt_ids,
                timesteps=timesteps,
                positive_extension=positive_extension,
                negative_extension=negative_extension,
            )
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

    @staticmethod
    def _chroma_timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: int = 10000,
        time_factor: float = 1000.0,
    ) -> torch.Tensor:
        """Build the Chroma sinusoidal embedding on the input tensor's device.

        Keeping frequency/trigonometric evaluation device-local is intentional: moving
        this construction through a host/precomputed path changes FP16 rounding before
        the first transformer block.
        """
        t = time_factor * t
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        if torch.is_floating_point(t):
            embedding = embedding.to(t)
        return embedding

    def _build_chroma_cuda_input_vec(self, timesteps: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        """Build Chroma's 344x64 modulation input on the model device and dtype.

        This intentionally bypasses Diffusers' timestep helper. Reference-parity probes
        showed that the alternate construction changes ``input_vec`` before block 0.
        """
        mod_index_length = 344
        distill_timestep = self._chroma_timestep_embedding(timesteps.detach().clone(), 16).to(
            device=img.device, dtype=img.dtype
        )
        guidance = torch.zeros_like(timesteps)
        distil_guidance = self._chroma_timestep_embedding(guidance.detach().clone(), 16).to(
            device=img.device, dtype=img.dtype
        )
        modulation_index = self._chroma_timestep_embedding(torch.arange(mod_index_length, device=img.device), 32).to(
            device=img.device, dtype=img.dtype
        )
        modulation_index = modulation_index.unsqueeze(0).repeat(img.shape[0], 1, 1)
        timestep_guidance = (
            torch.cat([distill_timestep, distil_guidance], dim=1)
            .unsqueeze(1)
            .repeat(1, mod_index_length, 1)
            .to(device=img.device, dtype=img.dtype)
        )
        return torch.cat([timestep_guidance, modulation_index], dim=-1).to(device=img.device, dtype=img.dtype)

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
        prediction_dtype = img.dtype
        incoming_img_dtype = img.dtype
        if self._model_input_dtype is not None:
            img = img.to(dtype=self._model_input_dtype)
            img_ids = img_ids.to(dtype=self._model_input_dtype)
            txt = txt.to(dtype=self._model_input_dtype)
            txt_ids = txt_ids.to(dtype=self._model_input_dtype)

        attention_mask = torch.cat(
            [
                text_attention_mask,
                torch.ones(img.shape[:2], dtype=torch.bool, device=img.device),
            ],
            dim=1,
        )

        txt_ids = self._normalize_position_ids(txt_ids)
        img_ids = self._normalize_position_ids(img_ids)

        # Unpadded conditioning does not require an attention mask. Avoiding an all-true
        # mask also lets the runtime use the preferred fused attention implementation.
        model_attention_mask = None if bool(torch.all(text_attention_mask).item()) else attention_mask

        use_cudnn_attention = not self._cudnn_attention_disabled and _should_use_chroma_cudnn_attention(
            self.model,
            sampler_input_dtype=incoming_img_dtype,
            device_type=img.device.type,
        )
        chroma_cuda_input_vec: torch.Tensor | None = None
        if self._executor is not None:
            chroma_cuda_input_vec = self._build_chroma_cuda_input_vec(timesteps, img)

        def run_model_forward() -> torch.Tensor:
            if self._executor is not None:
                if chroma_cuda_input_vec is None:
                    raise RuntimeError("Missing Chroma modulation input")
                return self._executor(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    modulation_input=chroma_cuda_input_vec,
                    attention_mask=model_attention_mask,
                )

            output = self.model(
                hidden_states=img,
                encoder_hidden_states=txt,
                timestep=timesteps,
                img_ids=img_ids,
                txt_ids=txt_ids,
                attention_mask=model_attention_mask,
                return_dict=False,
            )[0]
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"Expected Chroma transformer tensor output, got {type(output).__name__}")
            return output

        with _chroma_fp16_accumulation():
            if use_cudnn_attention:
                try:
                    with _chroma_cudnn_attention(self.model, enabled=True):
                        prediction = run_model_forward()
                except RuntimeError as error:
                    if not _is_chroma_cudnn_backend_unavailable_error(error):
                        raise
                    self._cudnn_attention_disabled = True
                    InvokeAILogger.get_logger(__name__).warning(
                        "Chroma cuDNN attention is unavailable for this input; retrying with the "
                        "previous attention backend for the rest of this denoise: %s",
                        error,
                    )
                    prediction = run_model_forward()
            else:
                prediction = run_model_forward()

        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"Expected Chroma transformer tensor output, got {type(prediction).__name__}")
        if prediction.dtype != prediction_dtype:
            prediction = prediction.to(dtype=prediction_dtype)
        return prediction
