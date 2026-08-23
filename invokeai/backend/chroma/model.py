import math
from contextlib import contextmanager
from typing import Any, cast

import torch
from diffusers import ChromaTransformer2DModel
from diffusers.models.transformers.transformer_chroma import (
    ChromaAdaLayerNormContinuousPruned,
    ChromaAdaLayerNormZeroPruned,
    ChromaAdaLayerNormZeroSinglePruned,
    ChromaSingleTransformerBlock,
    ChromaTransformerBlock,
)

from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.util.devices import TorchDevice
from invokeai.backend.util.logging import InvokeAILogger


def _chroma_ada_layer_norm_zero(
    module: ChromaAdaLayerNormZeroPruned,
    x: torch.Tensor,
    timestep: torch.Tensor | None = None,
    class_labels: torch.LongTensor | None = None,
    hidden_dtype: torch.dtype | None = None,
    emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if module.emb is not None:
        emb = module.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
    if emb is None:
        raise ValueError("Chroma adaptive normalization requires an embedding")
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.flatten(1, 2).chunk(6, dim=1)
    normalized = module.norm(x)
    normalized = torch.addcmul(shift_msa[:, None], normalized, 1 + scale_msa[:, None])
    return normalized, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _chroma_ada_layer_norm_zero_single(
    module: ChromaAdaLayerNormZeroSinglePruned,
    x: torch.Tensor,
    emb: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if emb is None:
        raise ValueError("Chroma adaptive normalization requires an embedding")
    shift_msa, scale_msa, gate_msa = emb.flatten(1, 2).chunk(3, dim=1)
    normalized = module.norm(x)
    normalized = torch.addcmul(shift_msa[:, None], normalized, 1 + scale_msa[:, None])
    return normalized, gate_msa


def _chroma_ada_layer_norm_continuous(
    module: ChromaAdaLayerNormContinuousPruned,
    x: torch.Tensor,
    emb: torch.Tensor,
) -> torch.Tensor:
    shift, scale = torch.chunk(emb.flatten(1, 2).to(x.dtype), 2, dim=1)
    normalized = module.norm(x)
    return torch.addcmul(shift[:, None, :], normalized, (1 + scale)[:, None, :])


def _chroma_double_block(
    block: ChromaTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run a Chroma double block with fused FP16 AdaLN modulation."""
    temb_img, temb_txt = temb[:, :6], temb[:, 6:]
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = _chroma_ada_layer_norm_zero(
        block.norm1, hidden_states, emb=temb_img
    )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _chroma_ada_layer_norm_zero(
        block.norm1_context, encoder_hidden_states, emb=temb_txt
    )
    joint_attention_kwargs = joint_attention_kwargs or {}
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

    attention_outputs = block.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        **joint_attention_kwargs,
    )
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
        ip_attn_output = None
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs
    else:
        raise ValueError(f"Unexpected Chroma attention output count: {len(attention_outputs)}")

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = torch.addcmul(
        shift_mlp[:, None],
        block.norm2(hidden_states),
        1 + scale_mlp[:, None],
    )
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * block.ff(norm_hidden_states)
    if ip_attn_output is not None:
        hidden_states = hidden_states + ip_attn_output

    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    norm_encoder_hidden_states = torch.addcmul(
        c_shift_mlp[:, None],
        block.norm2_context(encoder_hidden_states),
        1 + c_scale_mlp[:, None],
    )
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * block.ff_context(
        norm_encoder_hidden_states
    )
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _chroma_single_block(
    block: ChromaSingleTransformerBlock,
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    joint_attention_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run a Chroma single block with fused FP16 AdaLN modulation."""
    residual = hidden_states
    norm_hidden_states, gate = _chroma_ada_layer_norm_zero_single(block.norm, hidden_states, emb=temb)
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}

    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

    attn_output = block.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        **joint_attention_kwargs,
    )
    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    hidden_states = residual + gate.unsqueeze(1) * block.proj_out(hidden_states)
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return hidden_states


def _set_chroma_attention_contract(attention: Any) -> None:
    # Chroma uses the input dtype's RMSNorm epsilon instead of Diffusers' fixed 1e-6.
    for name in ("norm_q", "norm_k", "norm_added_q", "norm_added_k"):
        norm = getattr(attention, name, None)
        if norm is not None:
            norm.eps = None


class InvokeAIChromaTransformerExecutor:
    """Execute a Diffusers Chroma transformer's modules with InvokeAI-owned numerical semantics.

    Chroma FP16 parity depends on fused ``torch.addcmul`` modulation and dtype-local
    RMSNorm epsilon selection. Diffusers exposes the correct weights and module layout,
    but its public forward uses a different arithmetic decomposition. Keeping the
    execution here makes that dependency explicit and avoids mutating module ``forward``
    methods at runtime.
    """

    def __init__(self, model: ChromaTransformer2DModel):
        self.model = cast(Any, model)
        for block in self.model.transformer_blocks:
            _set_chroma_attention_contract(block.attn)
        for block in self.model.single_transformer_blocks:
            _set_chroma_attention_contract(block.attn)

    def __call__(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        modulation_input: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        hidden_states = self.model.x_embedder(hidden_states)
        pooled_temb = self.model.distilled_guidance_layer(modulation_input)
        encoder_hidden_states = self.model.context_embedder(encoder_hidden_states)

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.model.pos_embed(ids)

        img_offset = 3 * len(self.model.single_transformer_blocks)
        txt_offset = img_offset + 6 * len(self.model.transformer_blocks)
        for index_block, block in enumerate(self.model.transformer_blocks):
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            temb = torch.cat(
                (
                    pooled_temb[:, img_modulation : img_modulation + 6],
                    pooled_temb[:, text_modulation : text_modulation + 6],
                ),
                dim=1,
            )
            encoder_hidden_states, hidden_states = _chroma_double_block(
                block=block,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        for index_block, block in enumerate(self.model.single_transformer_blocks):
            start_idx = 3 * index_block
            hidden_states = _chroma_single_block(
                block=block,
                hidden_states=hidden_states,
                temb=pooled_temb[:, start_idx : start_idx + 3],
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
            )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
        hidden_states = _chroma_ada_layer_norm_continuous(self.model.norm_out, hidden_states, pooled_temb[:, -2:])
        output = self.model.proj_out(hidden_states)
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected Chroma executor tensor output, got {type(output).__name__}")
        return output


@contextmanager
def _chroma_fp16_accumulation() -> Any:
    matmul_backend = torch.backends.cuda.matmul
    if not hasattr(matmul_backend, "allow_fp16_accumulation"):
        yield
        return
    previous = matmul_backend.allow_fp16_accumulation
    matmul_backend.allow_fp16_accumulation = True
    try:
        yield
    finally:
        matmul_backend.allow_fp16_accumulation = previous


def _should_use_chroma_cudnn_attention(
    model: Any,
    *,
    sampler_input_dtype: torch.dtype,
    device_type: str,
) -> bool:
    """Use cuDNN only for the reference-compatible FP16 Chroma execution path."""
    if not isinstance(model, ChromaTransformer2DModel):
        return False
    if device_type != "cuda" or sampler_input_dtype != torch.float16:
        return False
    model_weight = getattr(getattr(model, "x_embedder", None), "weight", None)
    if model_weight is None or model_weight.dtype != torch.float16:
        return False
    return bool(cast(Any, torch.backends.cudnn).is_available())


@contextmanager
def _chroma_cudnn_attention(model: Any, *, enabled: bool) -> Any:
    """Temporarily use cuDNN attention for a capability-checked Chroma forward."""
    if not enabled:
        yield
        return

    previous_backends: list[tuple[Any, Any]] = []
    try:
        blocks = [*model.transformer_blocks, *model.single_transformer_blocks]
        for block in blocks:
            attention = block.attn
            processor = attention.processor
            previous_backends.append((processor, getattr(processor, "_attention_backend", None)))
            attention.set_attention_backend("_native_cudnn")
        yield
    finally:
        for processor, previous_backend in previous_backends:
            processor._attention_backend = previous_backend


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
        """Build the Chroma sinusoidal embedding on the input tensor's device."""
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
        """Build Chroma's modulation input on the model device."""
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

        use_cudnn_attention = _should_use_chroma_cudnn_attention(
            self.model,
            sampler_input_dtype=incoming_img_dtype,
            device_type=img.device.type,
        )
        chroma_cuda_input_vec: torch.Tensor | None = None
        if self._executor is not None:
            chroma_cuda_input_vec = self._build_chroma_cuda_input_vec(timesteps, img)

        with (
            _chroma_fp16_accumulation(),
            _chroma_cudnn_attention(self.model, enabled=use_cudnn_attention),
        ):
            if self._executor is not None:
                if chroma_cuda_input_vec is None:
                    raise RuntimeError("Missing Chroma modulation input")
                prediction = self._executor(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    modulation_input=chroma_cuda_input_vec,
                    attention_mask=model_attention_mask,
                )
            else:
                prediction = self.model(
                    hidden_states=img,
                    encoder_hidden_states=txt,
                    timestep=timesteps,
                    img_ids=img_ids,
                    txt_ids=txt_ids,
                    attention_mask=model_attention_mask,
                    return_dict=False,
                )[0]

        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"Expected Chroma transformer tensor output, got {type(prediction).__name__}")
        if prediction.dtype != prediction_dtype:
            prediction = prediction.to(dtype=prediction_dtype)
        return prediction
