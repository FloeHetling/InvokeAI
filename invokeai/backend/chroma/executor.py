"""InvokeAI-owned numerical execution semantics for Diffusers Chroma modules.

Diffusers remains responsible for model construction, weights, and module definitions.
This module owns only the operations whose FP16 rounding/runtime semantics must remain
aligned with the reference Chroma execution path.
"""

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
    # Keep this modulation fused. In FP16, ``addcmul`` is not bit-equivalent to
    # multiplying and adding separately; the latter changes the first-block trajectory.
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


def _expand_chroma_attention_mask(attention_mask: torch.Tensor | None) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    return attention_mask[:, None, None, :] * attention_mask[:, None, :, None]


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
    # ``eps=None`` makes PyTorch choose the input dtype's epsilon. In FP16 this differs
    # materially from Diffusers' fixed 1e-6 and preserves the reference Chroma Q/K norms.
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
        attention_mask = _expand_chroma_attention_mask(attention_mask)

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
    """Match Chroma's reference FP16 GEMM accumulation policy for this forward only."""
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
