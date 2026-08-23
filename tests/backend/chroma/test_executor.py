from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import ChromaTransformer2DModel

from invokeai.backend.chroma.executor import _chroma_ada_layer_norm_zero, _chroma_fp16_accumulation
from invokeai.backend.chroma.model import ChromaTransformerAdapter


def test_chroma_transformer_adapter_configures_the_model_numeric_contract_without_patching_forwards(
    monkeypatch,
) -> None:
    # Having cuDNN available must not pin Chroma to the cuDNN-only SDPA backend. Some
    # supported Chroma paths reach attention in FP32, which cuDNN SDPA rejects.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: True)
    model = ChromaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        approximator_num_channels=8,
        approximator_hidden_dim=16,
        approximator_layers=1,
    )

    double_block = model.transformer_blocks[0]
    single_block = model.single_transformer_blocks[0]
    modules_with_invoke_owned_execution = (
        double_block,
        double_block.norm1,
        double_block.norm1_context,
        single_block.norm,
        model.norm_out,
    )
    assert all("forward" not in module.__dict__ for module in modules_with_invoke_owned_execution)

    ChromaTransformerAdapter(model)

    assert all("forward" not in module.__dict__ for module in modules_with_invoke_owned_execution)
    assert double_block.attn.norm_q.eps is None
    assert double_block.attn.norm_k.eps is None
    assert double_block.attn.norm_added_q.eps is None
    assert double_block.attn.norm_added_k.eps is None
    assert single_block.attn.norm_q.eps is None
    assert single_block.attn.norm_k.eps is None
    assert double_block.attn.processor._attention_backend is None
    assert single_block.attn.processor._attention_backend is None

    hidden_states = torch.randn(1, 3, 8)
    embedding = torch.randn(1, 6, 8)
    normalized, *_rest = _chroma_ada_layer_norm_zero(double_block.norm1, hidden_states, emb=embedding)
    shift, scale, *_unused = embedding.flatten(1, 2).chunk(6, dim=1)
    expected = torch.addcmul(shift[:, None], double_block.norm1.norm(hidden_states), 1 + scale[:, None])
    assert torch.equal(normalized, expected)


def test_chroma_adaln_preserves_fused_fp16_rounding() -> None:
    module = SimpleNamespace(emb=None, norm=lambda value: value)
    hidden_states = torch.tensor([[[0.1]]], dtype=torch.float16)
    embedding = torch.zeros((1, 6, 1), dtype=torch.float16)
    embedding[:, 0, :] = 0.1
    embedding[:, 1, :] = 0.5

    normalized, *_rest = _chroma_ada_layer_norm_zero(module, hidden_states, emb=embedding)
    shift, scale, *_unused = embedding.flatten(1, 2).chunk(6, dim=1)
    fused = torch.addcmul(shift[:, None], hidden_states, 1 + scale[:, None])
    decomposed = shift[:, None] + hidden_states * (1 + scale[:, None])

    assert torch.equal(normalized, fused)
    assert not torch.equal(normalized, decomposed)


def test_chroma_fp16_accumulation_context_enables_and_restores_runtime_flag() -> None:
    matmul_backend = torch.backends.cuda.matmul
    if not hasattr(matmul_backend, "allow_fp16_accumulation"):
        pytest.skip("This PyTorch build does not expose FP16 accumulation control")

    previous = matmul_backend.allow_fp16_accumulation
    try:
        matmul_backend.allow_fp16_accumulation = False
        with _chroma_fp16_accumulation():
            assert matmul_backend.allow_fp16_accumulation is True
        assert matmul_backend.allow_fp16_accumulation is False
    finally:
        matmul_backend.allow_fp16_accumulation = previous


def test_chroma_transformer_adapter_executes_real_model_without_calling_diffusers_forward() -> None:
    model = ChromaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        approximator_num_channels=64,
        approximator_hidden_dim=16,
        approximator_layers=1,
    )
    adapter = ChromaTransformerAdapter(model)
    model.forward = MagicMock(side_effect=AssertionError("Diffusers Chroma forward must not be called"))

    prediction = adapter._forward_model(
        img=torch.randn(1, 4, 4),
        img_ids=torch.zeros(4, 3),
        txt=torch.randn(1, 3, 12),
        txt_ids=torch.zeros(3, 3),
        timesteps=torch.tensor([0.5]),
        text_attention_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    assert prediction.shape == (1, 4, 4)
    assert torch.isfinite(prediction).all()
    model.forward.assert_not_called()
