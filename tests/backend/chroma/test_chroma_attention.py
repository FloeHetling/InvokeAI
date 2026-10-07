import torch
from diffusers import ChromaTransformer2DModel

from invokeai.backend.chroma.attention import (
    _should_use_chroma_cudnn_attention,
)


def test_chroma_cudnn_attention_policy_is_scoped_to_fp16_cuda(monkeypatch) -> None:
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
    ).to(dtype=torch.float16)

    assert _should_use_chroma_cudnn_attention(
        model,
        sampler_input_dtype=torch.float16,
        device_type="cuda",
    )
    assert not _should_use_chroma_cudnn_attention(
        model,
        sampler_input_dtype=torch.bfloat16,
        device_type="cuda",
    )
    assert not _should_use_chroma_cudnn_attention(
        model,
        sampler_input_dtype=torch.float16,
        device_type="cpu",
    )

    model.to(dtype=torch.float32)
    assert not _should_use_chroma_cudnn_attention(
        model,
        sampler_input_dtype=torch.float16,
        device_type="cuda",
    )

    model.to(dtype=torch.float16)
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: False)
    assert not _should_use_chroma_cudnn_attention(
        model,
        sampler_input_dtype=torch.float16,
        device_type="cuda",
    )
