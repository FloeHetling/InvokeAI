from contextlib import contextmanager
from typing import Any, cast

import torch
from diffusers import ChromaTransformer2DModel


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


def _is_chroma_cudnn_backend_unavailable_error(error: RuntimeError) -> bool:
    """Return whether a RuntimeError indicates that cuDNN SDPA cannot serve this input.

    Do not treat arbitrary CUDA/model failures as an attention-backend compatibility
    issue. The caller may retry only these known "no supported plan/kernel" failures.
    """
    message = str(error).lower()
    unavailable_markers = (
        "no available kernel",
        "no valid execution plans support the graph",
        "no execution plans support the graph",
        "cudnn_status_not_supported",
        "cudnn_status_arch_mismatch",
    )
    return any(marker in message for marker in unavailable_markers)
