import hashlib
import os

import torch

from invokeai.backend.util.logging import InvokeAILogger

_CHROMA_EXPERIMENT_ENV = "CH_EXPERIMENT"
_FP32_STATE_EXPERIMENT = "fp32_state"


def get_chroma_experiment() -> str | None:
    value = os.getenv(_CHROMA_EXPERIMENT_ENV, "").strip().lower()
    return value or None


def is_chroma_numeric_diagnostics_enabled() -> bool:
    """Enable tensor fingerprints for any active Chroma experiment."""
    return get_chroma_experiment() is not None


def should_use_fp32_sampler_state() -> bool:
    return get_chroma_experiment() == _FP32_STATE_EXPERIMENT


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().contiguous().cpu().reshape(-1)
    raw_bytes = cpu_tensor.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw_bytes).hexdigest()


def log_tensor_fingerprint(label: str, tensor: torch.Tensor) -> None:
    """Log a deterministic fingerprint without changing the tensor used by inference."""
    if not is_chroma_numeric_diagnostics_enabled():
        return

    logger = InvokeAILogger.get_logger(__name__)
    detached = tensor.detach()
    original_device = str(detached.device)
    cpu_tensor = detached.contiguous().cpu()
    numeric = cpu_tensor.float().reshape(-1)

    if numeric.numel() == 0:
        minimum = maximum = mean = std = float("nan")
        finite_count = 0
    else:
        finite_mask = torch.isfinite(numeric)
        finite_count = int(finite_mask.sum().item())
        if finite_count == 0:
            minimum = maximum = mean = std = float("nan")
        else:
            finite_values = numeric[finite_mask]
            minimum = finite_values.min().item()
            maximum = finite_values.max().item()
            mean = finite_values.mean().item()
            std = finite_values.std(unbiased=False).item()

    logger.info(
        "Chroma experiment diagnostics tensor=%s shape=%s dtype=%s device=%s numel=%d finite=%d "
        "sha256=%s min=%.9g max=%.9g mean=%.9g std=%.9g",
        label,
        tuple(cpu_tensor.shape),
        cpu_tensor.dtype,
        original_device,
        cpu_tensor.numel(),
        finite_count,
        _tensor_sha256(cpu_tensor),
        minimum,
        maximum,
        mean,
        std,
    )
