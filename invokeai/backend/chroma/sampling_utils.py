import math

import torch


def get_chroma_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.Tensor:
    """Generate Chroma initial noise using a CPU-float32 random stream before casting."""
    noise = torch.randn(
        num_samples,
        16,
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    return noise.to(device=device, dtype=dtype)
