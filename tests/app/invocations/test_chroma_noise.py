from unittest.mock import MagicMock

import torch

from invokeai.app.invocations.chroma_denoise import ChromaDenoiseInvocation
from invokeai.backend.chroma.sampling_utils import get_chroma_noise


def test_chroma_noise_matches_cpu_float32_reference_before_cast() -> None:
    seed = 3920626376
    actual = get_chroma_noise(
        num_samples=1,
        height=1024,
        width=1024,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        seed=seed,
    )
    expected = torch.randn(
        (1, 16, 128, 128),
        device="cpu",
        dtype=torch.float32,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(torch.bfloat16)

    assert torch.equal(actual, expected)


def test_chroma_noise_is_not_the_legacy_cpu_float16_stream() -> None:
    seed = 3920626376
    actual = get_chroma_noise(
        num_samples=1,
        height=64,
        width=64,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        seed=seed,
    )
    legacy = torch.randn(
        actual.shape,
        device="cpu",
        dtype=torch.float16,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).to(torch.bfloat16)

    assert not torch.equal(actual, legacy)


def test_chroma_beta_noise_remains_float32_for_the_sampler_state() -> None:
    invocation = ChromaDenoiseInvocation.model_construct(
        width=64,
        height=64,
        seed=3920626376,
        scheduler="euler_cfg_pp_beta",
        noise=None,
    )

    noise = invocation._prepare_noise_tensor(MagicMock(), torch.bfloat16, torch.device("cpu"))

    assert noise.dtype is torch.float32


def test_chroma_euler_noise_uses_the_requested_sampler_dtype() -> None:
    invocation = ChromaDenoiseInvocation.model_construct(
        width=64,
        height=64,
        seed=3920626376,
        scheduler="euler",
        noise=None,
    )

    noise = invocation._prepare_noise_tensor(MagicMock(), torch.bfloat16, torch.device("cpu"))

    assert noise.dtype is torch.bfloat16
