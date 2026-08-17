"""Tests for the FLUX VAE decode path's handling of the diffusers ``AutoencoderKL`` config.

``shift_factor`` is optional on ``AutoencoderKL``. The FLUX VAE sets one, but a plain SD-style
config leaves it ``None``, and the decode used to add it to the latents unconditionally.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

from invokeai.app.invocations.flux_vae_decode import FluxVaeDecodeInvocation
from invokeai.app.invocations.flux_vae_encode import FluxVaeEncodeInvocation
from invokeai.backend.util.vae_working_memory import estimate_vae_working_memory_flux


def _loaded_vae(shift_factor: float | None, scaling_factor: float = 0.3611) -> MagicMock:
    """A LoadedModel wrapping a diffusers AutoencoderKL whose config may lack a shift factor."""
    vae = MagicMock(spec=AutoencoderKL)
    param = torch.zeros(1, dtype=torch.float32)
    vae.parameters.side_effect = lambda: iter([param])
    config = SimpleNamespace(scaling_factor=scaling_factor)
    if shift_factor is not None:
        config.shift_factor = shift_factor
    vae.config = config
    # decode() returns a tuple when return_dict=False; one 3-channel image is enough.
    vae.decode.return_value = (torch.zeros(1, 3, 8, 8),)

    vae_info = MagicMock()
    vae_info.model = vae
    vae_info.compute_device = torch.device("cpu")
    working_mem_requests: list[int | None] = []

    @contextmanager
    def _on_device(working_mem_bytes=None):
        working_mem_requests.append(working_mem_bytes)
        yield (None, vae)

    vae_info.model_on_device = _on_device
    vae_info.working_mem_requests = working_mem_requests
    return vae_info


@pytest.mark.parametrize("shift_factor", [None, 0.1159])
def test_decode_handles_a_config_with_or_without_a_shift_factor(shift_factor: float | None) -> None:
    """A config lacking `shift_factor` used to raise `TypeError: unsupported operand ... NoneType`."""
    vae_info = _loaded_vae(shift_factor)
    latents = torch.ones(1, 16, 1, 1)

    image = FluxVaeDecodeInvocation._vae_decode(
        FluxVaeDecodeInvocation.model_construct(), vae_info=vae_info, latents=latents
    )

    assert image.size == (8, 8)
    # The shift is applied only when the config carries one; otherwise the latents are scale-only.
    passed = vae_info.model.decode.call_args.args[0]
    expected = latents / 0.3611 + (shift_factor or 0.0)
    assert torch.allclose(passed, expected)
    assert vae_info.working_mem_requests == [
        estimate_vae_working_memory_flux(operation="decode", image_tensor=latents, vae=vae_info.model)
    ]


def test_decode_treats_an_explicit_none_shift_factor_as_no_shift() -> None:
    """diffusers sets the attribute to None rather than omitting it when it is not configured."""
    vae_info = _loaded_vae(None)
    vae_info.model.config.shift_factor = None
    latents = torch.ones(1, 16, 1, 1)

    FluxVaeDecodeInvocation._vae_decode(FluxVaeDecodeInvocation.model_construct(), vae_info=vae_info, latents=latents)

    passed = vae_info.model.decode.call_args.args[0]
    assert torch.allclose(passed, latents / 0.3611)


@pytest.mark.parametrize("shift_factor", [None, 0.1159])
def test_encode_supports_diffusers_autoencoder_kl(shift_factor: float | None) -> None:
    vae = MagicMock(spec=AutoencoderKL)
    vae.parameters.side_effect = lambda: iter([torch.zeros(1, dtype=torch.float32)])
    vae.config = SimpleNamespace(scaling_factor=0.3611, shift_factor=shift_factor)
    sampled_latents = torch.ones(1, 16, 1, 1)
    vae.encode.return_value.latent_dist.sample.return_value = sampled_latents

    vae_info = _loaded_vae(shift_factor)
    vae_info.model = vae
    working_mem_requests: list[int | None] = []

    @contextmanager
    def _on_device(working_mem_bytes=None):
        working_mem_requests.append(working_mem_bytes)
        yield (None, vae)

    vae_info.model_on_device = _on_device
    image_tensor = torch.zeros(1, 3, 8, 8)

    latents = FluxVaeEncodeInvocation.vae_encode(vae_info, image_tensor)

    expected = (sampled_latents - (shift_factor or 0.0)) * 0.3611
    assert torch.allclose(latents, expected)
    assert working_mem_requests == [
        estimate_vae_working_memory_flux(operation="encode", image_tensor=image_tensor, vae=vae)
    ]
