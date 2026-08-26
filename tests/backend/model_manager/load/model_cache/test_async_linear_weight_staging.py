import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules import (
    custom_linear as custom_linear_module,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.async_linear_weight_staging import (
    cuda_async_linear_weight_staging,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)


def _prepare_custom_linear(linear: CustomLinear) -> None:
    # CustomLinear is normally produced by apply_custom_layers_to_model(), which initializes these mixin fields.
    linear._device_autocasting_enabled = True
    linear._patches_and_weights = []


def test_custom_linear_uses_an_async_staged_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    linear = CustomLinear(4, 3, bias=False)
    _prepare_custom_linear(linear)
    input = torch.randn(2, 4)
    staged_weight = linear.weight.detach().clone()

    def fake_stage(tensor: torch.Tensor | None, _input: torch.Tensor) -> torch.Tensor | None:
        return staged_weight if tensor is linear.weight else None

    monkeypatch.setattr(custom_linear_module, "maybe_stage_tensor_for_input", fake_stage)

    output = linear(input)

    assert torch.equal(output, torch.nn.functional.linear(input, staged_weight))


def test_custom_linear_patch_path_does_not_use_async_staging(monkeypatch: pytest.MonkeyPatch) -> None:
    linear = CustomLinear(4, 3, bias=False)
    _prepare_custom_linear(linear)
    input = torch.randn(2, 4)

    def fail_if_staged(_tensor: torch.Tensor | None, _input: torch.Tensor) -> torch.Tensor | None:
        raise AssertionError("patched linear path must not use async weight staging")

    monkeypatch.setattr(custom_linear_module, "maybe_stage_tensor_for_input", fail_if_staged)

    output = linear._autocast_forward(input, allow_async_staging=False)

    assert torch.equal(output, torch.nn.functional.linear(input, linear.weight, linear.bias))


def test_async_linear_weight_staging_context_is_a_noop_on_cpu() -> None:
    with cuda_async_linear_weight_staging(torch.device("cpu")) as stager:
        assert stager is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for async staging integration")
def test_cuda_async_linear_weight_staging_preserves_linear_output() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    linear = CustomLinear(1024, 1024, bias=False, dtype=torch.float16)
    _prepare_custom_linear(linear)
    input = torch.randn(2, 1024, device=device, dtype=torch.float16)

    baseline_weight = linear.weight.detach().to(device)
    expected = torch.nn.functional.linear(input, baseline_weight)
    torch.cuda.synchronize(device)
    del baseline_weight

    with cuda_async_linear_weight_staging(device) as stager:
        assert stager is not None
        actual = linear(input)

    assert stager.stats.staged_tensors == 1
    assert stager.stats.synchronous_fallbacks == 0
    assert torch.equal(actual, expected)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for async staging integration")
def test_cuda_async_linear_weight_staging_reuses_host_slots_without_corruption() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    linears = [CustomLinear(1024, 1024, bias=False, dtype=torch.float16) for _ in range(3)]
    for linear in linears:
        _prepare_custom_linear(linear)

    inputs = [torch.randn(2, 1024, device=device, dtype=torch.float16) for _ in linears]
    expected = [
        torch.nn.functional.linear(input, linear.weight.detach().to(device))
        for linear, input in zip(linears, inputs, strict=True)
    ]
    torch.cuda.synchronize(device)

    actual: list[torch.Tensor] = []
    with cuda_async_linear_weight_staging(device) as stager:
        assert stager is not None
        for _ in range(6):
            for linear, input in zip(linears, inputs, strict=True):
                actual.append(linear(input))

    assert stager.stats.staged_tensors == 18
