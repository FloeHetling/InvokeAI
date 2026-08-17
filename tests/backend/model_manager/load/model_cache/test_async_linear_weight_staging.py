import pytest
import torch

from invokeai.backend.model_manager.load.model_cache.torch_module_autocast import (
    async_linear_weight_staging as staging_module,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.async_linear_weight_staging import (
    CudaAsyncLinearWeightStager,
    cuda_async_linear_weight_staging,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules import (
    custom_linear as custom_linear_module,
)
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.custom_modules.custom_linear import (
    CustomLinear,
)


def _prepare_custom_linear(linear: CustomLinear) -> None:
    # CustomLinear is normally produced by apply_custom_layers_to_model(), which initializes these mixin fields.
    linear._device_autocasting_enabled = True
    linear._patches_and_weights = []


def test_custom_linear_marks_an_async_staged_weight_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    linear = CustomLinear(4, 3, bias=False)
    _prepare_custom_linear(linear)
    input = torch.randn(2, 4)
    staged_weight = linear.weight.detach().clone()
    consumed: list[torch.Tensor] = []

    def fake_stage(tensor: torch.Tensor | None, _input: torch.Tensor) -> torch.Tensor | None:
        return staged_weight if tensor is linear.weight else None

    monkeypatch.setattr(custom_linear_module, "maybe_stage_tensor_for_input", fake_stage)
    monkeypatch.setattr(custom_linear_module, "mark_staged_tensor_consumed", consumed.append)

    output = linear(input)

    assert torch.equal(output, torch.nn.functional.linear(input, staged_weight))
    assert consumed == [staged_weight]


def test_linear_sidecar_path_suspends_async_staging() -> None:
    linear = CustomLinear(4, 3, bias=False)
    _prepare_custom_linear(linear)
    input = torch.randn(2, 4)

    class FailingStager:
        def try_stage(self, _tensor: torch.Tensor | None, _input: torch.Tensor) -> torch.Tensor | None:
            raise AssertionError("sidecar linear path must suspend async weight staging")

        def mark_consumed(self, _tensor: torch.Tensor | None) -> None:
            raise AssertionError("sidecar linear path must suspend async weight staging")

    token = staging_module._ACTIVE_STAGER.set(FailingStager())  # pyright: ignore[reportPrivateUsage]
    try:
        output = custom_linear_module.autocast_linear_forward_sidecar_patches(
            linear,
            input,
            patches_and_weights=[],
        )
    finally:
        staging_module._ACTIVE_STAGER.reset(token)  # pyright: ignore[reportPrivateUsage]

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for persistent staging integration")
def test_cuda_async_linear_weight_staging_reuses_persistent_slots_without_corruption() -> None:
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
        for _ in range(12):
            for linear, input in zip(linears, inputs, strict=True):
                actual.append(linear(input))

    torch.cuda.synchronize(device)
    for index, output in enumerate(actual):
        assert torch.equal(output, expected[index % len(expected)])
    assert stager.stats.synchronous_fallbacks == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for persistent staging integration")
def test_cuda_staging_slot_is_reserved_until_the_staged_tensor_is_consumed() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    stager = CudaAsyncLinearWeightStager(device)
    input = torch.randn(1, 1024, device=device, dtype=torch.float16)
    weights = [torch.randn(1024, 1024, dtype=torch.float16) for _ in range(3)]

    first = stager.try_stage(weights[0], input)
    second = stager.try_stage(weights[1], input)
    third = stager.try_stage(weights[2], input)

    assert first is not None
    assert second is not None
    assert third is None
    assert stager.stats.synchronous_fallbacks == 1

    stager.mark_consumed(first)
    stager.mark_consumed(second)
    stager.close()
