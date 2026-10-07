import torch

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput


def test_controlnet_apply_weight_preserves_repeated_tensor_aliasing() -> None:
    first = torch.tensor([2.0])
    second = torch.tensor([4.0])
    output = ControlNetFluxOutput(
        single_block_residuals=[first, first, second],
        double_block_residuals=[first, second, second],
    )

    output.apply_weight(0.5)

    assert output.single_block_residuals is not None
    assert output.double_block_residuals is not None
    assert output.single_block_residuals[0] is output.single_block_residuals[1]
    assert output.single_block_residuals[0] is output.double_block_residuals[0]
    assert output.single_block_residuals[2] is output.double_block_residuals[1]
    assert output.double_block_residuals[1] is output.double_block_residuals[2]
    assert torch.equal(output.single_block_residuals[0], torch.tensor([1.0]))
    assert torch.equal(output.single_block_residuals[2], torch.tensor([2.0]))
    assert output.single_block_residuals[0] is not first
    assert output.single_block_residuals[2] is not second


def test_controlnet_apply_unit_weight_keeps_original_tensors() -> None:
    residual = torch.tensor([3.0])
    output = ControlNetFluxOutput(single_block_residuals=[residual, residual], double_block_residuals=None)

    output.apply_weight(1.0)

    assert output.single_block_residuals is not None
    assert output.single_block_residuals[0] is residual
    assert output.single_block_residuals[1] is residual
