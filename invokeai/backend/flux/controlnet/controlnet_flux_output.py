from dataclasses import dataclass

import torch


@dataclass
class ControlNetFluxOutput:
    single_block_residuals: list[torch.Tensor] | None
    double_block_residuals: list[torch.Tensor] | None

    def apply_weight(self, weight: float) -> None:
        # InstantX expands a smaller set of ControlNet outputs to FLUX's 19 double +
        # 38 single block layout by repeating references. Weight each unique tensor only
        # once, then preserve that aliasing instead of materializing dozens of copies.
        if weight == 1.0:
            return

        weighted_by_source_id: dict[int, torch.Tensor] = {}

        def apply_to_list(residuals: list[torch.Tensor] | None) -> None:
            if residuals is None:
                return
            for index, residual in enumerate(residuals):
                source_id = id(residual)
                weighted = weighted_by_source_id.get(source_id)
                if weighted is None:
                    weighted = residual * weight
                    weighted_by_source_id[source_id] = weighted
                residuals[index] = weighted

        apply_to_list(self.single_block_residuals)
        apply_to_list(self.double_block_residuals)


def add_tensor_lists_elementwise(
    list1: list[torch.Tensor] | None, list2: list[torch.Tensor] | None
) -> list[torch.Tensor] | None:
    """Add two tensor lists elementwise that could be None."""
    if list1 is None and list2 is None:
        return None
    if list1 is None:
        return list2
    if list2 is None:
        return list1

    new_list: list[torch.Tensor] = []
    for list1_tensor, list2_tensor in zip(list1, list2, strict=True):
        new_list.append(list1_tensor + list2_tensor)
    return new_list


def add_controlnet_flux_outputs(
    controlnet_output_1: ControlNetFluxOutput, controlnet_output_2: ControlNetFluxOutput
) -> ControlNetFluxOutput:
    return ControlNetFluxOutput(
        single_block_residuals=add_tensor_lists_elementwise(
            controlnet_output_1.single_block_residuals, controlnet_output_2.single_block_residuals
        ),
        double_block_residuals=add_tensor_lists_elementwise(
            controlnet_output_1.double_block_residuals, controlnet_output_2.double_block_residuals
        ),
    )


def sum_controlnet_flux_outputs(
    controlnet_outputs: list[ControlNetFluxOutput],
) -> ControlNetFluxOutput:
    controlnet_output_sum = ControlNetFluxOutput(single_block_residuals=None, double_block_residuals=None)

    for controlnet_output in controlnet_outputs:
        controlnet_output_sum = add_controlnet_flux_outputs(controlnet_output_sum, controlnet_output)

    return controlnet_output_sum
