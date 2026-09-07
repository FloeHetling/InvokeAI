"""Chroma-specific FLUX ControlNet residency helpers."""

import torch

from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.model_manager.load.load_base import LoadedModel
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.async_linear_weight_staging import (
    suspend_cuda_async_linear_weight_staging,
)


class ChromaInstantXControlNetExtension(InstantXControlNetExtension):
    """Run InstantX ControlNet in a phase-swapped model-cache residency window.

    Chroma1-HD and the FLUX Union ControlNet do not fit comfortably in VRAM together
    on common 16 GB cards. Keeping only a LoadedModel handle lets ModelCache evict
    the previously used large model before loading the next phase.
    """

    def __init__(
        self,
        *,
        model_info: LoadedModel,
        controlnet_cond: torch.Tensor,
        instantx_control_mode: torch.Tensor | None,
        weight: float | list[float],
        begin_step_percent: float,
        end_step_percent: float,
    ) -> None:
        model = model_info.model
        if not isinstance(model, InstantXControlNetFlux):
            raise TypeError(
                "Chroma ControlNet compatibility mode currently supports only InstantX FLUX ControlNet models, "
                f"got {type(model).__name__}"
            )
        super().__init__(
            model=model,
            controlnet_cond=controlnet_cond,
            instantx_control_mode=instantx_control_mode,
            weight=weight,
            begin_step_percent=begin_step_percent,
            end_step_percent=end_step_percent,
        )
        self._model_info = model_info

    def run_controlnet(
        self,
        timestep_index: int,
        total_num_timesteps: int,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        txt: torch.Tensor,
        txt_ids: torch.Tensor,
        y: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: torch.Tensor | None,
    ) -> ControlNetFluxOutput:
        # Avoid swapping a multi-GB model in for disabled schedule regions or zero weight.
        weight = self._get_weight(timestep_index=timestep_index, total_num_timesteps=total_num_timesteps)
        if weight < 1e-6:
            return ControlNetFluxOutput(single_block_residuals=None, double_block_residuals=None)

        # The Chroma async stager owns persistent FP16 buffers. InstantX executes in
        # BF16; allowing it to claim those slots would disable the stager when Chroma
        # switches back to FP16. Keep ControlNet on the normal model-cache/autocast path.
        with suspend_cuda_async_linear_weight_staging():
            with self._model_info.model_on_device() as (_cached_weights, loaded_controlnet):
                if loaded_controlnet is not self._model:
                    raise RuntimeError(
                        "Chroma ControlNet phase-swap handle returned an unexpected ControlNet instance"
                    )
                return super().run_controlnet(
                    timestep_index=timestep_index,
                    total_num_timesteps=total_num_timesteps,
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=y,
                    timesteps=timesteps,
                    guidance=guidance,
                )
