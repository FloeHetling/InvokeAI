import math
from collections.abc import Callable

import torch
from tqdm import tqdm

from invokeai.backend.chroma.model import ChromaTransformerAdapter
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusers_pipeline import PipelineIntermediateState
from invokeai.backend.util.devices import TorchDevice


def euler_cfg_pp_step(
    img: torch.Tensor,
    positive_pred: torch.Tensor,
    negative_pred: torch.Tensor,
    sigma: float,
    sigma_next: float,
    cfg_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one deterministic Euler CFG++ step for rectified-flow predictions.

    Chroma predicts rectified-flow velocity, so x0 = x - sigma * v. CFG is
    applied in x0 space, while the transition to the next sigma follows the
    unconditional noise direction.

    Returns ``(next_img, guided_denoised)``. The latter is the x0 estimate used for
    previews and for the terminal sigma=0 step.
    """
    if sigma <= 0.0:
        raise ValueError("Euler CFG++ requires the current sigma to be greater than zero")

    sigma_tensor = img.new_tensor(sigma)
    positive_denoised = img - sigma_tensor * positive_pred
    negative_denoised = img - sigma_tensor * negative_pred
    guided_denoised = negative_denoised + cfg_scale * (positive_denoised - negative_denoised)

    # At terminal sigma=0, the guided denoised estimate is the final sample.
    # Avoid computing a noise direction that would be multiplied by zero.
    if math.isclose(sigma_next, 0.0, abs_tol=1e-12):
        return guided_denoised, guided_denoised

    # For this rectified-flow parameterization, alpha(sigma) = 1 - sigma.
    # Deterministic CFG++ keeps the transition on the unconditional noise direction.
    alpha_s = img.new_tensor(1.0 - sigma)
    alpha_next = img.new_tensor(1.0 - sigma_next)
    sigma_next_tensor = img.new_tensor(sigma_next)
    negative_noise = (img - alpha_s * negative_denoised) / sigma_tensor
    next_img = alpha_next * guided_denoised + sigma_next_tensor * negative_noise
    return next_img, guided_denoised


@torch.no_grad()
def denoise_euler_cfg_pp(
    *,
    model: ChromaTransformerAdapter,
    img: torch.Tensor,
    img_ids: torch.Tensor,
    positive_extension: RegionalPromptingExtension,
    negative_extension: RegionalPromptingExtension,
    timesteps: list[float],
    cfg_scale: list[float],
    step_callback: Callable[[PipelineIntermediateState], None],
    inpaint_extension: RectifiedFlowInpaintExtension | None,
    allow_batched_cfg: bool,
    model_input_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Denoise Chroma latents with deterministic Euler CFG++."""
    total_steps = len(timesteps) - 1
    if total_steps <= 0:
        return img
    if len(cfg_scale) < total_steps:
        raise ValueError("CFG scale schedule is shorter than the Chroma timestep schedule")

    iterator = enumerate(zip(timesteps[:-1], timesteps[1:], strict=True))
    for step_index, (sigma, sigma_next) in tqdm(
        iterator,
        total=total_steps,
        desc=f"Denoising{TorchDevice.get_session_device_label()}",
    ):
        model_img = img if model_input_dtype is None else img.to(dtype=model_input_dtype)
        model_img_ids = img_ids if model_input_dtype is None else img_ids.to(dtype=model_input_dtype)
        timestep_vec = torch.full((img.shape[0],), sigma, dtype=img.dtype, device=img.device)

        positive_pred, negative_pred = model.predict_cfg_branches(
            img=model_img,
            img_ids=model_img_ids,
            timesteps=timestep_vec,
            positive_extension=positive_extension,
            negative_extension=negative_extension,
            allow_batched=allow_batched_cfg,
        )

        if model_input_dtype is not None:
            positive_pred = positive_pred.to(dtype=img.dtype)
            negative_pred = negative_pred.to(dtype=img.dtype)

        img, preview_img = euler_cfg_pp_step(
            img=img,
            positive_pred=positive_pred,
            negative_pred=negative_pred,
            sigma=sigma,
            sigma_next=sigma_next,
            cfg_scale=cfg_scale[step_index],
        )

        if inpaint_extension is not None:
            img = inpaint_extension.merge_intermediate_latents_with_init_latents(img, sigma_next)
            preview_img = inpaint_extension.merge_intermediate_latents_with_init_latents(preview_img, 0.0)

        step_callback(
            PipelineIntermediateState(
                step=step_index + 1,
                order=1,
                total_steps=total_steps,
                timestep=int(sigma),
                latents=preview_img,
            )
        )

    return img
