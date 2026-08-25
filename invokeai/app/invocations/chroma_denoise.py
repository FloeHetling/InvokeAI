import math
from contextlib import ExitStack
from typing import Optional

import torch
from diffusers import ChromaTransformer2DModel
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import invocation
from invokeai.app.invocations.fields import FluxConditioningField, InputField
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.chroma.denoise import denoise_euler_cfg_pp
from invokeai.backend.chroma.model import ChromaTransformerAdapter
from invokeai.backend.chroma.sampling_utils import get_chroma_noise
from invokeai.backend.chroma.schedulers import (
    CHROMA_SCHEDULER_LABELS,
    CHROMA_SCHEDULER_NAME_VALUES,
    get_chroma_beta_schedule,
)
from invokeai.backend.flux.denoise import denoise
from invokeai.backend.flux.extensions.regional_prompting_extension import RegionalPromptingExtension
from invokeai.backend.flux.sampling_utils import (
    clip_timestep_schedule_fractional,
    generate_img_ids,
    get_schedule,
    pack,
    unpack,
)
from invokeai.backend.flux.schedulers import FLUX_SCHEDULER_MAP
from invokeai.backend.flux.text_conditioning import FluxTextConditioning
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ChromaConditioningInfo
from invokeai.backend.util.devices import TorchDevice


@invocation(
    "chroma_denoise",
    title="Chroma Denoise",
    tags=["image", "latents", "chroma"],
    category="latents",
    version="1.0.0",
)
class ChromaDenoiseInvocation(FluxDenoiseInvocation):
    """Run Chroma denoising with T5 attention masks and Chroma-specific guidance modes."""

    cfg_scale: float | list[float] = InputField(default=3.0, description="Classifier-free guidance scale")
    num_steps: int = InputField(default=40, gt=0, description="Number of Chroma denoising steps")
    scheduler: CHROMA_SCHEDULER_NAME_VALUES = InputField(  # type: ignore[assignment]
        default="euler",
        description="Scheduler used for Chroma rectified-flow denoising",
        ui_choice_labels=CHROMA_SCHEDULER_LABELS,
    )
    guidance: float = InputField(
        default=0.0,
        description="Unused by Chroma; retained for graph field compatibility",
        ui_hidden=True,
    )

    @field_validator("scheduler", mode="before")
    @classmethod
    def _migrate_legacy_scheduler(cls, value: object) -> object:
        """Normalize the scheduler value persisted by early Chroma builds."""
        if value == "euler_cfg_pp":
            return "euler_cfg_pp_beta"
        return value

    def _run_diffusion(self, context: InvocationContext) -> torch.Tensor:
        unsupported = {
            "Control LoRA": self.control_lora,
            "Redux": self.redux_conditioning,
            "Fill conditioning": self.fill_conditioning,
            "ControlNet": self.control,
            "ControlNet VAE": self.controlnet_vae,
            "IP-Adapter": self.ip_adapter,
            "Kontext": self.kontext_conditioning,
        }
        enabled = [name for name, value in unsupported.items() if value is not None]
        if enabled:
            raise ValueError(f"Chroma does not support these FLUX-only inputs: {', '.join(enabled)}")
        if self.transformer.loras:
            raise ValueError("Chroma transformer LoRA patches are not supported")
        if self.dype_preset != "off" or self.dype_scale is not None or self.dype_exponent is not None:
            raise ValueError("Chroma does not support FLUX DyPE settings")

        inference_dtype = torch.bfloat16
        transformer_dtype = torch.float16
        device = TorchDevice.choose_torch_device()
        fp32_sampler_state = self.scheduler == "euler_cfg_pp_beta"
        sampler_state_dtype = torch.float32 if fp32_sampler_state else inference_dtype
        init_latents = context.tensors.load(self.latents.latents_name) if self.latents else None
        if init_latents is not None:
            init_latents = init_latents.to(device=device, dtype=sampler_state_dtype)

        should_ignore_noise = init_latents is not None and not self.add_noise and self.denoise_mask is None
        noise: Optional[torch.Tensor]
        if should_ignore_noise:
            if init_latents is None:
                raise RuntimeError("Initial Chroma latents unexpectedly became unavailable")
            noise = None
            batch_size, _channels, latent_height, latent_width = init_latents.shape
        else:
            noise = self._prepare_noise_tensor(context, inference_dtype, device)
            if noise is None:
                raise RuntimeError("Noise was not prepared for Chroma denoising")
            batch_size, _channels, latent_height, latent_width = noise.shape

        packed_height = latent_height // 2
        packed_width = latent_width // 2
        image_seq_len = packed_height * packed_width

        positive = self._load_chroma_conditioning(context, self.positive_text_conditioning, transformer_dtype, device)
        negative = (
            self._load_chroma_conditioning(context, self.negative_text_conditioning, transformer_dtype, device)
            if self.negative_text_conditioning is not None
            else None
        )
        positive_extension = RegionalPromptingExtension.from_text_conditioning(positive, [], image_seq_len)
        negative_extension = (
            RegionalPromptingExtension.from_text_conditioning(negative, [], image_seq_len) if negative else None
        )
        # Chroma prompt conditioning is unpadded. Preserve that fact instead of carrying
        # an all-true CUDA mask into every transformer call; batched CFG recreates a mask
        # only when combining branches of different lengths requires padding.
        positive_extension.regional_text_conditioning.attention_mask = None
        if negative_extension is not None:
            negative_extension.regional_text_conditioning.attention_mask = None

        transformer_config = context.models.get_config(self.transformer.transformer)
        if transformer_config.base is not BaseModelType.Chroma or transformer_config.type is not ModelType.Main:
            raise ValueError("The selected transformer is not a Chroma main model")

        if self.scheduler == "euler_cfg_pp_beta":
            timesteps = get_chroma_beta_schedule(self.num_steps)
        else:
            timesteps = get_schedule(self.num_steps, image_seq_len=image_seq_len, shift=True)
        timesteps = clip_timestep_schedule_fractional(timesteps, self.denoising_start, self.denoising_end)

        scheduler = None
        # The shared native Euler loop consumes the already shifted schedule exactly as
        # Chroma's reference pipeline does. Passing its terminal zero through the
        # Diffusers Euler scheduler would append a second zero and waste one full CFG pass.
        if self.scheduler not in {"euler", "euler_cfg_pp_beta"} and self.scheduler in FLUX_SCHEDULER_MAP:
            scheduler = FLUX_SCHEDULER_MAP[self.scheduler](num_train_timesteps=1000)  # type: ignore[call-arg]

        if init_latents is not None:
            if self.add_noise:
                if noise is None:
                    raise RuntimeError("Noise was not prepared for noisy Chroma img2img")
                first_timestep = timesteps[0]
                latents = first_timestep * noise + (1.0 - first_timestep) * init_latents
            else:
                latents = init_latents
        else:
            if self.denoising_start > 1e-5:
                raise ValueError("denoising_start must be 0 when initial latents are not provided")
            if noise is None:
                raise RuntimeError("Noise was not prepared for Chroma txt2img")
            latents = noise

        if len(timesteps) <= 1:
            return latents

        inpaint_mask = self._prep_inpaint_mask(context, latents)
        image_ids = generate_img_ids(
            h=latent_height,
            w=latent_width,
            batch_size=batch_size,
            device=latents.device,
            dtype=latents.dtype,
        )

        packed_init_latents = pack(init_latents) if init_latents is not None else None
        packed_mask = pack(inpaint_mask) if inpaint_mask is not None else None
        packed_noise = pack(noise) if noise is not None else None
        packed_latents = pack(latents)

        inpaint_extension = None
        if packed_mask is not None:
            if packed_init_latents is None or packed_noise is None:
                raise ValueError("Chroma inpainting requires both initial latents and noise")
            inpaint_extension = RectifiedFlowInpaintExtension(
                init_latents=packed_init_latents,
                inpaint_mask=packed_mask,
                noise=packed_noise,
            )

        cfg_scale = self.prep_cfg_scale(
            cfg_scale=self.cfg_scale,
            timesteps=timesteps,
            cfg_scale_start_step=self.cfg_scale_start_step,
            cfg_scale_end_step=self.cfg_scale_end_step,
        )

        with ExitStack() as exit_stack:
            _cached_weights, transformer = exit_stack.enter_context(
                context.models.load(self.transformer.transformer).model_on_device()
            )
            if not isinstance(transformer, ChromaTransformer2DModel):
                raise TypeError(f"Expected ChromaTransformer2DModel, got {type(transformer).__name__}")

            adapter = ChromaTransformerAdapter(transformer, model_input_dtype=transformer_dtype)
            sequential_guidance = context.config.get().sequential_guidance
            if self.scheduler == "euler_cfg_pp_beta":
                if negative_extension is None:
                    raise ValueError("Negative text conditioning is required for Chroma Euler CFG++")
                if sequential_guidance:
                    context.logger.info(
                        "Chroma CFG++: sequential positive/negative guidance enabled by server setting."
                    )
                else:
                    context.logger.info("Chroma CFG++: batched positive/negative forward enabled.")

                packed_latents = denoise_euler_cfg_pp(
                    model=adapter,
                    img=packed_latents,
                    img_ids=image_ids,
                    positive_extension=positive_extension,
                    negative_extension=negative_extension,
                    timesteps=timesteps,
                    cfg_scale=cfg_scale,
                    step_callback=self._build_step_callback(context),
                    inpaint_extension=inpaint_extension,
                    allow_batched_cfg=not sequential_guidance,
                    model_input_dtype=(transformer_dtype if transformer_dtype != sampler_state_dtype else None),
                )
            else:
                denoise_cfg_scale = cfg_scale
                denoise_negative_extension = negative_extension
                has_guided_steps = any(not math.isclose(scale, 1.0) for scale in cfg_scale)
                if has_guided_steps and negative_extension is not None and not sequential_guidance:
                    # Reuse InvokeAI's existing global guidance policy: the default is parallel/batched guidance,
                    # while sequential_guidance=true is the low-memory opt-out. The adapter owns the Chroma-specific
                    # batch assembly and OOM fallback; the shared FLUX loop therefore sees CFG=1 and performs one
                    # model call per scheduler step.
                    adapter.enable_batched_cfg(negative_extension=negative_extension, cfg_scale=cfg_scale)
                    denoise_cfg_scale = [1.0] * len(cfg_scale)
                    denoise_negative_extension = None
                    context.logger.info("Chroma CFG: batched positive/negative forward enabled.")
                elif has_guided_steps and sequential_guidance:
                    context.logger.info("Chroma CFG: sequential guidance enabled by server setting.")

                packed_latents = denoise(
                    model=adapter,  # type: ignore[arg-type]
                    img=packed_latents,
                    img_ids=image_ids,
                    pos_regional_prompting_extension=positive_extension,
                    neg_regional_prompting_extension=denoise_negative_extension,
                    timesteps=timesteps,
                    step_callback=self._build_step_callback(context),
                    guidance=0.0,
                    cfg_scale=denoise_cfg_scale,
                    inpaint_extension=inpaint_extension,
                    controlnet_extensions=[],
                    pos_ip_adapter_extensions=[],
                    neg_ip_adapter_extensions=[],
                    img_cond=None,
                    scheduler=scheduler,
                )

        return unpack(packed_latents.float(), self.height, self.width)

    def _prepare_noise_tensor(
        self, context: InvocationContext, inference_dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        # Chroma noise is generated in CPU float32 before casting. Generating randn
        # directly in float16 changes the random stream for the same seed.
        target_dtype = torch.float32 if self.scheduler == "euler_cfg_pp_beta" else inference_dtype
        if self.noise is not None:
            return super()._prepare_noise_tensor(context, target_dtype, device)

        return get_chroma_noise(
            num_samples=1,
            height=self.height,
            width=self.width,
            device=device,
            dtype=target_dtype,
            seed=self.seed,
        )

    @staticmethod
    def _load_chroma_conditioning(
        context: InvocationContext,
        conditioning_field: FluxConditioningField | list[FluxConditioningField],
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[FluxTextConditioning]:
        fields = [conditioning_field] if isinstance(conditioning_field, FluxConditioningField) else conditioning_field
        result: list[FluxTextConditioning] = []
        for field in fields:
            if field.mask is not None:
                raise ValueError("Regional prompt masks are not supported by Chroma")
            data = context.conditioning.load(field.conditioning_name)
            if len(data.conditionings) != 1 or not isinstance(data.conditionings[0], ChromaConditioningInfo):
                raise TypeError("Expected Chroma conditioning data")
            conditioning = data.conditionings[0].to(device=device, dtype=dtype)
            dummy_pooled = torch.zeros((conditioning.prompt_embeds.shape[0], 768), dtype=dtype, device=device)
            result.append(
                FluxTextConditioning(
                    t5_embeddings=conditioning.prompt_embeds,
                    clip_embeddings=dummy_pooled,
                    mask=None,
                    attention_mask=conditioning.prompt_attention_mask,
                )
            )
        return result
