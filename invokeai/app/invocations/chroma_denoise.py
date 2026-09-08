import math
from contextlib import ExitStack
from typing import Optional

import torch
from diffusers import ChromaTransformer2DModel
from PIL import Image
from pydantic import field_validator

from invokeai.app.invocations.baseinvocation import invocation
from invokeai.app.invocations.fields import FluxConditioningField, InputField
from invokeai.app.invocations.flux_controlnet import FluxControlNetField
from invokeai.app.invocations.flux_denoise import FluxDenoiseInvocation
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.backend.chroma.controlnet import ChromaInstantXControlNetExtension
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
from invokeai.backend.flux.text_conditioning import FluxReduxConditioning, FluxTextConditioning
from invokeai.backend.model_manager.load.model_cache.torch_module_autocast.async_linear_weight_staging import (
    cuda_async_linear_weight_staging,
)
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType
from invokeai.backend.rectified_flow.rectified_flow_inpaint_extension import RectifiedFlowInpaintExtension
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ChromaConditioningInfo
from invokeai.backend.util.devices import TorchDevice


def _prepare_chroma_controlnet_image(image: Image.Image) -> Image.Image:
    """Drop alpha without compositing so structural ControlNet RGB data is preserved."""
    return image.convert("RGB")


def _validate_chroma_controlnet_scheduler(scheduler: str) -> None:
    if scheduler not in {"euler", "euler_cfg_pp_beta"}:
        raise ValueError(
            "Chroma ControlNet compatibility mode currently supports only the Euler and Euler CFG++ (Beta) schedulers"
        )


@invocation(
    "chroma_denoise",
    title="Chroma Denoise",
    tags=["image", "latents", "chroma"],
    category="latents",
    version="1.2.0",
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
            "Fill conditioning": self.fill_conditioning,
            "IP-Adapter": self.ip_adapter,
            "Kontext": self.kontext_conditioning,
        }
        enabled = [name for name, value in unsupported.items() if value is not None]
        if enabled:
            raise ValueError(f"Chroma does not support these FLUX-only inputs: {', '.join(enabled)}")
        if self.control is None and self.controlnet_vae is not None:
            raise ValueError("controlnet_vae requires a ControlNet input")
        if self.control is not None:
            _validate_chroma_controlnet_scheduler(self.scheduler)
            if self.redux_conditioning is not None:
                raise ValueError("Chroma ControlNet compatibility mode does not yet support Redux conditioning")
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
        redux_conditionings: list[FluxReduxConditioning] = self._load_redux_conditioning(
            context=context,
            redux_cond_field=self.redux_conditioning,
            packed_height=packed_height,
            packed_width=packed_width,
            device=device,
            dtype=transformer_dtype,
        )
        if redux_conditionings:
            context.logger.info(
                f"Chroma Redux conditioning enabled with {len(redux_conditionings)} reference image(s)."
            )
        positive_extension = RegionalPromptingExtension.from_text_conditioning(
            positive,
            redux_conditionings,
            image_seq_len,
        )
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
            controlnet_extensions = self._prep_chroma_controlnet_extensions(
                context=context,
                latent_height=latent_height,
                latent_width=latent_width,
                dtype=inference_dtype,
                device=device,
            )
            controlnet_guidance = 3.5 if controlnet_extensions else 0.0
            if controlnet_extensions:
                context.logger.info(
                    f"Chroma FLUX ControlNet compatibility mode enabled with {len(controlnet_extensions)} "
                    f"ControlNet(s); side-model guidance={controlnet_guidance:.1f}; phase-swapped residency enabled."
                )

            transformer_info = context.models.load(self.transformer.transformer)
            if controlnet_extensions:
                # Do not pin the 17 GB Chroma transformer in VRAM while the 6 GB ControlNet
                # is active. The adapter reacquires it around each Chroma forward, allowing
                # ModelCache to phase-swap the two large models between denoising phases.
                transformer = transformer_info.model
            else:
                # Preserve the established non-ControlNet Chroma lifecycle exactly.
                _cached_weights, transformer = exit_stack.enter_context(transformer_info.model_on_device())

            if not isinstance(transformer, ChromaTransformer2DModel):
                raise TypeError(f"Expected ChromaTransformer2DModel, got {type(transformer).__name__}")

            weight_stager = exit_stack.enter_context(cuda_async_linear_weight_staging(device))

            adapter = ChromaTransformerAdapter(
                transformer,
                model_input_dtype=transformer_dtype,
                loaded_model=transformer_info if controlnet_extensions else None,
            )
            sequential_guidance = context.config.get().sequential_guidance
            if self.scheduler == "euler_cfg_pp_beta":
                if negative_extension is None:
                    raise ValueError("Negative text conditioning is required for Chroma Euler CFG++")
                if controlnet_extensions:
                    context.logger.info(
                        "Chroma CFG++ ControlNet: positive-only residuals enabled; active ControlNet steps use "
                        "sequential positive/negative forwards."
                    )
                elif sequential_guidance:
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
                    controlnet_extensions=controlnet_extensions,
                    controlnet_guidance=controlnet_guidance,
                    controlnet_input_dtype=inference_dtype,
                )
            else:
                denoise_cfg_scale = cfg_scale
                denoise_negative_extension = negative_extension
                has_guided_steps = any(not math.isclose(scale, 1.0) for scale in cfg_scale)
                if (
                    has_guided_steps
                    and negative_extension is not None
                    and not sequential_guidance
                    and not controlnet_extensions
                ):
                    # Reuse InvokeAI's existing global guidance policy: the default is parallel/batched guidance,
                    # while sequential_guidance=true is the low-memory opt-out. The adapter owns the Chroma-specific
                    # batch assembly and OOM fallback; the shared FLUX loop therefore sees CFG=1 and performs one
                    # model call per scheduler step.
                    adapter.enable_batched_cfg(negative_extension=negative_extension, cfg_scale=cfg_scale)
                    denoise_cfg_scale = [1.0] * len(cfg_scale)
                    denoise_negative_extension = None
                    context.logger.info("Chroma CFG: batched positive/negative forward enabled.")
                elif has_guided_steps and controlnet_extensions:
                    # The shared FLUX denoiser applies ControlNet residuals only to the positive
                    # branch. Keep CFG sequential until the Chroma adapter has an explicitly
                    # branch-aware batched ControlNet path.
                    context.logger.info(
                        "Chroma ControlNet: sequential positive/negative CFG enabled so ControlNet residuals "
                        "apply only to the positive branch."
                    )
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
                    guidance=controlnet_guidance,
                    cfg_scale=denoise_cfg_scale,
                    inpaint_extension=inpaint_extension,
                    controlnet_extensions=controlnet_extensions,
                    pos_ip_adapter_extensions=[],
                    neg_ip_adapter_extensions=[],
                    img_cond=None,
                    scheduler=scheduler,
                )

            if weight_stager is not None and weight_stager.stats.staged_tensors > 0:
                stats = weight_stager.stats
                context.logger.info(
                    f"Chroma async linear weight staging: {stats.staged_tensors} tensors "
                    f"({stats.staged_bytes / 2**30:.2f} GiB), {stats.slot_waits} waits for reusable slots, "
                    f"{stats.synchronous_fallbacks} synchronous fallbacks."
                )

        return unpack(packed_latents.float(), self.height, self.width)

    def _prep_chroma_controlnet_extensions(
        self,
        context: InvocationContext,
        latent_height: int,
        latent_width: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> list[ChromaInstantXControlNetExtension]:
        if self.control is None:
            return []

        controlnets: list[FluxControlNetField]
        if isinstance(self.control, FluxControlNetField):
            controlnets = [self.control]
        elif isinstance(self.control, list):
            controlnets = self.control
        else:
            raise ValueError(f"Unsupported Chroma ControlNet input type: {type(self.control)}")

        if self.controlnet_vae is None:
            raise ValueError("A ControlNet VAE is required when using an InstantX FLUX ControlNet with Chroma")

        # Encode control images before retaining any ControlNet model handles. This keeps
        # the VAE phase isolated and avoids admitting a 6+ GB side model before VAE work.
        vae_info = context.models.load(self.controlnet_vae.vae)
        extensions: list[ChromaInstantXControlNetExtension] = []
        for controlnet in controlnets:
            controlnet_image = _prepare_chroma_controlnet_image(context.images.get_pil(controlnet.image.image_name))
            controlnet_cond = ChromaInstantXControlNetExtension.prepare_controlnet_cond(
                controlnet_image=controlnet_image,
                vae_info=vae_info,
                latent_height=latent_height,
                latent_width=latent_width,
                dtype=dtype,
                device=device,
                resize_mode=controlnet.resize_mode,
            )

            instantx_control_mode: torch.Tensor | None = None
            if controlnet.instantx_control_mode is not None and controlnet.instantx_control_mode >= 0:
                instantx_control_mode = torch.tensor(
                    controlnet.instantx_control_mode,
                    dtype=torch.long,
                ).reshape([-1, 1])

            # Keep only the LoadedModel handle here. The extension acquires model_on_device()
            # just for its own forward, so the model cache can evict Chroma for the ControlNet
            # phase and evict ControlNet again for the following Chroma phase.
            controlnet_model_info = context.models.load(controlnet.control_model)
            extensions.append(
                ChromaInstantXControlNetExtension(
                    model_info=controlnet_model_info,
                    controlnet_cond=controlnet_cond,
                    instantx_control_mode=instantx_control_mode,
                    weight=controlnet.control_weight,
                    begin_step_percent=controlnet.begin_step_percent,
                    end_step_percent=controlnet.end_step_percent,
                )
            )

        return extensions

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
