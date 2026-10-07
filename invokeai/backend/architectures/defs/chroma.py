"""What the chroma architecture declares."""

from invokeai.backend.architectures.facets.conditioning import ConditioningFacet
from invokeai.backend.architectures.facets.default_settings import DefaultSettingsFacet
from invokeai.backend.architectures.facets.features import FeaturesFacet, NegativePrompt
from invokeai.backend.architectures.facets.latent_space import FLUX_16, LatentSpaceFacet
from invokeai.backend.architectures.facets.modality import ModalityFacet
from invokeai.backend.architectures.facets.vae import VaeCompatibility, VaeFacet
from invokeai.backend.architectures.registry import register
from invokeai.backend.model_manager.configs.default_settings import MainModelDefaultSettings
from invokeai.backend.model_manager.taxonomy import BaseModelType
from invokeai.backend.stable_diffusion.diffusion.conditioning_data import ChromaConditioningInfo

# Chroma1-HD is a FLUX.1-schnell derivative: it decodes with the FLUX.1 VAE, so it shares that latent space.
register(
    BaseModelType.Chroma,
    LatentSpaceFacet(FLUX_16),
    ConditioningFacet(ChromaConditioningInfo),
    # Chroma1-HD model card: 40 steps, CFG 3.0. It has no distilled guidance, so unlike FLUX there is one slider.
    DefaultSettingsFacet(
        {None: MainModelDefaultSettings(scheduler="euler", steps=40, cfg_scale=3.0, width=1024, height=1024)}
    ),
    ModalityFacet(frozenset({"txt2img", "img2img", "inpaint", "outpaint"}), metadata_slug="chroma"),
    FeaturesFacet(
        # CFG++ needs the negative branch, and plain CFG above 1.0 uses it too.
        negative_prompt=NegativePrompt(visible=True, usage="cfg-gated"),
        dimension_grid=16,  # equals `multiple_of` on chroma_denoise.width/height (inherited from flux_denoise)
        guidance_label="CFG",
        scheduler_set="chroma",
        scheduler_applies_to_graph=True,
        # FLUX ControlNets (Union, Union Pro 2.0, InstantX) run against the Chroma transformer; Redux
        # conditioning is accepted as reference images.
        control_kinds=frozenset({"controlnet"}),
        adapter_bases=frozenset({BaseModelType.Flux}),
        max_reference_images=5,
    ),
    VaeFacet(frozenset({VaeCompatibility(BaseModelType.Flux)})),
)
