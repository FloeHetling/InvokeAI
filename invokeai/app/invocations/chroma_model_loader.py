from invokeai.app.invocations.baseinvocation import BaseInvocation, BaseInvocationOutput, invocation, invocation_output
from invokeai.app.invocations.fields import FieldDescriptions, InputField, OutputField
from invokeai.app.invocations.model import ModelIdentifierField, T5EncoderField, TransformerField, VAEField
from invokeai.app.services.shared.invocation_context import InvocationContext
from invokeai.app.util.t5_model_identifier import (
    preprocess_t5_encoder_model_identifier,
    preprocess_t5_tokenizer_model_identifier,
)
from invokeai.backend.model_manager.configs.main import Main_Diffusers_Chroma_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelType, SubModelType


@invocation_output("chroma_model_loader_output")
class ChromaModelLoaderOutput(BaseInvocationOutput):
    transformer: TransformerField = OutputField(description=FieldDescriptions.transformer, title="Transformer")
    t5_encoder: T5EncoderField = OutputField(description=FieldDescriptions.t5_encoder, title="T5 Encoder")
    vae: VAEField = OutputField(description=FieldDescriptions.vae, title="VAE")


@invocation(
    "chroma_model_loader",
    title="Main Model - Chroma",
    tags=["model", "chroma"],
    category="model",
    version="1.0.0",
)
class ChromaModelLoaderInvocation(BaseInvocation):
    """Load a Chroma transformer with its T5-XXL encoder and FLUX VAE."""

    model: ModelIdentifierField = InputField(
        description="Chroma main model",
        ui_model_base=BaseModelType.Chroma,
        ui_model_type=ModelType.Main,
    )
    t5_encoder_model: ModelIdentifierField | None = InputField(
        default=None,
        description=FieldDescriptions.t5_encoder,
        title="T5 Encoder",
        ui_model_type=ModelType.T5Encoder,
    )
    vae_model: ModelIdentifierField | None = InputField(
        default=None,
        description=FieldDescriptions.vae_model,
        title="VAE",
        ui_model_base=BaseModelType.Flux,
        ui_model_type=ModelType.VAE,
    )

    def invoke(self, context: InvocationContext) -> ChromaModelLoaderOutput:
        for identifier in (self.model, self.t5_encoder_model, self.vae_model):
            if identifier is not None and not context.models.exists(identifier.key):
                raise ValueError(f"Unknown model: {identifier.key}")

        main_config = context.models.get_config(self.model)
        if main_config.base is not BaseModelType.Chroma or main_config.type is not ModelType.Main:
            raise ValueError("The selected main model is not a Chroma model")

        required_submodels = {
            SubModelType.Transformer,
            SubModelType.TextEncoder,
            SubModelType.Tokenizer,
            SubModelType.VAE,
        }
        is_complete_pipeline = isinstance(main_config, Main_Diffusers_Chroma_Config) and required_submodels.issubset(
            (main_config.submodels or {}).keys()
        )

        t5_source = self.t5_encoder_model or (self.model if is_complete_pipeline else None)
        vae_source = self.vae_model or (self.model if is_complete_pipeline else None)
        missing = [label for label, source in (("T5 Encoder", t5_source), ("FLUX VAE", vae_source)) if source is None]
        if missing:
            raise ValueError(
                f"The selected Chroma model does not contain its own {', '.join(missing)}; "
                "select the missing component explicitly."
            )

        transformer = self.model.model_copy(update={"submodel_type": SubModelType.Transformer})
        if t5_source is self.model:
            tokenizer = t5_source.model_copy(update={"submodel_type": SubModelType.Tokenizer})
            text_encoder = t5_source.model_copy(update={"submodel_type": SubModelType.TextEncoder})
        else:
            tokenizer = preprocess_t5_tokenizer_model_identifier(t5_source)
            text_encoder = preprocess_t5_encoder_model_identifier(t5_source)

        vae = vae_source.model_copy(update={"submodel_type": SubModelType.VAE})
        return ChromaModelLoaderOutput(
            transformer=TransformerField(transformer=transformer, loras=[]),
            t5_encoder=T5EncoderField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[]),
            vae=VAEField(vae=vae),
        )
