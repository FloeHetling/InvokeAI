"""Model loaders for the Chroma image generation architecture."""

from pathlib import Path
from typing import Optional

import accelerate
import torch
from diffusers import ChromaTransformer2DModel
from diffusers.loaders.single_file_utils import convert_chroma_transformer_checkpoint_to_diffusers
from safetensors.torch import load_file
from transformers import T5Config, T5EncoderModel

from invokeai.backend.model_manager.configs.base import Checkpoint_Config_Base, Diffusers_Config_Base
from invokeai.backend.model_manager.configs.factory import AnyModelConfig
from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Chroma_Config
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.load_default import (
    ModelLoader,
    resolve_submodel_path,
)
from invokeai.backend.model_manager.load.model_loader_registry import ModelLoaderRegistry
from invokeai.backend.model_manager.load.model_loaders.generic_diffusers import GenericDiffusersLoader
from invokeai.backend.model_manager.taxonomy import (
    AnyModel,
    BaseModelType,
    ModelFormat,
    ModelType,
    SubModelType,
)
from invokeai.backend.util.devices import TorchDevice

CHROMA_TRANSFORMER_CONFIG = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "axes_dims_rope": (16, 56, 56),
    "approximator_num_channels": 64,
    "approximator_hidden_dim": 5120,
    "approximator_layers": 5,
}


def _t5_xxl_config() -> T5Config:
    """Return the T5-v1.1-XXL encoder config used by FLUX and Chroma."""
    return T5Config(
        vocab_size=32128,
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        is_gated_act=True,
        dense_act_fn="gelu_new",
        tie_word_embeddings=False,
        use_cache=False,
    )


def _raise_on_incomplete_load(model: torch.nn.Module, missing_keys: list[str], unexpected_keys: list[str]) -> None:
    allowed_missing = {"encoder.embed_tokens.weight"}
    disallowed_missing = set(missing_keys) - allowed_missing
    meta_params = [name for name, param in model.named_parameters() if param.is_meta]
    if disallowed_missing or unexpected_keys or meta_params:
        raise RuntimeError(
            "Incomplete Chroma component load: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected_keys)}, meta={meta_params}"
        )


@ModelLoaderRegistry.register(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.Checkpoint)
class ChromaCheckpointModel(ModelLoader):
    """Load a native single-file Chroma transformer through Diffusers' canonical converter."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, Main_Checkpoint_Chroma_Config):
            raise TypeError(f"Expected Main_Checkpoint_Chroma_Config, got {type(config).__name__}")
        if submodel_type is not SubModelType.Transformer:
            raise ValueError(
                f"Only the Transformer submodel is available in a Chroma checkpoint; received {submodel_type}"
            )

        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(self._torch_device)
        state_dict = load_file(Path(config.path))
        state_dict = convert_chroma_transformer_checkpoint_to_diffusers(state_dict)

        with accelerate.init_empty_weights():
            model = ChromaTransformer2DModel(**CHROMA_TRANSFORMER_CONFIG)

        new_state_dict_size = sum(tensor.nelement() * model_dtype.itemsize for tensor in state_dict.values())
        self._ram_cache.make_room(new_state_dict_size)
        for key in state_dict:
            state_dict[key] = state_dict[key].to(model_dtype)

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        _raise_on_incomplete_load(model, missing_keys, unexpected_keys)
        return self._apply_fp8_layerwise_casting(model, config, SubModelType.Transformer)


@ModelLoaderRegistry.register(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.Diffusers)
class ChromaDiffusersModel(GenericDiffusersLoader):
    """Load components from a full Chroma Diffusers pipeline or a loose transformer folder."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if isinstance(config, Checkpoint_Config_Base):
            raise TypeError("Checkpoint configs are handled by ChromaCheckpointModel")
        if submodel_type is None:
            raise ValueError("A submodel type is required when loading a Chroma Diffusers model")

        model_root = Path(config.path)
        loose_transformer = model_root / "config.json"
        if submodel_type is SubModelType.Transformer and loose_transformer.exists():
            load_class = ChromaTransformer2DModel
            model_path = model_root
        else:
            load_class = self.get_hf_load_class(model_root, submodel_type)
            model_path = resolve_submodel_path(config, submodel_type, model_root / submodel_type.value)

        if submodel_type in {SubModelType.Tokenizer, SubModelType.Tokenizer2, SubModelType.Tokenizer3}:
            return load_class.from_pretrained(model_path, local_files_only=True)

        repo_variant = config.repo_variant if isinstance(config, Diffusers_Config_Base) else None
        variant = repo_variant.value if repo_variant else None
        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(self._torch_device)
        result: AnyModel = load_class.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            variant=variant,
            local_files_only=True,
        )
        return self._apply_fp8_layerwise_casting(result, config, submodel_type)


@ModelLoaderRegistry.register(base=BaseModelType.Any, type=ModelType.T5Encoder, format=ModelFormat.Checkpoint)
class T5EncoderRawCheckpointModel(ModelLoader):
    """Load a raw Transformers-format T5-XXL encoder checkpoint and bundled tokenizer."""

    def _load_model(
        self,
        config: AnyModelConfig,
        submodel_type: Optional[SubModelType] = None,
    ) -> AnyModel:
        if not isinstance(config, T5Encoder_Checkpoint_Config):
            raise TypeError(f"Expected T5Encoder_Checkpoint_Config, got {type(config).__name__}")

        if submodel_type in {SubModelType.Tokenizer, SubModelType.Tokenizer2, SubModelType.Tokenizer3}:
            from invokeai.backend.t5.t5_tokenizer import load_bundled_t5_tokenizer

            tokenizer = load_bundled_t5_tokenizer()
            tokenizer.model_max_length = 512
            return tokenizer
        if submodel_type not in {SubModelType.TextEncoder, SubModelType.TextEncoder2, SubModelType.TextEncoder3}:
            raise ValueError(f"Unsupported T5 checkpoint submodel: {submodel_type}")

        model_dtype = TorchDevice.choose_bfloat16_safe_dtype(self._torch_device)
        state_dict = load_file(Path(config.path))

        new_state_dict_size = sum(tensor.nelement() * model_dtype.itemsize for tensor in state_dict.values())
        self._ram_cache.make_room(new_state_dict_size)
        for key in state_dict:
            state_dict[key] = state_dict[key].to(model_dtype)

        with accelerate.init_empty_weights():
            model = T5EncoderModel(_t5_xxl_config())

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False, assign=True)
        model.encoder.embed_tokens.weight = model.shared.weight
        _raise_on_incomplete_load(model, missing_keys, unexpected_keys)

        # Keep runtime T5 weights at model_dtype after decoding an FP8 checkpoint. ModelLoader's normal FP8 policy
        # deliberately excludes TextEncoder* submodels; bypassing that policy here can leave a Float8 weight visible
        # to F.linear (for example, via an unsupported addmm_cuda Float8_e4m3fn execution path).

        return model
