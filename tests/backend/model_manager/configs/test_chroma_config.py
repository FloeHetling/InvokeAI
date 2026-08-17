from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.backend.model_manager.configs.identification_utils import NotAMatchError
from invokeai.backend.model_manager.configs.main import (
    Main_Checkpoint_Chroma_Config,
    Main_Checkpoint_FLUX_Config,
    Main_Diffusers_Chroma_Config,
)
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType


def _overrides(path: Path, name: str = "test-model") -> dict:
    return {
        "hash": "test-hash",
        "path": path.as_posix(),
        "file_size": 0,
        "name": name,
        "source": path.as_posix(),
        "source_type": "path",
    }


def _mock_file(tmp_path: Path, state_dict: dict) -> MagicMock:
    path = tmp_path / "model.safetensors"
    path.touch()
    mod = MagicMock()
    mod.path = path
    mod.load_state_dict.return_value = state_dict
    mod.metadata.return_value = {}
    return mod


def _chroma_state_dict(prefix: str = "") -> dict[str, torch.Tensor]:
    tensor = torch.empty(1, device="meta")
    return {
        f"{prefix}distilled_guidance_layer.in_proj.weight": tensor,
        f"{prefix}distilled_guidance_layer.layers.4.out_layer.weight": tensor,
        f"{prefix}double_blocks.0.img_attn.norm.key_norm.scale": tensor,
        f"{prefix}single_blocks.37.linear2.weight": tensor,
    }


@pytest.mark.parametrize("prefix", ["", "model.diffusion_model."])
def test_chroma_checkpoint_is_identified_without_falling_through_to_flux(tmp_path: Path, prefix: str) -> None:
    mod = _mock_file(tmp_path, _chroma_state_dict(prefix))

    config = Main_Checkpoint_Chroma_Config.from_model_on_disk(mod, _overrides(mod.path))

    assert config.base is BaseModelType.Chroma
    assert config.type is ModelType.Main
    assert config.format is ModelFormat.Checkpoint
    with pytest.raises(NotAMatchError, match="Chroma"):
        Main_Checkpoint_FLUX_Config.from_model_on_disk(mod, _overrides(mod.path))


def test_raw_t5_xxl_checkpoint_is_identified_by_architecture(tmp_path: Path) -> None:
    state_dict = {
        "shared.weight": torch.empty((32128, 4096), device="meta"),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.empty(1, device="meta"),
        "encoder.block.23.layer.1.DenseReluDense.wo.weight": torch.empty(1, device="meta"),
        "encoder.final_layer_norm.weight": torch.empty(1, device="meta"),
    }
    mod = _mock_file(tmp_path, state_dict)

    config = T5Encoder_Checkpoint_Config.from_model_on_disk(mod, _overrides(mod.path, "t5-xxl"))

    assert config.base is BaseModelType.Any
    assert config.type is ModelType.T5Encoder
    assert config.format is ModelFormat.Checkpoint


def test_raw_t5_checkpoint_rejects_an_incompatible_embedding_shape(tmp_path: Path) -> None:
    state_dict = {
        "shared.weight": torch.empty((32000, 4096), device="meta"),
        "encoder.block.0.layer.0.SelfAttention.q.weight": torch.empty(1, device="meta"),
        "encoder.block.23.layer.1.DenseReluDense.wo.weight": torch.empty(1, device="meta"),
        "encoder.final_layer_norm.weight": torch.empty(1, device="meta"),
    }
    mod = _mock_file(tmp_path, state_dict)

    with pytest.raises(NotAMatchError, match="embedding shape"):
        T5Encoder_Checkpoint_Config.from_model_on_disk(mod, _overrides(mod.path, "wrong-t5"))


def test_chroma_diffusers_pipeline_records_loadable_components(tmp_path: Path) -> None:
    model_index = {
        "_class_name": "ChromaPipeline",
        "transformer": ["diffusers", "ChromaTransformer2DModel"],
        "text_encoder": ["transformers", "T5EncoderModel"],
        "tokenizer": ["transformers", "T5TokenizerFast"],
        "vae": ["diffusers", "AutoencoderKL"],
    }
    import json

    (tmp_path / "model_index.json").write_text(json.dumps(model_index), encoding="utf-8")
    for component in ("transformer", "text_encoder", "tokenizer", "vae"):
        (tmp_path / component).mkdir()

    mod = MagicMock()
    mod.path = tmp_path
    config = Main_Diffusers_Chroma_Config.from_model_on_disk(mod, _overrides(tmp_path, "chroma-pipeline"))

    assert config.submodels is not None
    assert set(config.submodels) == {
        SubModelType.Transformer,
        SubModelType.TextEncoder,
        SubModelType.Tokenizer,
        SubModelType.VAE,
    }
