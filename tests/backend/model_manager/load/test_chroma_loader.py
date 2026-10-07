from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.backend.model_manager.configs.main import Main_Checkpoint_Chroma_Config, Main_Diffusers_Chroma_Config
from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders.chroma import (
    ChromaCheckpointModel,
    ChromaDiffusersModel,
    T5EncoderRawCheckpointModel,
)
from invokeai.backend.model_manager.taxonomy import SubModelType


class _TinyT5Encoder(torch.nn.Module):
    def __init__(self, _config) -> None:
        super().__init__()
        self.shared = torch.nn.Embedding(2, 2)
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = torch.nn.Embedding(2, 2)


class _TinyChromaTransformer(torch.nn.Module):
    loaded_dtype: torch.dtype | None = None

    def __init__(self, **_kwargs) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def load_state_dict(self, state_dict, **_kwargs):
        type(self).loaded_dtype = state_dict["weight"].dtype
        return [], []


def test_raw_fp8_t5_is_materialized_in_compute_dtype(monkeypatch, tmp_path) -> None:
    checkpoint_path = tmp_path / "t5xxl_fp8.safetensors"
    checkpoint_path.touch()
    config = T5Encoder_Checkpoint_Config.model_construct(path=str(checkpoint_path), name="raw-fp8-t5")
    loader = object.__new__(T5EncoderRawCheckpointModel)
    loader._torch_device = torch.device("cuda")
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._logger = MagicMock()

    fp8_weight = torch.ones(2, 2).to(torch.float8_e4m3fn)
    state_dict = {
        "shared.weight": fp8_weight.clone(),
        "encoder.embed_tokens.weight": fp8_weight.clone(),
    }

    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.load_file",
        lambda _path: state_dict,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.T5EncoderModel",
        _TinyT5Encoder,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.accelerate.init_empty_weights",
        lambda: nullcontext(),
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.TorchDevice.choose_bfloat16_safe_dtype",
        lambda _device: torch.bfloat16,
    )

    result = loader._load_model(config, SubModelType.TextEncoder2)

    assert isinstance(result, _TinyT5Encoder)
    assert result.shared.weight.dtype is torch.bfloat16
    assert result.encoder.embed_tokens.weight is result.shared.weight


def test_chroma_checkpoint_transformer_is_materialized_in_float16(monkeypatch, tmp_path) -> None:
    _TinyChromaTransformer.loaded_dtype = None
    checkpoint_path = tmp_path / "chroma.safetensors"
    checkpoint_path.touch()
    config = Main_Checkpoint_Chroma_Config.model_construct(path=str(checkpoint_path), name="chroma")
    loader = object.__new__(ChromaCheckpointModel)
    loader._ram_cache = SimpleNamespace(make_room=MagicMock())
    loader._apply_fp8_layerwise_casting = MagicMock(side_effect=lambda model, *_args: model)

    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.load_file",
        lambda _path: {"weight": torch.ones(1, dtype=torch.float32)},
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.convert_chroma_transformer_checkpoint_to_diffusers",
        lambda state_dict: state_dict,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.ChromaTransformer2DModel",
        _TinyChromaTransformer,
    )
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.accelerate.init_empty_weights",
        lambda: nullcontext(),
    )

    result = loader._load_model(config, SubModelType.Transformer)

    assert isinstance(result, _TinyChromaTransformer)
    assert _TinyChromaTransformer.loaded_dtype is torch.float16


def test_chroma_diffusers_transformer_is_loaded_in_float16(monkeypatch, tmp_path) -> None:
    (tmp_path / "config.json").touch()
    config = Main_Diffusers_Chroma_Config.model_construct(path=str(tmp_path), name="chroma")
    loader = object.__new__(ChromaDiffusersModel)
    loader._torch_device = torch.device("cuda")
    loader._apply_fp8_layerwise_casting = MagicMock(side_effect=lambda model, *_args: model)
    loaded_model = MagicMock()
    from_pretrained = MagicMock(return_value=loaded_model)
    monkeypatch.setattr(_TinyChromaTransformer, "from_pretrained", from_pretrained, raising=False)
    monkeypatch.setattr(
        "invokeai.backend.model_manager.load.model_loaders.chroma.ChromaTransformer2DModel",
        _TinyChromaTransformer,
    )

    result = loader._load_model(config, SubModelType.Transformer)

    assert result is loaded_model
    assert from_pretrained.call_args.kwargs["torch_dtype"] is torch.float16
