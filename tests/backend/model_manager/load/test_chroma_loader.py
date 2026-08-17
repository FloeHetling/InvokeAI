from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from invokeai.backend.model_manager.configs.t5_encoder import T5Encoder_Checkpoint_Config
from invokeai.backend.model_manager.load.model_loaders.chroma import T5EncoderRawCheckpointModel
from invokeai.backend.model_manager.taxonomy import SubModelType


class _TinyT5Encoder(torch.nn.Module):
    def __init__(self, _config) -> None:
        super().__init__()
        self.shared = torch.nn.Embedding(2, 2)
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = torch.nn.Embedding(2, 2)


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
