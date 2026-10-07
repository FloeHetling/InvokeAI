from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.chroma.chroma_denoise import _validate_chroma_controlnet_scheduler
from invokeai.backend.chroma.model import ChromaTransformerAdapter


def _regional_extension(dtype: torch.dtype = torch.float16) -> SimpleNamespace:
    conditioning = SimpleNamespace(
        t5_embeddings=torch.zeros((1, 1, 8), dtype=dtype),
        t5_txt_ids=torch.zeros((1, 1, 3), dtype=dtype),
        clip_embeddings=torch.zeros((1, 8), dtype=dtype),
        attention_mask=None,
    )
    return SimpleNamespace(regional_text_conditioning=conditioning, restricted_attn_mask=None)


def test_chroma_controlnet_scheduler_gate_rejects_other_schedulers() -> None:
    with pytest.raises(ValueError, match=r"Euler CFG\+\+ \(Beta\)"):
        _validate_chroma_controlnet_scheduler("ddim")


def test_cfg_pp_controlnet_residuals_apply_only_to_positive_branch(monkeypatch) -> None:
    adapter = ChromaTransformerAdapter(MagicMock())
    positive_extension = _regional_extension()
    negative_extension = _regional_extension()
    positive_pred = torch.full((1, 1, 4), 1.0)
    negative_pred = torch.full((1, 1, 4), -1.0)
    forward_model = MagicMock(side_effect=[positive_pred, negative_pred])
    can_batch = MagicMock(return_value=True)
    batched_branches = MagicMock(return_value=(positive_pred, negative_pred))
    monkeypatch.setattr(adapter, "_forward_model", forward_model)
    monkeypatch.setattr(adapter, "_can_batch_cfg", can_batch)
    monkeypatch.setattr(adapter, "_run_batched_cfg_branches", batched_branches)

    double_block_residuals = [torch.ones((1, 1, 4))]
    result = adapter.predict_cfg_branches(
        img=torch.zeros((1, 1, 4)),
        img_ids=torch.zeros((1, 1, 3)),
        timesteps=torch.tensor([0.5]),
        positive_extension=positive_extension,  # type: ignore[arg-type]
        negative_extension=negative_extension,  # type: ignore[arg-type]
        allow_batched=True,
        controlnet_double_block_residuals=double_block_residuals,
        controlnet_single_block_residuals=None,
    )

    assert result[0] is positive_pred
    assert result[1] is negative_pred
    assert not can_batch.called
    assert not batched_branches.called
    assert forward_model.call_count == 2
    positive_call = forward_model.call_args_list[0].kwargs
    negative_call = forward_model.call_args_list[1].kwargs
    assert positive_call["controlnet_double_block_residuals"] is double_block_residuals
    assert positive_call["controlnet_single_block_residuals"] is None
    assert negative_call["controlnet_double_block_residuals"] is None
    assert negative_call["controlnet_single_block_residuals"] is None
