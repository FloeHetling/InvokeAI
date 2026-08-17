from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.chroma_denoise import _validate_chroma_controlnet_scheduler
from invokeai.backend.chroma.denoise import denoise_euler_cfg_pp
from invokeai.backend.chroma.model import ChromaTransformerAdapter
from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput


def _regional_extension(dtype: torch.dtype = torch.float16) -> SimpleNamespace:
    conditioning = SimpleNamespace(
        t5_embeddings=torch.zeros((1, 1, 8), dtype=dtype),
        t5_txt_ids=torch.zeros((1, 1, 3), dtype=dtype),
        clip_embeddings=torch.zeros((1, 8), dtype=dtype),
        attention_mask=None,
    )
    return SimpleNamespace(regional_text_conditioning=conditioning, restricted_attn_mask=None)


@pytest.mark.parametrize("scheduler", ["euler", "euler_cfg_pp_beta"])
def test_chroma_controlnet_scheduler_gate_allows_supported_schedulers(scheduler: str) -> None:
    _validate_chroma_controlnet_scheduler(scheduler)


def test_chroma_controlnet_scheduler_gate_rejects_other_schedulers() -> None:
    with pytest.raises(ValueError, match=r"Euler CFG\+\+ \(Beta\)"):
        _validate_chroma_controlnet_scheduler("ddim")


def test_cfg_pp_runs_controlnet_in_side_model_dtype_and_forwards_residuals() -> None:
    positive_extension = _regional_extension()
    negative_extension = _regional_extension()
    model = MagicMock()
    model.predict_cfg_branches.return_value = (
        torch.zeros((1, 1, 4), dtype=torch.float16),
        torch.zeros((1, 1, 4), dtype=torch.float16),
    )

    double_block_residuals = [torch.ones((1, 1, 4), dtype=torch.bfloat16)]
    controlnet_extension = MagicMock()
    controlnet_extension.run_controlnet.return_value = ControlNetFluxOutput(
        single_block_residuals=None,
        double_block_residuals=double_block_residuals,
    )

    output = denoise_euler_cfg_pp(
        model=model,
        img=torch.ones((1, 1, 4), dtype=torch.float32),
        img_ids=torch.zeros((1, 1, 3), dtype=torch.float32),
        positive_extension=positive_extension,  # type: ignore[arg-type]
        negative_extension=negative_extension,  # type: ignore[arg-type]
        timesteps=[1.0, 0.0],
        cfg_scale=[3.0],
        step_callback=lambda _state: None,
        inpaint_extension=None,
        allow_batched_cfg=True,
        model_input_dtype=torch.float16,
        controlnet_extensions=[controlnet_extension],
        controlnet_guidance=3.5,
        controlnet_input_dtype=torch.bfloat16,
    )

    controlnet_call = controlnet_extension.run_controlnet.call_args.kwargs
    assert controlnet_call["img"].dtype == torch.bfloat16
    assert controlnet_call["img_ids"].dtype == torch.bfloat16
    assert controlnet_call["timesteps"].dtype == torch.bfloat16
    assert controlnet_call["guidance"].dtype == torch.bfloat16
    assert torch.all(controlnet_call["guidance"] == 3.5)
    assert controlnet_call["txt"] is positive_extension.regional_text_conditioning.t5_embeddings
    assert controlnet_call["txt_ids"] is positive_extension.regional_text_conditioning.t5_txt_ids
    assert controlnet_call["y"] is positive_extension.regional_text_conditioning.clip_embeddings

    model_call = model.predict_cfg_branches.call_args.kwargs
    assert model_call["img"].dtype == torch.float16
    assert model_call["img_ids"].dtype == torch.float16
    assert model_call["timesteps"].dtype == torch.float32
    assert model_call["controlnet_double_block_residuals"] is double_block_residuals
    assert model_call["controlnet_single_block_residuals"] is None
    assert output.dtype == torch.float32


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


def test_cfg_pp_inactive_controlnet_residuals_keep_batched_branch_path(monkeypatch) -> None:
    adapter = ChromaTransformerAdapter(MagicMock())
    positive_extension = _regional_extension()
    negative_extension = _regional_extension()
    positive_pred = torch.full((1, 1, 4), 1.0)
    negative_pred = torch.full((1, 1, 4), -1.0)
    can_batch = MagicMock(return_value=True)
    batched_branches = MagicMock(return_value=(positive_pred, negative_pred))
    monkeypatch.setattr(adapter, "_can_batch_cfg", can_batch)
    monkeypatch.setattr(adapter, "_run_batched_cfg_branches", batched_branches)

    result = adapter.predict_cfg_branches(
        img=torch.zeros((1, 1, 4)),
        img_ids=torch.zeros((1, 1, 3)),
        timesteps=torch.tensor([0.5]),
        positive_extension=positive_extension,  # type: ignore[arg-type]
        negative_extension=negative_extension,  # type: ignore[arg-type]
        allow_batched=True,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
    )

    assert result[0] is positive_pred
    assert result[1] is negative_pred
    can_batch.assert_called_once()
    batched_branches.assert_called_once()
