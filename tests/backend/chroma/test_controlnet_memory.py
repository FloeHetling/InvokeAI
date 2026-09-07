from contextlib import contextmanager
from typing import Iterator

import torch
from diffusers import ChromaTransformer2DModel

from invokeai.backend.chroma.controlnet import ChromaInstantXControlNetExtension
from invokeai.backend.chroma.model import ChromaTransformerAdapter
from invokeai.backend.flux.controlnet.controlnet_flux_output import ControlNetFluxOutput
from invokeai.backend.flux.controlnet.instantx_controlnet_flux import InstantXControlNetFlux
from invokeai.backend.flux.extensions.instantx_controlnet_extension import InstantXControlNetExtension
from invokeai.backend.flux.model import FluxParams


class _LoadedModelStub:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.calls = 0

    @contextmanager
    def model_on_device(self) -> Iterator[tuple[None, torch.nn.Module]]:
        self.calls += 1
        yield None, self.model


def _build_tiny_instantx_controlnet() -> InstantXControlNetFlux:
    return InstantXControlNetFlux(
        FluxParams(
            in_channels=4,
            vec_in_dim=8,
            context_in_dim=8,
            hidden_size=8,
            mlp_ratio=1.0,
            num_heads=1,
            depth=1,
            depth_single_blocks=1,
            axes_dim=[2, 2, 4],
            theta=10000,
            qkv_bias=True,
            guidance_embed=False,
        ),
        num_control_modes=1,
    )


def _controlnet_call_kwargs() -> dict[str, object]:
    return {
        "timestep_index": 0,
        "total_num_timesteps": 1,
        "img": torch.zeros(1, 1, 4),
        "img_ids": torch.zeros(1, 1, 3),
        "txt": torch.zeros(1, 1, 8),
        "txt_ids": torch.zeros(1, 1, 3),
        "y": torch.zeros(1, 8),
        "timesteps": torch.zeros(1),
        "guidance": None,
    }


def test_chroma_controlnet_preserves_user_weight() -> None:
    model = _build_tiny_instantx_controlnet()
    loaded_model = _LoadedModelStub(model)

    scalar_extension = ChromaInstantXControlNetExtension(
        model_info=loaded_model,  # type: ignore[arg-type]
        controlnet_cond=torch.zeros(1, 1, 4),
        instantx_control_mode=None,
        weight=0.5,
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )
    assert abs(scalar_extension._get_weight(timestep_index=0, total_num_timesteps=1) - 0.5) < 1e-12

    scheduled_extension = ChromaInstantXControlNetExtension(
        model_info=loaded_model,  # type: ignore[arg-type]
        controlnet_cond=torch.zeros(1, 1, 4),
        instantx_control_mode=None,
        weight=[0.0, 0.5, 1.0],
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )
    assert [
        scheduled_extension._get_weight(timestep_index=index, total_num_timesteps=3) for index in range(3)
    ] == [0.0, 0.5, 1.0]


def test_chroma_controlnet_acquires_model_only_for_active_controlnet_phase(monkeypatch) -> None:
    model = _build_tiny_instantx_controlnet()
    loaded_model = _LoadedModelStub(model)
    extension = ChromaInstantXControlNetExtension(
        model_info=loaded_model,  # type: ignore[arg-type]
        controlnet_cond=torch.zeros(1, 1, 4),
        instantx_control_mode=None,
        weight=0.5,
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )
    expected = ControlNetFluxOutput(single_block_residuals=None, double_block_residuals=None)

    def fake_run_controlnet(self: InstantXControlNetExtension, **kwargs: object) -> ControlNetFluxOutput:
        return expected

    monkeypatch.setattr(InstantXControlNetExtension, "run_controlnet", fake_run_controlnet)

    result = extension.run_controlnet(**_controlnet_call_kwargs())  # type: ignore[arg-type]

    assert result is expected
    assert loaded_model.calls == 1


def test_chroma_controlnet_zero_weight_does_not_swap_model_in() -> None:
    model = _build_tiny_instantx_controlnet()
    loaded_model = _LoadedModelStub(model)
    extension = ChromaInstantXControlNetExtension(
        model_info=loaded_model,  # type: ignore[arg-type]
        controlnet_cond=torch.zeros(1, 1, 4),
        instantx_control_mode=None,
        weight=0.0,
        begin_step_percent=0.0,
        end_step_percent=1.0,
    )

    result = extension.run_controlnet(**_controlnet_call_kwargs())  # type: ignore[arg-type]

    assert result.single_block_residuals is None
    assert result.double_block_residuals is None
    assert loaded_model.calls == 0


def test_chroma_adapter_reacquires_phase_swapped_transformer_for_forward() -> None:
    model = ChromaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        approximator_num_channels=64,
        approximator_hidden_dim=16,
        approximator_layers=1,
    )
    loaded_model = _LoadedModelStub(model)
    adapter = ChromaTransformerAdapter(model, loaded_model=loaded_model)  # type: ignore[arg-type]

    prediction = adapter._forward_model(
        img=torch.randn(1, 4, 4),
        img_ids=torch.zeros(4, 3),
        txt=torch.randn(1, 3, 12),
        txt_ids=torch.zeros(3, 3),
        timesteps=torch.tensor([0.5]),
        text_attention_mask=None,
    )

    assert prediction.shape == (1, 4, 4)
    assert loaded_model.calls == 1
