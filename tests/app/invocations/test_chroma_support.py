from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from invokeai.app.invocations.chroma_denoise import ChromaDenoiseInvocation
from invokeai.app.invocations.chroma_model_loader import ChromaModelLoaderInvocation
from invokeai.app.invocations.model import ModelIdentifierField
from invokeai.app.services.shared.graph import Graph
from invokeai.backend.chroma.denoise import denoise_euler_cfg_pp, euler_cfg_pp_step
from invokeai.backend.chroma.model import ChromaTransformerAdapter
from invokeai.backend.model_manager.configs.main import Main_Diffusers_Chroma_Config
from invokeai.backend.model_manager.taxonomy import BaseModelType, ModelFormat, ModelType, SubModelType

_PIPELINE_SUBMODELS = {
    SubModelType.Transformer: object(),
    SubModelType.TextEncoder: object(),
    SubModelType.Tokenizer: object(),
    SubModelType.VAE: object(),
}


def _identifier(key: str, *, base: BaseModelType, model_type: ModelType) -> ModelIdentifierField:
    return ModelIdentifierField(key=key, hash=f"hash:{key}", name=key, base=base, type=model_type)


def _context(config) -> MagicMock:
    context = MagicMock()
    context.models.exists.return_value = True
    context.models.get_config.return_value = config
    return context


def test_chroma_denoise_migrates_legacy_cfg_pp_scheduler() -> None:
    graph = Graph.model_validate(
        {
            "nodes": {
                "chroma_denoise:test": {
                    "id": "chroma_denoise:test",
                    "type": "chroma_denoise",
                    "scheduler": "euler_cfg_pp",
                }
            }
        }
    )
    invocation = graph.nodes["chroma_denoise:test"]
    scheduler_schema = ChromaDenoiseInvocation.model_json_schema()["properties"]["scheduler"]

    assert isinstance(invocation, ChromaDenoiseInvocation)
    assert invocation.scheduler == "euler_cfg_pp_beta"
    assert scheduler_schema["enum"] == ["euler", "euler_cfg_pp_beta", "heun", "lcm"]


def test_complete_chroma_pipeline_supplies_transformer_t5_and_vae() -> None:
    main = _identifier("chroma-pipeline", base=BaseModelType.Chroma, model_type=ModelType.Main)
    config = Main_Diffusers_Chroma_Config.model_construct(
        format=ModelFormat.Diffusers,
        submodels=_PIPELINE_SUBMODELS,
    )
    invocation = ChromaModelLoaderInvocation.model_construct(model=main, t5_encoder_model=None, vae_model=None)

    output = invocation.invoke(_context(config))

    assert output.transformer.transformer.submodel_type is SubModelType.Transformer
    assert output.t5_encoder.text_encoder.submodel_type is SubModelType.TextEncoder
    assert output.t5_encoder.tokenizer.submodel_type is SubModelType.Tokenizer
    assert output.vae.vae.submodel_type is SubModelType.VAE
    assert output.transformer.transformer.key == main.key


def test_single_file_chroma_requires_external_t5_and_vae() -> None:
    main = _identifier("chroma-checkpoint", base=BaseModelType.Chroma, model_type=ModelType.Main)
    invocation = ChromaModelLoaderInvocation.model_construct(model=main, t5_encoder_model=None, vae_model=None)
    config = SimpleNamespace(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.Checkpoint)

    with pytest.raises(ValueError) as excinfo:
        invocation.invoke(_context(config))

    assert "T5 Encoder" in str(excinfo.value)
    assert "FLUX VAE" in str(excinfo.value)


def test_single_file_chroma_uses_raw_t5_and_flux_vae_identifiers() -> None:
    main = _identifier("chroma-checkpoint", base=BaseModelType.Chroma, model_type=ModelType.Main)
    t5 = _identifier("raw-t5", base=BaseModelType.Any, model_type=ModelType.T5Encoder)
    vae = _identifier("flux-vae", base=BaseModelType.Flux, model_type=ModelType.VAE)
    invocation = ChromaModelLoaderInvocation.model_construct(model=main, t5_encoder_model=t5, vae_model=vae)
    config = SimpleNamespace(base=BaseModelType.Chroma, type=ModelType.Main, format=ModelFormat.Checkpoint)

    output = invocation.invoke(_context(config))

    assert output.t5_encoder.text_encoder.key == t5.key
    assert output.t5_encoder.text_encoder.submodel_type is SubModelType.TextEncoder2
    assert output.t5_encoder.tokenizer.submodel_type is SubModelType.Tokenizer2
    assert output.vae.vae.key == vae.key
    assert output.vae.vae.submodel_type is SubModelType.VAE


def test_chroma_transformer_adapter_extends_the_text_mask_over_image_tokens() -> None:
    model = MagicMock()
    expected = torch.randn(1, 4, 64)
    model.return_value = (expected,)
    regional_prompting = SimpleNamespace(
        restricted_attn_mask=None,
        regional_text_conditioning=SimpleNamespace(
            attention_mask=torch.tensor([[True, True, False]], dtype=torch.bool)
        ),
    )
    img = torch.randn(1, 4, 64)
    img_ids = torch.zeros(1, 4, 3)
    txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    timesteps = torch.tensor([0.5])

    result = ChromaTransformerAdapter(model)(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=torch.zeros(1, 768),
        timesteps=timesteps,
        guidance=torch.zeros(1),
        timestep_index=0,
        total_num_timesteps=1,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
        ip_adapter_extensions=[],
        regional_prompting_extension=regional_prompting,
    )

    assert result is expected
    call = model.call_args.kwargs
    assert call["hidden_states"] is img
    assert call["encoder_hidden_states"] is txt
    assert call["timestep"] is timesteps
    assert call["attention_mask"].tolist() == [[True, True, False, True, True, True, True]]
    assert call["txt_ids"].shape == (3, 3)
    assert call["img_ids"].shape == (4, 3)


def test_chroma_transformer_adapter_omits_an_all_true_attention_mask() -> None:
    model = MagicMock()
    expected = torch.randn(1, 4, 64)
    model.return_value = (expected,)
    regional_prompting = SimpleNamespace(
        restricted_attn_mask=None,
        regional_text_conditioning=SimpleNamespace(
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
        ),
    )

    result = ChromaTransformerAdapter(model)(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        txt=torch.randn(1, 3, 4096),
        txt_ids=torch.zeros(1, 3, 3),
        y=torch.zeros(1, 768),
        timesteps=torch.tensor([0.5]),
        guidance=torch.zeros(1),
        timestep_index=0,
        total_num_timesteps=1,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
        ip_adapter_extensions=[],
        regional_prompting_extension=regional_prompting,
    )

    assert result is expected
    assert model.call_args.kwargs["attention_mask"] is None


def test_chroma_transformer_adapter_bridges_sampler_dtype_to_transformer_dtype() -> None:
    model = MagicMock()
    model_output = torch.randn(1, 4, 64, dtype=torch.float16)
    model.return_value = (model_output,)
    regional_prompting = SimpleNamespace(
        restricted_attn_mask=None,
        regional_text_conditioning=SimpleNamespace(
            attention_mask=torch.ones(1, 3, dtype=torch.bool),
        ),
    )
    img = torch.randn(1, 4, 64, dtype=torch.bfloat16)
    img_ids = torch.zeros(1, 4, 3, dtype=torch.bfloat16)
    txt = torch.randn(1, 3, 4096, dtype=torch.bfloat16)
    txt_ids = torch.zeros(1, 3, 3, dtype=torch.bfloat16)

    result = ChromaTransformerAdapter(model, model_input_dtype=torch.float16)(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        y=torch.zeros(1, 768),
        timesteps=torch.tensor([0.5], dtype=torch.bfloat16),
        guidance=torch.zeros(1),
        timestep_index=0,
        total_num_timesteps=1,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
        ip_adapter_extensions=[],
        regional_prompting_extension=regional_prompting,
    )

    call = model.call_args.kwargs
    assert call["hidden_states"].dtype is torch.float16
    assert call["encoder_hidden_states"].dtype is torch.float16
    assert call["img_ids"].dtype is torch.float16
    assert call["txt_ids"].dtype is torch.float16
    assert result.dtype is torch.bfloat16
    assert torch.equal(result, model_output.to(dtype=torch.bfloat16))


def test_chroma_modulation_input_preserves_reference_device_dtype_and_layout() -> None:
    adapter = ChromaTransformerAdapter(MagicMock())
    img = torch.zeros((2, 4, 64), dtype=torch.float16)
    timesteps = torch.tensor([0.25, 0.75], dtype=torch.float32)

    input_vec = adapter._build_chroma_modulation_input(timesteps, img)

    assert input_vec.shape == (2, 344, 64)
    assert input_vec.dtype is img.dtype
    assert input_vec.device == img.device
    assert torch.equal(input_vec[:, 0, :32], input_vec[:, -1, :32])
    assert not torch.equal(input_vec[0, 0, :16], input_vec[1, 0, :16])
    assert torch.equal(input_vec[:, 0, 16:24], torch.ones((2, 8), dtype=img.dtype))
    assert torch.equal(input_vec[:, 0, 24:32], torch.zeros((2, 8), dtype=img.dtype))
    assert not torch.equal(input_vec[0, 0, 32:], input_vec[0, 1, 32:])


def _chroma_cfg_extension(
    *,
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
    txt_ids: torch.Tensor,
) -> SimpleNamespace:
    return SimpleNamespace(
        restricted_attn_mask=None,
        regional_text_conditioning=SimpleNamespace(
            t5_embeddings=embeddings,
            t5_txt_ids=txt_ids,
            attention_mask=attention_mask,
        ),
    )


def test_chroma_transformer_adapter_batches_cfg_in_one_transformer_call() -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.return_value = (torch.cat((positive_pred, negative_pred), dim=0),)

    positive_txt = torch.randn(1, 3, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    positive_extension = _chroma_cfg_extension(
        embeddings=positive_txt,
        attention_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
        txt_ids=txt_ids,
    )
    negative_extension = _chroma_cfg_extension(
        embeddings=negative_txt,
        attention_mask=torch.tensor([[True, False, False]], dtype=torch.bool),
        txt_ids=txt_ids.clone(),
    )

    adapter = ChromaTransformerAdapter(model)
    adapter.enable_batched_cfg(negative_extension=negative_extension, cfg_scale=[3.0])
    result = adapter(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        txt=positive_txt,
        txt_ids=txt_ids,
        y=torch.zeros(1, 768),
        timesteps=torch.tensor([0.5]),
        guidance=torch.zeros(1),
        timestep_index=0,
        total_num_timesteps=1,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
        ip_adapter_extensions=[],
        regional_prompting_extension=positive_extension,
    )

    assert torch.equal(result, torch.full_like(result, 4.0))
    model.assert_called_once()
    call = model.call_args.kwargs
    assert call["hidden_states"].shape == (2, 4, 64)
    assert call["encoder_hidden_states"].shape == (2, 3, 4096)
    assert call["timestep"].shape == (2,)
    assert call["txt_ids"].shape == (3, 3)
    assert call["img_ids"].shape == (4, 3)
    assert call["attention_mask"].tolist() == [
        [True, True, False, True, True, True, True],
        [True, False, False, True, True, True, True],
    ]


def test_chroma_transformer_adapter_falls_back_to_sequential_cfg_after_batched_oom(monkeypatch) -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.side_effect = [
        torch.OutOfMemoryError("synthetic batched CFG OOM"),
        (positive_pred,),
        (negative_pred,),
    ]
    empty_cache = MagicMock()
    monkeypatch.setattr("invokeai.backend.chroma.model.TorchDevice.empty_cache", empty_cache)

    positive_txt = torch.randn(1, 3, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    positive_extension = _chroma_cfg_extension(
        embeddings=positive_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids,
    )
    negative_extension = _chroma_cfg_extension(
        embeddings=negative_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids.clone(),
    )

    adapter = ChromaTransformerAdapter(model)
    adapter.enable_batched_cfg(negative_extension=negative_extension, cfg_scale=[3.0])
    result = adapter(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        txt=positive_txt,
        txt_ids=txt_ids,
        y=torch.zeros(1, 768),
        timesteps=torch.tensor([0.5]),
        guidance=torch.zeros(1),
        timestep_index=0,
        total_num_timesteps=1,
        controlnet_double_block_residuals=None,
        controlnet_single_block_residuals=None,
        ip_adapter_extensions=[],
        regional_prompting_extension=positive_extension,
    )

    assert torch.equal(result, torch.full_like(result, 4.0))
    assert model.call_count == 3
    assert model.call_args_list[0].kwargs["hidden_states"].shape[0] == 2
    assert model.call_args_list[1].kwargs["hidden_states"].shape[0] == 1
    assert model.call_args_list[2].kwargs["hidden_states"].shape[0] == 1
    empty_cache.assert_called_once_with()


def test_chroma_euler_cfg_pp_step_matches_rectified_flow_reference() -> None:
    img = torch.tensor([[[10.0]]])
    positive_pred = torch.tensor([[[2.0]]])
    negative_pred = torch.tensor([[[1.0]]])

    next_img, guided_denoised = euler_cfg_pp_step(
        img=img,
        positive_pred=positive_pred,
        negative_pred=negative_pred,
        sigma=0.5,
        sigma_next=0.25,
        cfg_scale=3.0,
    )

    assert torch.equal(guided_denoised, torch.tensor([[[8.0]]]))
    assert torch.equal(next_img, torch.tensor([[[8.625]]]))


def test_chroma_euler_cfg_pp_terminal_step_returns_guided_denoised() -> None:
    img = torch.tensor([[[10.0]]])
    positive_pred = torch.tensor([[[2.0]]])
    negative_pred = torch.tensor([[[1.0]]])

    next_img, guided_denoised = euler_cfg_pp_step(
        img=img,
        positive_pred=positive_pred,
        negative_pred=negative_pred,
        sigma=0.5,
        sigma_next=0.0,
        cfg_scale=3.0,
    )

    assert torch.equal(guided_denoised, torch.tensor([[[8.0]]]))
    assert torch.equal(next_img, guided_denoised)


def test_chroma_transformer_adapter_returns_cfg_branches_in_one_batched_call() -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.return_value = (torch.cat((positive_pred, negative_pred), dim=0),)

    positive_txt = torch.randn(1, 3, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    positive_extension = _chroma_cfg_extension(
        embeddings=positive_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids,
    )
    negative_extension = _chroma_cfg_extension(
        embeddings=negative_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids.clone(),
    )

    adapter = ChromaTransformerAdapter(model)
    result_positive, result_negative = adapter.predict_cfg_branches(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        timesteps=torch.tensor([0.5]),
        positive_extension=positive_extension,
        negative_extension=negative_extension,
        allow_batched=True,
    )

    assert torch.equal(result_positive, positive_pred)
    assert torch.equal(result_negative, negative_pred)
    model.assert_called_once()
    assert model.call_args.kwargs["hidden_states"].shape[0] == 2


def test_chroma_transformer_adapter_returns_cfg_branches_sequentially_when_requested() -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.side_effect = [(positive_pred,), (negative_pred,)]

    positive_txt = torch.randn(1, 3, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    positive_extension = _chroma_cfg_extension(
        embeddings=positive_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids,
    )
    negative_extension = _chroma_cfg_extension(
        embeddings=negative_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids.clone(),
    )

    result_positive, result_negative = ChromaTransformerAdapter(model).predict_cfg_branches(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        timesteps=torch.tensor([0.5]),
        positive_extension=positive_extension,
        negative_extension=negative_extension,
        allow_batched=False,
    )

    assert torch.equal(result_positive, positive_pred)
    assert torch.equal(result_negative, negative_pred)
    assert model.call_count == 2
    assert all(call.kwargs["hidden_states"].shape[0] == 1 for call in model.call_args_list)


def test_chroma_transformer_adapter_cfg_branches_fall_back_after_batched_oom(monkeypatch) -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.side_effect = [
        torch.OutOfMemoryError("synthetic CFG++ batched OOM"),
        (positive_pred,),
        (negative_pred,),
    ]
    empty_cache = MagicMock()
    monkeypatch.setattr("invokeai.backend.chroma.model.TorchDevice.empty_cache", empty_cache)

    positive_txt = torch.randn(1, 3, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    txt_ids = torch.zeros(1, 3, 3)
    positive_extension = _chroma_cfg_extension(
        embeddings=positive_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids,
    )
    negative_extension = _chroma_cfg_extension(
        embeddings=negative_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=txt_ids.clone(),
    )

    result_positive, result_negative = ChromaTransformerAdapter(model).predict_cfg_branches(
        img=torch.randn(1, 4, 64),
        img_ids=torch.zeros(1, 4, 3),
        timesteps=torch.tensor([0.5]),
        positive_extension=positive_extension,
        negative_extension=negative_extension,
        allow_batched=True,
    )

    assert torch.equal(result_positive, positive_pred)
    assert torch.equal(result_negative, negative_pred)
    assert model.call_count == 3
    assert model.call_args_list[0].kwargs["hidden_states"].shape[0] == 2
    assert model.call_args_list[1].kwargs["hidden_states"].shape[0] == 1
    assert model.call_args_list[2].kwargs["hidden_states"].shape[0] == 1
    empty_cache.assert_called_once_with()


def test_chroma_euler_cfg_pp_keeps_negative_branch_at_cfg_one() -> None:
    adapter = MagicMock(spec=ChromaTransformerAdapter)
    positive_pred = torch.full((1, 1, 1), 2.0)
    negative_pred = torch.full((1, 1, 1), 1.0)
    adapter.predict_cfg_branches.return_value = (positive_pred, negative_pred)
    callback = MagicMock()
    positive_extension = MagicMock()
    negative_extension = MagicMock()

    result = denoise_euler_cfg_pp(
        model=adapter,
        img=torch.tensor([[[10.0]]]),
        img_ids=torch.zeros(1, 1, 3),
        positive_extension=positive_extension,
        negative_extension=negative_extension,
        timesteps=[0.5, 0.0],
        cfg_scale=[1.0],
        step_callback=callback,
        inpaint_extension=None,
        allow_batched_cfg=True,
    )

    # CFG=1 still needs the unconditional branch because CFG++ uses it to define
    # the transition direction. The terminal x0 itself is the positive branch.
    adapter.predict_cfg_branches.assert_called_once()
    assert adapter.predict_cfg_branches.call_args.kwargs["negative_extension"] is negative_extension
    assert torch.equal(result, torch.tensor([[[9.0]]]))
    callback.assert_called_once()
