from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
import torch

from invokeai.app.invocations.text_encoder.chroma_text_encoder import (
    ChromaTextEncoderInvocation,
)
from invokeai.backend.chroma.model import ChromaTransformerAdapter


def _extension(
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


def _tokenizer_for_segments(mapping: dict[str, list[int]]) -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 1
    tokenizer.pad_token_id = 0
    tokenizer.side_effect = lambda text, **_kwargs: {"input_ids": mapping[text] + [1]}
    return tokenizer


def test_chroma_prompt_attention_parses_parenthesis_weights() -> None:
    segments = ChromaTextEncoderInvocation._parse_prompt_attention(
        r"plain (boosted) ((nested)) (explicit:1.35) escaped \(literal\)"
    )

    assert segments == [
        ("plain ", 1.0),
        ("boosted", 1.1),
        (" ", 1.0),
        ("nested", pytest.approx(1.21)),
        (" ", 1.0),
        ("explicit", 1.35),
        (" escaped (literal)", 1.0),
    ]


def test_chroma_prompt_attention_reads_a_weight_after_the_closing_parenthesis() -> None:
    segments = ChromaTextEncoderInvocation._parse_prompt_attention("a (b c)1.3, (d).5 (e)2.")

    assert segments == [
        ("a ", 1.0),
        ("b c", 1.3),
        (", ", 1.0),
        ("d", 0.5),
        (" ", 1.0),
        ("e", 2.0),
        (".", 1.0),
    ]


def test_chroma_prompt_attention_multiplies_nested_weights() -> None:
    segments = ChromaTextEncoderInvocation._parse_prompt_attention("(a (b:1.5) (c)2 (d):1.2)")

    assert segments == [
        ("a ", pytest.approx(1.2)),
        ("b", pytest.approx(1.8)),
        (" ", pytest.approx(1.2)),
        ("c", pytest.approx(2.4)),
        (" ", pytest.approx(1.2)),
        ("d", pytest.approx(1.32)),
    ]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("(a)1.5x", [("a", 1.1), ("1.5x", 1.0)]),
        ("(a)1.3.5", [("a", 1.1), ("1.3.5", 1.0)]),
        ("(a)2cats", [("a", 1.1), ("2cats", 1.0)]),
        ("(a)-1", [("a", 1.1), ("-1", 1.0)]),
        ("(a:nan)", [("a:nan", 1.1)]),
        ("(a:inf)", [("a:inf", 1.1)]),
        ("(a:-1)", [("a:-1", 1.1)]),
        ("(a:1e3)", [("a:1e3", 1.1)]),
        ("(a: 1_0)", [("a: 1_0", 1.1)]),
    ],
)
def test_chroma_prompt_attention_keeps_malformed_weights_as_text(
    prompt: str, expected: list[tuple[str, float]]
) -> None:
    assert ChromaTextEncoderInvocation._parse_prompt_attention(prompt) == [
        (text, pytest.approx(weight)) for text, weight in expected
    ]


def test_chroma_text_encoder_tokenizes_weighted_segments_then_appends_one_eos() -> None:
    tokenizer = _tokenizer_for_segments(
        {
            "alpha ": [10, 11],
            "beta": [20],
            " gamma": [30, 31],
        }
    )

    result = ChromaTextEncoderInvocation.tokenize_prompt(tokenizer, "alpha (beta) gamma")

    assert result.input_ids.tolist() == [[10, 11, 20, 30, 31, 1]]
    assert result.token_weights[0].tolist() == pytest.approx([1.0, 1.0, 1.1, 1.0, 1.0, 1.0])
    assert result.has_weights
    assert tokenizer.call_args_list == [
        call("alpha ", add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False),
        call("beta", add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False),
        call(" gamma", add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False),
    ]


def test_chroma_prompt_weighting_matches_embedding_formula() -> None:
    prompt = torch.tensor([[[3.0, 5.0], [7.0, 11.0], [13.0, 17.0]]], dtype=torch.bfloat16)
    baseline = torch.tensor([[[1.0, 1.0], [2.0, 3.0], [5.0, 7.0]]], dtype=torch.bfloat16)
    weights = torch.tensor([[1.0, 1.5, 0.5]], dtype=torch.float32)

    result = ChromaTextEncoderInvocation.apply_prompt_weights(prompt, baseline, weights)

    expected = (prompt.float() - baseline.float()) * weights.unsqueeze(-1) + baseline.float()
    assert result.dtype is torch.float32
    assert torch.equal(result, expected)


def test_chroma_batched_cfg_pads_different_text_lengths_and_masks_the_padding() -> None:
    model = MagicMock()
    positive_pred = torch.full((1, 4, 64), 2.0)
    negative_pred = torch.full((1, 4, 64), 1.0)
    model.return_value = (torch.cat((positive_pred, negative_pred), dim=0),)

    positive_txt = torch.randn(1, 5, 4096)
    negative_txt = torch.randn(1, 3, 4096)
    positive_extension = _extension(
        embeddings=positive_txt,
        attention_mask=torch.ones(1, 5, dtype=torch.bool),
        txt_ids=torch.zeros(1, 5, 3),
    )
    negative_extension = _extension(
        embeddings=negative_txt,
        attention_mask=torch.ones(1, 3, dtype=torch.bool),
        txt_ids=torch.zeros(1, 3, 3),
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
    model.assert_called_once()

    call_args = model.call_args.kwargs
    assert call_args["encoder_hidden_states"].shape == (2, 5, 4096)
    assert torch.count_nonzero(call_args["encoder_hidden_states"][1, 3:]) == 0
    assert call_args["txt_ids"].shape == (5, 3)
    assert call_args["attention_mask"].tolist() == [
        [True, True, True, True, True, True, True, True, True],
        [True, True, True, False, False, True, True, True, True],
    ]
