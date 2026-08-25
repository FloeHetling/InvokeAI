import pytest

from invokeai.app.invocations.flux_text_encoder import FluxTextEncoderInvocation
from invokeai.backend.model_manager.taxonomy import ModelFormat


@pytest.mark.parametrize(
    ["model_format", "expected"],
    [
        (ModelFormat.T5Encoder, False),
        (ModelFormat.Diffusers, False),
        (ModelFormat.Checkpoint, False),
        (ModelFormat.BnbQuantizedLlmInt8b, True),
        (ModelFormat.BnbQuantizednf4b, True),
        (ModelFormat.GGUFQuantized, True),
        (ModelFormat.SDNQQuantized, True),
    ],
)
def test_t5_model_quantization_policy(model_format: ModelFormat, expected: bool):
    assert FluxTextEncoderInvocation._is_t5_model_quantized(model_format) is expected


def test_unsupported_t5_model_format_raises():
    with pytest.raises(ValueError, match="Unsupported model format"):
        FluxTextEncoderInvocation._is_t5_model_quantized(ModelFormat.ONNX)
