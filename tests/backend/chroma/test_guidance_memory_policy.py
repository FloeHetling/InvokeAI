import pytest

from invokeai.app.invocations.chroma_denoise import _resolve_chroma_sequential_guidance

GIB = 2**30


def test_explicit_sequential_guidance_always_wins() -> None:
    sequential, reason = _resolve_chroma_sequential_guidance(
        requested=True,
        transformer_size_bytes=17 * GIB,
        device_working_mem_bytes=3 * GIB,
        total_device_vram_bytes=48 * GIB,
    )

    assert sequential is True
    assert reason is None


def test_unknown_device_memory_keeps_batched_guidance_enabled() -> None:
    sequential, reason = _resolve_chroma_sequential_guidance(
        requested=False,
        transformer_size_bytes=17 * GIB,
        device_working_mem_bytes=3 * GIB,
        total_device_vram_bytes=None,
    )

    assert sequential is False
    assert reason is None


@pytest.mark.parametrize(
    ("total_device_vram_bytes", "expected_sequential"),
    [
        (16 * GIB, True),
        (22 * GIB, True),
        (24 * GIB, False),
    ],
)
def test_batched_guidance_requires_full_model_plus_two_working_memory_reserves(
    total_device_vram_bytes: int,
    expected_sequential: bool,
) -> None:
    sequential, reason = _resolve_chroma_sequential_guidance(
        requested=False,
        transformer_size_bytes=int(16.58 * GIB),
        device_working_mem_bytes=3 * GIB,
        total_device_vram_bytes=total_device_vram_bytes,
    )

    assert sequential is expected_sequential
    assert (reason is not None) is expected_sequential


def test_batched_guidance_is_allowed_exactly_at_the_required_vram_boundary() -> None:
    transformer_size_bytes = 10 * GIB
    working_mem_bytes = 2 * GIB
    required_vram_bytes = transformer_size_bytes + 2 * working_mem_bytes

    sequential, reason = _resolve_chroma_sequential_guidance(
        requested=False,
        transformer_size_bytes=transformer_size_bytes,
        device_working_mem_bytes=working_mem_bytes,
        total_device_vram_bytes=required_vram_bytes,
    )

    assert sequential is False
    assert reason is None


def test_auto_sequential_reason_reports_the_admission_numbers() -> None:
    sequential, reason = _resolve_chroma_sequential_guidance(
        requested=False,
        transformer_size_bytes=10 * GIB,
        device_working_mem_bytes=3 * GIB,
        total_device_vram_bytes=12 * GIB,
    )

    assert sequential is True
    assert reason is not None
    assert "transformer=10.00 GiB" in reason
    assert "working reserve=3.00 GiB per branch" in reason
    assert "batched requirement=16.00 GiB" in reason
    assert "device total=12.00 GiB" in reason
