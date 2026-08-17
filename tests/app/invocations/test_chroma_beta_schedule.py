import pytest

from invokeai.backend.chroma.schedulers import get_chroma_beta_schedule


def test_chroma_beta_schedule_matches_25_step_reference() -> None:
    # Current Chroma reference: Beta(0.6, 0.6) quantiles over the
    # 10,000-point float32 ModelSamplingFlux table shifted by 1.15.
    expected = [
        1.0,
        0.997228801,
        0.991158664,
        0.982451260,
        0.971211910,
        0.957517445,
        0.941276312,
        0.922322094,
        0.900541723,
        0.875612497,
        0.847310960,
        0.815341830,
        0.779252410,
        0.738673687,
        0.693239987,
        0.642623246,
        0.586361885,
        0.524296820,
        0.456678838,
        0.383749068,
        0.306970119,
        0.228410900,
        0.151473656,
        0.0819845051,
        0.0272741020,
        0.0,
    ]

    assert get_chroma_beta_schedule(25) == pytest.approx(expected, abs=5e-10)


def test_chroma_beta_schedule_is_descending_and_terminal_zero() -> None:
    schedule = get_chroma_beta_schedule(40)

    assert schedule[0] == 1.0
    assert schedule[-1] == 0.0
    assert all(a > b for a, b in zip(schedule[:-1], schedule[1:], strict=True))


def test_chroma_beta_schedule_rejects_non_positive_step_count() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        get_chroma_beta_schedule(0)
