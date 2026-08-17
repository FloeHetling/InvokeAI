import pytest

from invokeai.backend.chroma.schedulers import get_chroma_beta_schedule


def test_chroma_beta_schedule_matches_25_step_reference() -> None:
    # Reference values for Beta(0.6, 0.6) quantiles mapped onto the
    # 1000-point discrete-flow sigma table.
    expected = [
        1.0,
        0.991,
        0.973,
        0.947,
        0.914,
        0.877,
        0.836,
        0.790,
        0.742,
        0.691,
        0.638,
        0.583,
        0.528,
        0.473,
        0.418,
        0.363,
        0.310,
        0.259,
        0.211,
        0.165,
        0.124,
        0.087,
        0.054,
        0.028,
        0.010,
        0.0,
    ]

    assert get_chroma_beta_schedule(25) == pytest.approx(expected, abs=1e-12)


def test_chroma_beta_schedule_is_descending_and_terminal_zero() -> None:
    schedule = get_chroma_beta_schedule(40)

    assert schedule[0] == 1.0
    assert schedule[-1] == 0.0
    assert all(a > b for a, b in zip(schedule[:-1], schedule[1:], strict=True))


def test_chroma_beta_schedule_rejects_non_positive_step_count() -> None:
    with pytest.raises(ValueError, match="num_steps"):
        get_chroma_beta_schedule(0)
