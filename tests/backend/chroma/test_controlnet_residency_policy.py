import pytest

from invokeai.app.invocations.chroma_denoise import (
    CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR,
    _get_chroma_controlnet_residency_policy,
)


def test_chroma_controlnet_residency_policy_defaults_to_phase_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, raising=False)

    assert _get_chroma_controlnet_residency_policy() == "phase_swap"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("phase_swap", "phase_swap"),
        ("PHASE_SWAP", "phase_swap"),
        ("chroma_pinned", "chroma_pinned"),
        ("CHROMA_PINNED", "chroma_pinned"),
    ],
)
def test_chroma_controlnet_residency_policy_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, value)

    assert _get_chroma_controlnet_residency_policy() == expected


def test_chroma_controlnet_residency_policy_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, "mystery")

    with pytest.raises(ValueError, match="expected 'phase_swap' or 'chroma_pinned'"):
        _get_chroma_controlnet_residency_policy()
