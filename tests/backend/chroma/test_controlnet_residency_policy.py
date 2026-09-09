import pytest

from invokeai.app.invocations.chroma_denoise import (
    CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR,
    _get_chroma_controlnet_residency_policy,
)

PRO_2_SOURCE = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
UNION_V1_SOURCE = "InstantX/FLUX.1-dev-Controlnet-Union"


def test_chroma_controlnet_residency_policy_auto_pins_validated_pro_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, raising=False)

    assert _get_chroma_controlnet_residency_policy([PRO_2_SOURCE]) == "chroma_pinned"


@pytest.mark.parametrize(
    "sources",
    [
        [],
        [UNION_V1_SOURCE],
        ["C:/models/local-controlnet"],
        [PRO_2_SOURCE, PRO_2_SOURCE],
    ],
)
def test_chroma_controlnet_residency_policy_auto_uses_phase_swap_for_unvalidated_workloads(
    monkeypatch: pytest.MonkeyPatch, sources: list[str]
) -> None:
    monkeypatch.delenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, raising=False)

    assert _get_chroma_controlnet_residency_policy(sources) == "phase_swap"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("phase_swap", "phase_swap"),
        ("PHASE_SWAP", "phase_swap"),
        ("chroma_pinned", "chroma_pinned"),
        ("CHROMA_PINNED", "chroma_pinned"),
        ("auto", "chroma_pinned"),
        ("AUTO", "chroma_pinned"),
    ],
)
def test_chroma_controlnet_residency_policy_accepts_supported_values(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: str
) -> None:
    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, value)

    assert _get_chroma_controlnet_residency_policy([PRO_2_SOURCE]) == expected


def test_chroma_controlnet_residency_policy_explicit_overrides_ignore_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, "chroma_pinned")
    assert _get_chroma_controlnet_residency_policy([UNION_V1_SOURCE]) == "chroma_pinned"

    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, "phase_swap")
    assert _get_chroma_controlnet_residency_policy([PRO_2_SOURCE]) == "phase_swap"


def test_chroma_controlnet_residency_policy_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHROMA_CONTROLNET_RESIDENCY_POLICY_ENV_VAR, "mystery")

    with pytest.raises(ValueError, match="expected 'auto', 'phase_swap', or 'chroma_pinned'"):
        _get_chroma_controlnet_residency_policy([PRO_2_SOURCE])
