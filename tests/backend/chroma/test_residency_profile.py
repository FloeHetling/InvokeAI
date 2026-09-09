from unittest.mock import Mock

import pytest

from invokeai.backend.chroma.residency_profile import (
    CHROMA_RESIDENCY_PROFILE_ENV_VAR,
    ChromaResidencyProfiler,
)


class _SingleArgumentLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_residency_profiler_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, raising=False)

    assert ChromaResidencyProfiler.create_if_enabled(Mock()) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_residency_profiler_accepts_truthy_environment_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, value)

    logger = Mock()
    profiler = ChromaResidencyProfiler.create_if_enabled(logger)

    assert profiler is not None
    logger.info.assert_called_once()


def test_residency_profiler_rejects_other_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, "0")

    assert ChromaResidencyProfiler.create_if_enabled(Mock()) is None


def test_residency_profiler_uses_single_argument_logger_contract() -> None:
    logger = _SingleArgumentLogger()
    profiler = ChromaResidencyProfiler(logger)
    profiler.begin_step(timestep_index=0, total_num_timesteps=2)

    profiler.log_controlnet_skip(label="control", weight=0.5)
    profiler.log_controlnet(
        label="control",
        weight=0.5,
        acquire_ms=1.0,
        forward_ms=2.0,
        release_ms=3.0,
        mem_before="before",
        mem_after_acquire="acquired",
        mem_after_forward="forwarded",
        mem_after_release="released",
    )
    profiler.log_chroma(
        controlled=True,
        acquire_ms=4.0,
        forward_ms=5.0,
        release_ms=6.0,
        mem_before="before",
        mem_after_acquire="acquired",
        mem_after_forward="forwarded",
        mem_after_release="released",
    )
    profiler.log_summary()

    assert len(logger.messages) == 5
    assert "step=1/2 phase=controlnet" in logger.messages[1]
    assert "acquire_ms=1.00 forward_ms=2.00 release_ms=3.00" in logger.messages[2]
    assert "phase=chroma call=1 controlled=1" in logger.messages[3]
    assert "controlnet_calls=1 controlnet_skips=1" in logger.messages[4]
