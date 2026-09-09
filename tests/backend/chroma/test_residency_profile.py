from unittest.mock import Mock

import pytest

from invokeai.backend.chroma.residency_profile import (
    CHROMA_RESIDENCY_PROFILE_ENV_VAR,
    ChromaResidencyProfiler,
)


def test_residency_profiler_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, raising=False)

    assert ChromaResidencyProfiler.create_if_enabled(Mock()) is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_residency_profiler_accepts_truthy_environment_values(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, value)

    logger = Mock()
    profiler = ChromaResidencyProfiler.create_if_enabled(logger)

    assert profiler is not None
    logger.info.assert_called_once()


def test_residency_profiler_rejects_other_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, "0")

    assert ChromaResidencyProfiler.create_if_enabled(Mock()) is None
