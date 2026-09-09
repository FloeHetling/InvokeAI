"""Opt-in profiling for Chroma/ControlNet model-cache residency transitions."""

from __future__ import annotations

import os
import time
from typing import Any

import torch

CHROMA_RESIDENCY_PROFILE_ENV_VAR = "INVOKEAI_CHROMA_RESIDENCY_PROFILE"
_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled_from_environment() -> bool:
    return os.getenv(CHROMA_RESIDENCY_PROFILE_ENV_VAR, "").strip().lower() in _TRUE_VALUES


class ChromaResidencyProfiler:
    """Collect synchronized phase timings without changing the denoising math.

    CUDA synchronization is intentionally used while profiling so model-cache transfer
    work is charged to the acquire phase that triggered it instead of leaking into the
    following forward. This perturbs wall-clock performance, so the profiler is opt-in.
    """

    def __init__(self, logger: Any) -> None:
        self._logger = logger
        self._current_step: int | None = None
        self._total_steps = 0
        self._chroma_call_index = 0

        self._controlnet_calls = 0
        self._controlnet_skips = 0
        self._controlnet_acquire_ms = 0.0
        self._controlnet_forward_ms = 0.0
        self._controlnet_release_ms = 0.0
        self._chroma_calls = 0
        self._chroma_acquire_ms = 0.0
        self._chroma_forward_ms = 0.0
        self._chroma_release_ms = 0.0

        self._logger.info(
            "CHROMA_RESIDENCY_PROFILE enabled; CUDA synchronization will perturb wall-clock timing while profiling"
        )

    @classmethod
    def create_if_enabled(cls, logger: Any) -> ChromaResidencyProfiler | None:
        if not _is_enabled_from_environment():
            return None
        return cls(logger)

    def begin_step(self, timestep_index: int, total_num_timesteps: int) -> None:
        if self._current_step != timestep_index:
            self._current_step = timestep_index
            self._total_steps = total_num_timesteps
            self._chroma_call_index = 0

    @staticmethod
    def sync_and_now(device: torch.device) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    @staticmethod
    def memory_snapshot(device: torch.device) -> str:
        if device.type != "cuda":
            return f"device={device.type}"
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            allocated_bytes = torch.cuda.memory_allocated(device)
            reserved_bytes = torch.cuda.memory_reserved(device)
        except RuntimeError as error:
            return f"unavailable={type(error).__name__}"

        gib = 2**30
        return (
            f"alloc={allocated_bytes / gib:.2f}GiB,reserved={reserved_bytes / gib:.2f}GiB,"
            f"free={free_bytes / gib:.2f}GiB,total={total_bytes / gib:.2f}GiB"
        )

    def log_controlnet_skip(self, *, label: str, weight: float) -> None:
        self._controlnet_skips += 1
        self._logger.info(
            "CHROMA_RESIDENCY_PROFILE step=%s/%s phase=controlnet model=%s skipped=1 weight=%.6f",
            self._display_step(),
            self._total_steps,
            label,
            weight,
        )

    def log_controlnet(
        self,
        *,
        label: str,
        weight: float,
        acquire_ms: float,
        forward_ms: float,
        release_ms: float,
        mem_before: str,
        mem_after_acquire: str,
        mem_after_forward: str,
        mem_after_release: str,
    ) -> None:
        self._controlnet_calls += 1
        self._controlnet_acquire_ms += acquire_ms
        self._controlnet_forward_ms += forward_ms
        self._controlnet_release_ms += release_ms
        self._logger.info(
            "CHROMA_RESIDENCY_PROFILE step=%s/%s phase=controlnet model=%s weight=%.6f "
            "acquire_ms=%.2f forward_ms=%.2f release_ms=%.2f mem_before=[%s] "
            "mem_after_acquire=[%s] mem_after_forward=[%s] mem_after_release=[%s]",
            self._display_step(),
            self._total_steps,
            label,
            weight,
            acquire_ms,
            forward_ms,
            release_ms,
            mem_before,
            mem_after_acquire,
            mem_after_forward,
            mem_after_release,
        )

    def log_chroma(
        self,
        *,
        controlled: bool,
        acquire_ms: float,
        forward_ms: float,
        release_ms: float,
        mem_before: str,
        mem_after_acquire: str,
        mem_after_forward: str,
        mem_after_release: str,
    ) -> None:
        self._chroma_call_index += 1
        self._chroma_calls += 1
        self._chroma_acquire_ms += acquire_ms
        self._chroma_forward_ms += forward_ms
        self._chroma_release_ms += release_ms
        self._logger.info(
            "CHROMA_RESIDENCY_PROFILE step=%s/%s phase=chroma call=%s controlled=%s "
            "acquire_ms=%.2f forward_ms=%.2f release_ms=%.2f mem_before=[%s] "
            "mem_after_acquire=[%s] mem_after_forward=[%s] mem_after_release=[%s]",
            self._display_step(),
            self._total_steps,
            self._chroma_call_index,
            int(controlled),
            acquire_ms,
            forward_ms,
            release_ms,
            mem_before,
            mem_after_acquire,
            mem_after_forward,
            mem_after_release,
        )

    def log_summary(self) -> None:
        self._logger.info(
            "CHROMA_RESIDENCY_PROFILE summary controlnet_calls=%s controlnet_skips=%s "
            "controlnet_acquire_ms=%.2f controlnet_forward_ms=%.2f controlnet_release_ms=%.2f "
            "chroma_calls=%s chroma_acquire_ms=%.2f chroma_forward_ms=%.2f chroma_release_ms=%.2f",
            self._controlnet_calls,
            self._controlnet_skips,
            self._controlnet_acquire_ms,
            self._controlnet_forward_ms,
            self._controlnet_release_ms,
            self._chroma_calls,
            self._chroma_acquire_ms,
            self._chroma_forward_ms,
            self._chroma_release_ms,
        )

    def _display_step(self) -> str:
        return "?" if self._current_step is None else str(self._current_step + 1)
