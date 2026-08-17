from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Generator

import torch

_MIN_STAGING_BYTES = 1 * 2**20
_MAX_SLOT_BYTES = 96 * 2**20
_NUM_STAGING_SLOTS = 2


@dataclass
class AsyncLinearWeightStagingStats:
    staged_tensors: int = 0
    staged_bytes: int = 0
    synchronous_fallbacks: int = 0
    slot_waits: int = 0


@dataclass
class _StagingSlot:
    stream: torch.cuda.Stream
    copy_done: torch.cuda.Event
    use_done: torch.cuda.Event
    host_buffer: torch.Tensor | None = None
    device_buffer: torch.Tensor | None = None
    buffer_dtype: torch.dtype | None = None
    copy_pending: bool = False
    awaiting_consumption: bool = False
    use_pending: bool = False


class CudaAsyncLinearWeightStager:
    """Bounded two-slot staging for large CPU-resident linear weights.

    Each slot owns one pinned host buffer and one persistent CUDA buffer. The CUDA buffer is allocated on the slot's
    staging stream and is never reallocated during the denoise. H->D copies wait device-side for the previous GEMM
    that consumed the slot, while the compute stream waits only for the new copy to finish.

    The two persistent slots avoid the per-weight cross-stream allocator churn of allocating a fresh CUDA tensor for
    every streamed linear weight.
    """

    def __init__(self, device: torch.device):
        if device.type != "cuda":
            raise ValueError("CUDA async linear weight staging requires a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())

        self._device = device
        self._slots = [
            _StagingSlot(
                stream=torch.cuda.Stream(device=device),
                copy_done=torch.cuda.Event(blocking=False),
                use_done=torch.cuda.Event(blocking=False),
            )
            for _ in range(_NUM_STAGING_SLOTS)
        ]
        self._next_slot = 0
        self._inflight: dict[int, _StagingSlot] = {}
        self._disabled = False
        self.stats = AsyncLinearWeightStagingStats()

    def _is_eligible(self, tensor: torch.Tensor | None, input: torch.Tensor) -> bool:
        if self._disabled or tensor is None:
            return False
        if tensor.device.type != "cpu" or input.device.type != "cuda":
            return False
        if input.device != self._device:
            return False
        if not tensor.is_floating_point() or tensor.dtype != input.dtype:
            return False
        if not tensor.is_contiguous():
            return False

        tensor_bytes = tensor.numel() * tensor.element_size()
        return _MIN_STAGING_BYTES <= tensor_bytes <= _MAX_SLOT_BYTES

    def _select_slot(self) -> _StagingSlot | None:
        for offset in range(len(self._slots)):
            slot_index = (self._next_slot + offset) % len(self._slots)
            slot = self._slots[slot_index]

            # A slot returned to CustomLinear is reserved until F.linear() has been enqueued and mark_consumed()
            # records the use_done event. This also makes the implementation safe if a future module stages more than
            # one eligible tensor before launching its compute kernel.
            if slot.awaiting_consumption:
                continue

            # The pinned host buffer cannot be overwritten while an earlier H->D transfer is still reading it.
            # Keep this selection path non-blocking; try_stage() can wait for a reusable slot if both copies are busy.
            if slot.copy_pending and not slot.copy_done.query():
                continue

            slot.copy_pending = False
            self._next_slot = (slot_index + 1) % len(self._slots)
            return slot

        return None

    def _wait_for_reusable_slot(self) -> _StagingSlot | None:
        """Wait only for a prior H->D copy to release a pinned host buffer.

        A slot that is still awaiting consumption by CustomLinear must not be reused. For a consumed slot, however,
        copy_done is sufficient to make its pinned host buffer writable again. The staging stream will separately
        wait on use_done before overwriting the persistent CUDA buffer.
        """
        for offset in range(len(self._slots)):
            slot_index = (self._next_slot + offset) % len(self._slots)
            slot = self._slots[slot_index]
            if slot.awaiting_consumption:
                continue

            if slot.copy_pending:
                slot.copy_done.synchronize()
                slot.copy_pending = False

            self._next_slot = (slot_index + 1) % len(self._slots)
            return slot
        return None

    def _ensure_buffers(self, slot: _StagingSlot, tensor: torch.Tensor) -> bool:
        if slot.host_buffer is not None or slot.device_buffer is not None:
            # Chroma uses FP16 transformer inputs. Keep this path narrow rather than reallocating persistent
            # cross-stream buffers if a different dtype unexpectedly appears.
            return slot.host_buffer is not None and slot.device_buffer is not None and slot.buffer_dtype == tensor.dtype

        capacity_numel = _MAX_SLOT_BYTES // tensor.element_size()
        try:
            host_buffer = torch.empty(capacity_numel, dtype=tensor.dtype, device="cpu", pin_memory=True)
            # Allocation and first ownership of the CUDA buffer happen on the stream that will overwrite it. This is
            # important for safe reuse with PyTorch's stream-aware caching allocator.
            with torch.cuda.stream(slot.stream):
                device_buffer = torch.empty(capacity_numel, dtype=tensor.dtype, device=self._device)
        except RuntimeError:
            return False

        slot.host_buffer = host_buffer
        slot.device_buffer = device_buffer
        slot.buffer_dtype = tensor.dtype
        return True

    @torch.no_grad()
    def try_stage(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        if not self._is_eligible(tensor, input):
            return None
        assert tensor is not None

        slot = self._select_slot()
        if slot is None:
            slot = self._wait_for_reusable_slot()
            if slot is None:
                self.stats.synchronous_fallbacks += 1
                return None
            self.stats.slot_waits += 1

        if not self._ensure_buffers(slot, tensor):
            self._disabled = True
            self.stats.synchronous_fallbacks += 1
            return None

        assert slot.host_buffer is not None
        assert slot.device_buffer is not None

        numel = tensor.numel()
        host_flat = slot.host_buffer[:numel]
        device_flat = slot.device_buffer[:numel]
        host_flat.copy_(tensor.detach().view(-1))

        with torch.cuda.stream(slot.stream):
            if slot.use_pending:
                # The next H->D overwrite cannot start until the previous GEMM using this persistent device buffer
                # has completed. This is a device-side dependency and does not block the Python launch thread.
                slot.stream.wait_event(slot.use_done)
            device_flat.copy_(host_flat, non_blocking=True)
            slot.copy_done.record()

        slot.copy_pending = True
        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_event(slot.copy_done)

        staged = device_flat.view(tensor.shape)
        slot.awaiting_consumption = True
        self._inflight[id(staged)] = slot

        tensor_bytes = tensor.numel() * tensor.element_size()
        self.stats.staged_tensors += 1
        self.stats.staged_bytes += tensor_bytes
        return staged

    def mark_consumed(self, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return

        slot = self._inflight.pop(id(tensor), None)
        if slot is None:
            return

        # Called immediately after F.linear() is enqueued on the compute stream. The slot's staging stream will wait
        # on this event before overwriting its persistent device buffer on the next reuse.
        slot.use_done.record(torch.cuda.current_stream(self._device))
        slot.awaiting_consumption = False
        slot.use_pending = True

    def close(self) -> None:
        """Wait for outstanding slot work before releasing the persistent buffers."""
        for slot in self._slots:
            if slot.awaiting_consumption:
                # An exception may have happened after staging but before the corresponding F.linear().
                slot.copy_done.synchronize()
            elif slot.use_pending:
                slot.use_done.synchronize()
            elif slot.copy_pending:
                slot.copy_done.synchronize()
        self._inflight.clear()


_ACTIVE_STAGER: ContextVar[CudaAsyncLinearWeightStager | None] = ContextVar(
    "invokeai_cuda_async_linear_weight_stager", default=None
)


@contextmanager
def cuda_async_linear_weight_staging(
    device: torch.device,
) -> Generator[CudaAsyncLinearWeightStager | None, None, None]:
    """Enable bounded async linear-weight staging for the current execution context.

    The context is a no-op off CUDA. It is currently entered only by Chroma denoise, keeping the optimization isolated
    from the rest of the model cache while preserving the normal synchronous path as a fallback.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        yield None
        return

    existing = _ACTIVE_STAGER.get()
    if existing is not None:
        yield existing
        return

    stager = CudaAsyncLinearWeightStager(device)
    token = _ACTIVE_STAGER.set(stager)
    try:
        yield stager
    finally:
        try:
            stager.close()
        finally:
            _ACTIVE_STAGER.reset(token)


@contextmanager
def suspend_cuda_async_linear_weight_staging() -> Generator[None, None, None]:
    """Temporarily disable async linear-weight staging while preserving the active stager for the outer context."""
    token = _ACTIVE_STAGER.set(None)
    try:
        yield
    finally:
        _ACTIVE_STAGER.reset(token)


def maybe_stage_tensor_for_input(tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
    """Return an asynchronously staged tensor, or None when the caller should use the normal cast path."""
    stager = _ACTIVE_STAGER.get()
    if stager is None:
        return None
    return stager.try_stage(tensor, input)


def mark_staged_tensor_consumed(tensor: torch.Tensor | None) -> None:
    stager = _ACTIVE_STAGER.get()
    if stager is not None:
        stager.mark_consumed(tensor)
