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


@dataclass
class _StagingSlot:
    stream: torch.cuda.Stream
    copy_done: torch.cuda.Event
    host_buffer: torch.Tensor | None = None
    buffer_dtype: torch.dtype | None = None
    copy_pending: bool = False


class CudaAsyncLinearWeightStager:
    """Bounded two-slot staging for large CPU-resident linear weights.

    This is intentionally narrow. It only stages contiguous floating-point tensors that already have the compute
    dtype. Pageable CPU weights are first copied into one of two reusable pinned host buffers, then copied to a
    fresh allocator-managed CUDA tensor on that slot's dedicated stream. The current compute stream waits for the copy event only
    immediately before it consumes the returned tensor.

    The two slots bound pinned RAM. CUDA destination lifetime is deliberately delegated to PyTorch's caching
    allocator and record_stream() instead of manually reusing device buffers across GEMMs. If neither host slot is
    reusable without a CPU wait, the caller falls back to the existing synchronous cast path instead of stalling
    Python to wait for staging capacity.
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
            )
            for _ in range(_NUM_STAGING_SLOTS)
        ]
        self._next_slot = 0
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

            # The pinned host buffer cannot be overwritten while an earlier H->D transfer is still reading it.
            # Do not CPU-synchronize here; if both host slots are busy, the normal synchronous cast is cheaper than
            # turning the staging optimization into another launch-thread stall.
            if slot.copy_pending and not slot.copy_done.query():
                continue

            slot.copy_pending = False
            self._next_slot = (slot_index + 1) % len(self._slots)
            return slot

        return None

    def _ensure_host_buffer(self, slot: _StagingSlot, tensor: torch.Tensor) -> None:
        numel = tensor.numel()
        dtype_changed = slot.buffer_dtype != tensor.dtype
        host_too_small = slot.host_buffer is None or slot.host_buffer.numel() < numel

        if dtype_changed or host_too_small:
            slot.host_buffer = torch.empty(numel, dtype=tensor.dtype, device="cpu", pin_memory=True)

        slot.buffer_dtype = tensor.dtype

    @torch.no_grad()
    def try_stage(self, tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
        if not self._is_eligible(tensor, input):
            return None
        assert tensor is not None

        slot = self._select_slot()
        if slot is None:
            self.stats.synchronous_fallbacks += 1
            return None

        try:
            self._ensure_host_buffer(slot, tensor)
        except RuntimeError:
            # Pinned-memory allocation can fail independently of the normal model path. Disable the
            # experiment for the remainder of this denoise and let CustomLinear fall back to its established cast.
            self._disabled = True
            self.stats.synchronous_fallbacks += 1
            return None

        assert slot.host_buffer is not None

        numel = tensor.numel()
        host_flat = slot.host_buffer[:numel]
        host_flat.copy_(tensor.detach().view(-1))

        try:
            # Allocate and first-use the destination on the staging stream. Allocating it on the compute stream and
            # immediately writing it from another stream can race with prior work when the caching allocator reuses
            # a block whose compute-stream use has not completed yet.
            with torch.cuda.stream(slot.stream):
                staged = torch.empty_like(tensor, device=self._device)
                staged.copy_(host_flat.view(tensor.shape), non_blocking=True)
                slot.copy_done.record()
        except RuntimeError:
            self.stats.synchronous_fallbacks += 1
            return None

        slot.copy_pending = True
        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_event(slot.copy_done)
        staged.record_stream(current_stream)

        tensor_bytes = tensor.numel() * tensor.element_size()
        self.stats.staged_tensors += 1
        self.stats.staged_bytes += tensor_bytes
        return staged

    def close(self) -> None:
        """Wait for outstanding H->D copies before releasing pinned host buffers."""
        for slot in self._slots:
            if slot.copy_pending:
                slot.copy_done.synchronize()


_ACTIVE_STAGER: ContextVar[CudaAsyncLinearWeightStager | None] = ContextVar(
    "invokeai_cuda_async_linear_weight_stager", default=None
)


@contextmanager
def cuda_async_linear_weight_staging(
    device: torch.device,
) -> Generator[CudaAsyncLinearWeightStager | None, None, None]:
    """Enable bounded async linear-weight staging for the current execution context.

    The context is a no-op off CUDA. It is currently entered only by Chroma denoise, keeping this experiment isolated
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


def maybe_stage_tensor_for_input(tensor: torch.Tensor | None, input: torch.Tensor) -> torch.Tensor | None:
    """Return an asynchronously staged tensor, or None when the caller should use the normal cast path."""
    stager = _ACTIVE_STAGER.get()
    if stager is None:
        return None
    return stager.try_stage(tensor, input)
