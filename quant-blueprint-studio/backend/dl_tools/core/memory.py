"""Memory release and accounting: move weights off the accelerator, drop refs, flush caches, collect garbage."""
from __future__ import annotations

import gc
import weakref
from typing import Any

import torch


def device_bytes(device: str) -> int:
    """Bytes currently allocated by torch on a device (0 for cpu, which torch does not track)."""
    if device == 'cuda' and torch.cuda.is_available():
        return int(torch.cuda.memory_allocated())
    if device == 'mps' and torch.backends.mps.is_available():
        return int(torch.mps.current_allocated_memory())
    return 0


def flush(device: str) -> None:
    gc.collect()
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == 'mps' and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def release(model: Any, device: str) -> None:
    """Unload a model. Parameters are moved to cpu first so the accelerator frees them immediately."""
    if isinstance(model, torch.nn.Module):
        model.to('cpu')
        for p in model.parameters():
            p.grad = None
    del model
    flush(device)


def tracker(obj: Any) -> 'weakref.ref[Any]':
    """Weak reference used by tests to prove a model object was actually freed after unload()."""
    return weakref.ref(obj)
