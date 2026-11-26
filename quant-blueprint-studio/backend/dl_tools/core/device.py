"""Device selection with a controlled CPU fallback (no GPU, unsupported backend or forced cpu)."""
from __future__ import annotations

import logging
from typing import Literal

import torch

DeviceName = Literal['cpu', 'cuda', 'mps']
log = logging.getLogger('dl_tools')


def available() -> list[DeviceName]:
    found: list[DeviceName] = ['cpu']
    if torch.cuda.is_available():
        found.insert(0, 'cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        found.insert(0, 'mps')
    return found


def resolve_device(requested: str, supported: tuple[str, ...]) -> DeviceName:
    """'auto' picks the best supported accelerator; an unavailable request falls back to cpu with a warning."""
    have = [d for d in available() if d in supported] or ['cpu']
    if requested == 'auto':
        return have[0]
    if requested in have:
        return requested  # type: ignore[return-value]
    log.warning('Device %s unavailable or unsupported; falling back to cpu.', requested)
    return 'cpu'
