"""Chronological, leakage-free splits.

Targets look ``horizon`` bars ahead, so the last ``horizon`` rows of each block would
peek into the next block. ``embargo`` (>= horizon) bars are dropped at every boundary.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    def sizes(self) -> dict[str, int]:
        return {'train': len(self.train), 'validation': len(self.validation), 'test': len(self.test)}


def chronological_split(frame: pd.DataFrame, train: float = 0.6, validation: float = 0.2,
                        embargo: int = 1) -> Split:
    if not 0 < train < 1 or not 0 <= validation < 1 or train + validation >= 1:
        raise ValueError('Proporciones inválidas: 0<train<1, train+validation<1.')
    if embargo < 0:
        raise ValueError('embargo >= 0.')
    if not frame.index.is_monotonic_increasing:
        raise ValueError('El frame debe estar ordenado cronológicamente.')
    n = len(frame)
    a, b = int(n * train), int(n * (train + validation))
    parts = (frame.iloc[:max(a - embargo, 0)], frame.iloc[a:max(b - embargo, a)], frame.iloc[b:])
    if min(len(p) for p in parts) == 0:
        raise ValueError(f'Datos insuficientes ({n} filas) para el split con embargo {embargo}.')
    return Split(*parts)


def walk_forward(frame: pd.DataFrame, folds: int = 5, min_train: float = 0.4, embargo: int = 1):
    """Expanding-window folds: yields (train, test) with train strictly before test."""
    n = len(frame)
    start = int(n * min_train)
    step = (n - start) // folds
    if step <= embargo:
        raise ValueError('Datos insuficientes para walk-forward.')
    for k in range(folds):
        cut = start + k * step
        yield frame.iloc[:cut - embargo], frame.iloc[cut:cut + step]
