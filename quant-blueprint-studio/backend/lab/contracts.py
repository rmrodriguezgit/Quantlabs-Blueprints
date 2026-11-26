"""Uniform dataset contract shared by every data source of the ML lab.

Every connector (synthetic, yfinance, polymarket, ...) returns a :class:`Dataset`:
a frame indexed by a tz-naive UTC ``DatetimeIndex`` (sorted, unique) plus a
:class:`DatasetMeta` that records where the data came from and how it was built.
``validate_frame`` is the single gate: downstream code may assume its guarantees.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1
SOURCES = ('synthetic', 'yfinance', 'polymarket')


class ContractError(ValueError):
    """The frame does not satisfy the dataset contract."""


@dataclass(frozen=True)
class DatasetMeta:
    source: str
    symbol: str
    freq: str
    params: dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ''
    rows: int = 0
    start: str = ''
    end: str = ''
    columns: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of what was requested (not when), for artifact traceability."""
        key = json.dumps({'source': self.source, 'symbol': self.symbol, 'freq': self.freq, 'params': self.params},
                         sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:12]


@dataclass
class Dataset:
    frame: pd.DataFrame
    meta: DatasetMeta


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Coerce to tz-naive UTC, sorted, de-duplicated (keep last observation)."""
    out = frame.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    if idx.tz is not None:
        idx = idx.tz_convert('UTC').tz_localize(None)
    out.index = idx
    out.index.name = 'timestamp'
    out = out[~out.index.duplicated(keep='last')].sort_index()
    return out


def validate_frame(frame: pd.DataFrame, required: tuple[str, ...] = ('Close',)) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ContractError('Dataset vacío.')
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ContractError('El índice debe ser DatetimeIndex.')
    if frame.index.tz is not None:
        raise ContractError('El índice debe ser UTC sin zona (tz-naive).')
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ContractError('El índice debe ser estrictamente creciente y sin duplicados.')
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractError(f'Faltan columnas requeridas: {missing}.')
    for c in required:
        values = pd.to_numeric(frame[c], errors='coerce').to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ContractError(f'La columna {c} tiene valores no finitos.')


def make_dataset(frame: pd.DataFrame, *, source: str, symbol: str, freq: str, params: dict[str, Any],
                 required: tuple[str, ...] = ('Close',)) -> Dataset:
    if source not in SOURCES:
        raise ContractError(f'Fuente desconocida {source!r}; válidas: {SOURCES}.')
    frame = normalize_index(frame)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ContractError(f'Faltan columnas requeridas: {missing}.')
    frame = frame.dropna(subset=list(required))
    validate_frame(frame, required)
    frame.attrs = {}  # never leak connector-private state into the contract
    meta = DatasetMeta(source=source, symbol=symbol, freq=freq, params=dict(params),
                       fetched_at=pd.Timestamp.now(tz='UTC').isoformat(), rows=len(frame),
                       start=str(frame.index[0]), end=str(frame.index[-1]), columns=tuple(frame.columns))
    return Dataset(frame, meta)
