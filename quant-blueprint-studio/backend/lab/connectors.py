"""Isolated data connectors behind one interface: ``load(source, params) -> Dataset``.

Network connectors reuse the validated app nodes (``yfinance_nodes`` / ``polymarket_nodes``)
so the lab and the blueprint editor read data the same way, and add bounded retries with
exponential backoff. The synthetic generator is fully determined by its seed.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from .contracts import Dataset, make_dataset


class ConnectorError(RuntimeError):
    """A source could not deliver data (network, rate limit, empty answer)."""


def _retry(fn: Callable[[], Any], *, retries: int, backoff: float, what: str) -> Any:
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except ValueError as exc:          # validation errors are deterministic: do not retry
            raise ConnectorError(f'{what}: {exc}') from exc
        except Exception as exc:           # network / rate-limit / transient
            last = exc
            if attempt < retries:
                time.sleep(backoff * 2 ** attempt)
    raise ConnectorError(f'{what}: falló tras {retries + 1} intentos ({last}).') from last


def synthetic(params: dict[str, Any]) -> Dataset:
    """Regime-switching GBM OHLCV. Same params (incl. seed) -> identical frame."""
    n = int(params.get('n', 1000))
    seed = int(params.get('seed', 42))
    freq = str(params.get('freq', 'B'))
    if not 50 <= n <= 1_000_000:
        raise ConnectorError('synthetic: n debe estar entre 50 y 1e6.')
    rng = np.random.default_rng(seed)
    vol = np.where(rng.random(n).cumsum() % 200 < 140, 0.010, 0.022)   # calm / turbulent regimes
    drift = float(params.get('drift', 0.0003))
    close = float(params.get('start_price', 100.0)) * np.exp(np.cumsum(rng.normal(drift, vol)))
    opening = close * np.exp(rng.normal(0, 0.003, n))
    spread = np.abs(rng.normal(0, 0.004, n))
    index = pd.date_range(end=str(params.get('end', '2025-12-31')), periods=n, freq=freq)
    frame = pd.DataFrame({'Open': opening, 'High': np.maximum(opening, close) * (1 + spread),
                          'Low': np.minimum(opening, close) * (1 - spread), 'Close': close,
                          'Volume': rng.integers(100_000, 5_000_000, n).astype(float)}, index=index)
    return make_dataset(frame, source='synthetic', symbol=f'SYN-{seed}', freq=freq,
                        params={'n': n, 'seed': seed, 'freq': freq, 'drift': drift})


def yfinance(params: dict[str, Any]) -> Dataset:
    from backend.engine.nodes.yfinance_nodes import yfinance_data
    p = {'ticker': 'SPY', 'interval': '1d', 'period': '2y', 'timezone': 'UTC', 'timeout': 15, **params}
    frame, _ = _retry(lambda: yfinance_data(p), retries=int(params.get('retries', 2)),
                      backoff=float(params.get('backoff', 1.0)), what=f'yfinance {p["ticker"]}')
    return make_dataset(frame, source='yfinance', symbol=str(p['ticker']).upper(), freq=str(p['interval']),
                        params={k: v for k, v in p.items() if k not in ('retries', 'backoff')})


def polymarket(params: dict[str, Any]) -> Dataset:
    from backend.engine.nodes.polymarket_nodes import polymarket_source
    if not params.get('market'):
        raise ConnectorError('polymarket: falta "market" (slug, condition id o URL).')
    p = {'history_interval': '1d', 'fidelity': 60, 'timeout': 8, **params}
    frame, _ = _retry(lambda: polymarket_source(p), retries=int(params.get('retries', 2)),
                      backoff=float(params.get('backoff', 1.0)), what=f'polymarket {p["market"]}')
    # Book_* columns are a live snapshot written only on the last row; they are not a time series.
    frame = frame[[c for c in frame.columns if not c.startswith('Book_')]]
    return make_dataset(frame, source='polymarket', symbol=str(p['market']), freq=f'{p["fidelity"]}min',
                        params={k: v for k, v in p.items() if k not in ('retries', 'backoff')})


CONNECTORS: dict[str, Callable[[dict[str, Any]], Dataset]] = {
    'synthetic': synthetic, 'yfinance': yfinance, 'polymarket': polymarket,
}


def load(source: str, params: dict[str, Any] | None = None) -> Dataset:
    if source not in CONNECTORS:
        raise ConnectorError(f'Fuente desconocida {source!r}; válidas: {sorted(CONNECTORS)}.')
    return CONNECTORS[source](dict(params or {}))
