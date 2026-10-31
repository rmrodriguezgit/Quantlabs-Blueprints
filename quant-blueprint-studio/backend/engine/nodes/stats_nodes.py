"""Chronological preparation: at time t, predict the target at t+1."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose


@dataclass
class PreparedData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    returns: np.ndarray
    dates: list[str]
    scaler: StandardScaler | MinMaxScaler
    decomposition: list[dict[str, Any]]
    target: str
    feature_names: list[str]


def prepare(frame: pd.DataFrame, params: dict[str, Any]) -> PreparedData:
    target = str(params.get('target', 'LogReturn'))
    if target not in ('LogReturn', 'Close', 'Volatility') or target not in frame:
        raise ValueError('Target must be LogReturn, Close or Volatility; connect Features first.')
    lookback = int(params.get('lookback', 20))
    train_ratio = float(params.get('train_ratio', 0.65))
    val_ratio = float(params.get('val_ratio', 0.15))
    period = int(params.get('period', 20))
    if not 1 <= lookback <= 120 or not 2 <= period <= 120:
        raise ValueError('Lookback must be 1–120; period 2–120.')
    if train_ratio < 0.3 or val_ratio < 0.05 or train_ratio + val_ratio > 0.95:
        raise ValueError('Require train ≥ .3, validation ≥ .05 and at least .05 test.')
    requested = str(params.get('feature_columns', '')).strip()
    names = [name.strip() for name in requested.split(',') if name.strip()] if requested else list(frame.select_dtypes(include=[np.number]).columns)
    if not names or len(names) != len(set(names)):
        raise ValueError('Select at least one unique input feature.')
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f'Selected features unavailable: {", ".join(sorted(missing))}. Enable their indicators first.')
    values = frame[names].to_numpy(dtype=np.float64)
    count = len(values) - lookback
    train_end, val_end = int(count * train_ratio), int(count * (train_ratio + val_ratio))
    if min(train_end, val_end - train_end, count - val_end) < 3:
        required_samples = 1
        while min(int(required_samples * train_ratio), int(required_samples * (train_ratio + val_ratio)) - int(required_samples * train_ratio), required_samples - int(required_samples * (train_ratio + val_ratio))) < 3:
            required_samples += 1
        raw_rows = int(frame.attrs.get('raw_rows', len(frame)))
        raise ValueError(
            f'Insufficient data: {len(frame)} usable rows from {raw_rows} source rows; '
            f'lookback={lookback} leaves {max(count, 0)} sequences. '
            f'Splits train/validation/test={max(train_end, 0)}/{max(val_end-train_end, 0)}/{max(count-val_end, 0)}; '
            f'each needs at least 3. Need at least {lookback + required_samples} usable rows. '
            'Increase Market Data history (e.g. 3–5 years), reduce indicator windows/lags (e.g. 20 bars), '
            'or reduce Series Preparation lookback (1 for a one-bar Dense input).')
    if len(frame) < 2 * period:
        raise ValueError(f'Decomposition period={period} needs {2 * period} usable rows; only {len(frame)} available. Reduce the decomposition period or increase data history.')
    # Training windows end at lookback + train_end - 2. Fit nowhere else.
    scaler = MinMaxScaler() if params.get('scaler') == 'minmax' else StandardScaler()
    scaler.fit(values[:lookback + train_end - 1])
    scaled = scaler.transform(values).astype(np.float32)
    x = np.stack([scaled[i-lookback:i] for i in range(lookback, len(values))])
    y = frame[target].to_numpy(dtype=np.float32)[lookback:]
    # Decomposition is a display-only diagnostic, never an input to training.
    dec = seasonal_decompose(frame['Close'], model='additive', period=period, two_sided=False)
    decomposition = [{'date': str(date.date()), 'TR': float(tr), 'VA': float(va), 'TE': float(te)}
                     for date, tr, va, te in zip(frame.index, dec.trend, dec.seasonal, dec.resid)
                     if np.isfinite(tr) and np.isfinite(va) and np.isfinite(te)]
    return PreparedData(x[:train_end], y[:train_end], x[train_end:val_end], y[train_end:val_end],
                        x[val_end:], y[val_end:], frame['LogReturn'].to_numpy()[lookback+val_end:],
                        [str(d.date()) for d in frame.index[lookback+val_end:]], scaler, decomposition, target, names)
