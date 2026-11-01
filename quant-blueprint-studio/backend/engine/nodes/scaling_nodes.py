"""Training-only scaler construction and serializable preprocessing diagnostics."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

Scaler = StandardScaler | MinMaxScaler


def matrix(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 2 or not data.shape[0] or not data.shape[1] or not np.isfinite(data).all():
        raise ValueError('Scaler requires a non-empty, finite 2-D training matrix.')
    return data


def fit_scaler(training: np.ndarray, names: list[str], method: str = 'standard', scope: str = 'training', title: str = 'Input features X') -> tuple[Scaler, dict[str, Any]]:
    data = matrix(training)
    if len(names) != data.shape[1] or len(names) != len(set(names)):
        raise ValueError('Scaler feature names must uniquely match the training columns.')
    if method not in ('standard', 'minmax'):
        raise ValueError('Scaler must be standard or minmax.')
    scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
    scaler.fit(data)
    transformed = scaler.transform(data)
    report = {'title': title, 'method': type(scaler).__name__, 'scope': scope, 'fit_rows': len(data),
              'ddof': 0, 'inverse_before_metrics': False, 'columns': [
                  {'name': name, 'mean': float(data[:, i].mean()), 'variance': float(data[:, i].var(ddof=0)),
                   'center': float(scaler.mean_[i]) if isinstance(scaler, StandardScaler) else float(scaler.data_min_[i]),
                   # Both diagnostics describe divisor, so z=(x-center)/scale.
                   'scale': float(scaler.scale_[i]) if isinstance(scaler, StandardScaler) else float(1 / scaler.scale_[i]),
                   'transformed_mean': float(transformed[:, i].mean()),
                   'transformed_std': float(transformed[:, i].std(ddof=0)),
                   'constant': bool(np.ptp(data[:, i]) == 0)} for i, name in enumerate(names)],
              'note': 'Fit once on the stated fitting rows; validation/test/grid only transform. '+('StandardScaler uses ddof=0. Constant or numerically near-constant columns do not necessarily have unit transformed variance.' if method=='standard' else 'MinMaxScaler uses the fitting minimum and range; it does not impose zero mean or unit variance.')}
    return scaler, report


def disabled_audit(rows: int, title: str, scope: str, reason: str) -> dict[str, Any]:
    return {'title': title, 'method': 'Disabled', 'scope': scope, 'fit_rows': 0, 'available_rows': rows,
            'columns': [], 'note': reason, 'inverse_before_metrics': False}


def affine_audit(training: np.ndarray, center: float, scale: float, title: str, scope: str = 'training', method: str = 'Train mean/std', inverse: bool = True) -> dict[str, Any]:
    data = matrix(np.asarray(training).reshape(-1, 1))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Target center and scale must be finite; scale must be positive.')
    z = (data[:, 0] - center) / scale
    return {'title': title, 'method': method, 'scope': scope, 'fit_rows': len(data), 'ddof': 0,
            'inverse_before_metrics': inverse, 'columns': [
                {'name': 'target' if inverse else 'return', 'mean': float(data.mean()), 'variance': float(data.var(ddof=0)),
                 'center': center, 'scale': scale, 'transformed_mean': float(z.mean()),
                 'transformed_std': float(z.std(ddof=0)), 'constant': bool(np.ptp(data) == 0)}],
            'note': 'Affine target transform; predictions return to original units before errors. It is separate from the X scaler.' if inverse else 'Training return-window observations only; overlapping windows weight a return each time it occurs. Minimum divisor 1e-8.'}
