"""Reusable DataFrame operations. Originals remain available beside derived columns."""
from typing import Any
import numpy as np
import pandas as pd
from .scaling_nodes import fit_scaler, affine_audit


def columns(frame: pd.DataFrame, params: dict[str, Any]) -> list[str]:
    names = [s.strip() for s in str(params.get('columns', 'Close')).split(',') if s.strip()]
    if not names or len(names) != len(set(names)) or set(names) - set(frame.columns):
        raise ValueError('Choose unique available columns: ' + ', '.join(map(str, frame.columns)))
    return names


def resample_frame(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_unique:
        raise ValueError('Resampling requires unique dated observations.')
    rule = str(params.get('frequency', 'ME'))
    operation = str(params.get('aggregation', 'last'))
    if rule not in ('W-FRI', 'ME', 'QE', 'YE') or operation not in ('last', 'mean', 'sum'):
        raise ValueError('Choose W-FRI/ME/QE/YE and last/mean/sum.')
    selected = frame[columns(frame, params)].sort_index()
    result = getattr(selected.resample(rule), operation)().dropna()
    if params.get('exclude_incomplete', True) and len(result):
        now = pd.Timestamp.now(tz=result.index.tz)
        if result.index[-1].normalize() >= now.normalize():
            result = result.iloc[:-1]
    if result.empty:
        raise ValueError('Resampling produced no complete observations.')
    result.attrs = {**frame.attrs, "raw_rows": frame.attrs.get("raw_rows", len(frame))}
    return result


def coordinate(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    start, stop = float(params.get('start', -2)), float(params.get('stop', 2))
    name = str(params.get('output_column', 'Coordinate')).strip()
    if not np.isfinite([start, stop]).all() or start >= stop or len(frame) < 2:
        raise ValueError('Coordinate requires at least two rows and finite start < stop.')
    if not name or name in frame.columns:
        raise ValueError('Choose a new nonempty coordinate column name.')
    result = frame.copy()
    result[name] = np.linspace(start, stop, len(frame))
    return result


def transform(frame: pd.DataFrame, params: dict[str, Any], kind: str) -> pd.DataFrame:
    if kind == 'resample':
        return resample_frame(frame, params)
    if kind == 'coordinate':
        return coordinate(frame, params)
    names = columns(frame, params)
    if kind == 'select_columns':
        return frame[names].copy()
    values = frame[names].to_numpy(dtype=np.float64)
    fraction = float(params.get('fit_fraction', .65))
    if not np.isfinite(values).all() or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Transform requires finite values and fit fraction in (0, 1].')
    count = int(len(frame) * fraction)
    if count < 2 or not isinstance(frame.index, pd.DatetimeIndex) or not frame.index.is_monotonic_increasing:
        raise ValueError('At least two chronological dated fitting rows are required.')
    prefix = str(params.get('prefix', 'Centered_' if kind == 'center' else 'Scaled_')).strip()
    outputs = [prefix + name for name in names]
    if not prefix or set(outputs) & set(frame.columns):
        raise ValueError('Choose a prefix that creates new column names; originals are preserved.')
    scope = 'all fitting observations (descriptive)' if fraction == 1 else 'leading chronological fitting rows'
    if kind == 'center':
        mean = values[:count].mean(axis=0)
        transformed = values - mean
        audits = [affine_audit(values[:count,i], float(mean[i]), 1., 'Column centering', scope, method='Center only', inverse=False) for i in range(len(names))]
        audit = {**audits[0], 'columns': [{**item['columns'][0], 'name': name} for item,name in zip(audits,names)], 'note':'Subtract each fitting-column mean; no division by standard deviation. Originals are preserved.'}
    else:
        scaler, audit = fit_scaler(values[:count], names, str(params.get('method', 'standard')), scope, 'Column scaling')
        transformed = scaler.transform(values)
    result = frame.copy()
    for i, name in enumerate(outputs):
        result[name] = transformed[:, i]
    audit.update({'fit_start': str(frame.index[0].date()), 'fit_end': str(frame.index[count-1].date())})
    result.attrs['transform_audits'] = [audit]
    result.attrs['fitted_columns'] = [*frame.attrs.get('fitted_columns', []), {'columns': outputs, 'fit_end': str(frame.index[count-1])}]
    return result


def check_fit_boundary(frame: pd.DataFrame, names: list[str], last_training_date: pd.Timestamp) -> None:
    for fit in frame.attrs.get('fitted_columns', []):
        if set(names) & set(fit['columns']) and pd.Timestamp(fit['fit_end']) > last_training_date:
            raise ValueError('Upstream centering/scaling used held-out rows. Reduce its fit_fraction or use Series Preparation scaling fitted on TR.')
