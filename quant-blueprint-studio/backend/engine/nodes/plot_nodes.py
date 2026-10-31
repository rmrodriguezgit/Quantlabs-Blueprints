"""Declarative, serializable plots for frame, preparation, model and backtest pins."""
from typing import Any
import numpy as np
import pandas as pd
from .stats_nodes import PreparedData
from .dl_nodes import ModelResult


def plot(value: Any, params: dict[str, Any], node_id: str) -> dict[str, Any]:
    mode = str(params.get('plot_source', 'auto'))
    kind = str(params.get('chart_type', 'line'))
    if kind not in ('line', 'area', 'bar'):
        raise ValueError('Plot chart type must be line, area or bar.')
    if mode == 'auto':
        mode = 'columns' if isinstance(value, pd.DataFrame) else 'decomposition' if isinstance(value, PreparedData) else 'predictions' if isinstance(value, ModelResult) else 'backtest'
    rows: list[dict[str, Any]]
    defaults: list[str]
    x_key = 'date'
    if mode == 'columns' and isinstance(value, pd.DataFrame):
        numeric = list(value.select_dtypes(include=[np.number]).columns)
        requested = str(params.get('plot_columns', '')).strip()
        defaults = [name.strip() for name in requested.split(',') if name.strip()] if requested else ['Close']
        missing = set(defaults) - set(numeric)
        if missing:
            raise ValueError(f'Plot columns unavailable: {", ".join(sorted(missing))}. Enable those indicators or change the selected columns.')
        rows = [{'date': str(date.date()), **{column: float(row[column]) for column in defaults}} for date, row in value.iterrows()]
    elif mode == 'decomposition' and isinstance(value, (PreparedData, ModelResult)):
        data = value.prepared if isinstance(value, ModelResult) else value
        rows, defaults = data.decomposition, ['TR', 'VA', 'TE']
    elif mode == 'predictions' and isinstance(value, ModelResult):
        rows = [{'date': date, 'actual': float(actual), 'predicted': float(predicted)} for date, actual, predicted in zip(value.prepared.dates, value.prepared.y_test, value.predicted)]
        defaults = ['actual', 'predicted']
    elif mode == 'loss' and isinstance(value, ModelResult):
        rows, defaults, x_key = value.history, ['train', 'validation'], 'epoch'
    elif mode in ('backtest', 'signals') and isinstance(value, dict) and 'backtest' in value:
        rows, defaults = value['backtest'], ['signal'] if mode == 'signals' else ['strategy', 'market']
    else:
        raise ValueError(f'Plot source "{mode}" does not match its input. Connect Features for columns, Preparation for TR/VA/TE, Model for predictions/loss, or Backtest for equity/signals.')
    selected = str(params.get('plot_columns', '')).strip()
    series = [column.strip() for column in selected.split(',') if column.strip()] if selected else defaults
    if not rows:
        raise ValueError('No observations available for this plot. Increase data history or reduce indicator windows.')
    available = {key for key, number in rows[0].items() if key != x_key and isinstance(number, (int, float))}
    if not series or len(series) > 12 or len(series) != len(set(series)) or set(series) - available:
        raise ValueError(f'Choose 1–12 unique numeric series. Available: {", ".join(sorted(available))}.')
    return {'id': node_id, 'title': str(params.get('title', 'Custom plot')),
            'chart_type': kind, 'source': mode, 'x_key': x_key, 'series': series,
            'rows': [{x_key: row[x_key], **{column: row[column] for column in series}} for row in rows]}
