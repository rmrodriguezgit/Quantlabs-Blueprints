"""Declarative, serializable plots for frame, preparation, model and backtest pins."""
from typing import Any
import numpy as np
import pandas as pd
from .stats_nodes import PreparedData
from .dl_nodes import ModelResult
from .metrics_nodes import RiskAnalysis
from .portfolio_nodes import PortfolioData, PortfolioResult

PALETTE = {'Mint':'#6cc9ac', 'Violet':'#9c9af8', 'Orange':'#f2ad65', 'Blue':'#79b4ef', 'Pink':'#ec9dd6', 'Yellow':'#d5d77e', 'Coral':'#f07777', 'Cyan':'#56d9dd'}


def plot(value: Any, params: dict[str, Any], node_id: str) -> dict[str, Any]:
    mode = str(params.get('plot_source', 'auto'))
    kind = str(params.get('chart_type', 'line'))
    if kind not in ('line', 'area', 'bar'):
        raise ValueError('Plot chart type must be line, area or bar.')
    if mode == 'auto':
        mode = 'columns' if isinstance(value, pd.DataFrame) else 'decomposition' if isinstance(value, PreparedData) else 'predictions' if isinstance(value, ModelResult) else 'asset_prices' if isinstance(value, PortfolioData) else 'equity' if isinstance(value, (PortfolioResult, RiskAnalysis)) else 'backtest'
    rows: list[dict[str, Any]]
    defaults: list[str]
    x_key = 'date'
    if mode == 'asset_prices' and isinstance(value, (PortfolioData, PortfolioResult)):
        data = value.data if isinstance(value, PortfolioResult) else value
        rows = [{'date':str(date.date()), **{str(k):float(v) for k,v in row.items()}} for date,row in data.prices.iterrows()]
        defaults = list(data.prices.columns)[:12]
    elif mode == 'weights' and isinstance(value, PortfolioResult):
        x_key = 'asset'
        defaults = list(value.report['weights'])
        rows = [{'asset':name, **{strategy:allocation[name] for strategy,allocation in value.report['weights'].items()}} for name in value.data.prices.columns]
    elif mode in ('equity','cumulative_log','drawdown') and isinstance(value, (PortfolioResult, RiskAnalysis)):
        daily = value.returns
        equity = (1 + daily).cumprod()
        if mode == 'equity':
            series_frame = equity - 1
        elif mode == 'cumulative_log':
            series_frame = np.log1p(daily).cumsum()
        else:
            series_frame = equity / equity.cummax().clip(lower=1) - 1
        rows = [{'date':str(date.date()), **{str(k):float(v) for k,v in row.items()}} for date,row in series_frame.iterrows()]
        defaults = list(daily.columns)[:12]
    elif mode == 'columns'  and isinstance(value, pd.DataFrame):
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
        raise ValueError(f'Plot source "{mode}" does not match its input. Connect Features for columns, Preparation for TR/VA/TE, Model for predictions/loss, Backtest for equity/signals, or portfolio pins for asset_prices/equity/cumulative_log/drawdown/weights.')
    selected = str(params.get('plot_columns', '')).strip()
    series = [column.strip() for column in selected.split(',') if column.strip()] if selected else defaults
    if not rows:
        raise ValueError('No observations available for this plot. Increase data history or reduce indicator windows.')
    available = {key for key, number in rows[0].items() if key != x_key and isinstance(number, (int, float))}
    if not series or len(series) > 12 or len(series) != len(set(series)) or set(series) - available:
        raise ValueError(f'Choose 1–12 unique numeric series. Available: {", ".join(sorted(available))}.')
    palette_names = list(PALETTE)
    color = str(params.get('color','Auto'))
    if color != 'Auto' and color not in PALETTE:
        raise ValueError(f'Unknown plot color. Choose Auto or {", ".join(palette_names)}.')
    offset = int(params.get('_plot_index',0)) % len(palette_names) if color == 'Auto' else palette_names.index(color)
    assigned = [value.strip() for value in str(params.get('series_colors','')).split(',') if value.strip()]
    if assigned and (len(assigned) != len(series) or any(value not in PALETTE for value in assigned)):
        raise ValueError('Series colors must match the series count and use predefined palette names.')
    colors = {key: PALETTE[assigned[i] if assigned else palette_names[(offset+i)%len(palette_names)]] for i,key in enumerate(series)}
    custom_labels = [value.strip() for value in str(params.get('series_labels','')).split(',') if value.strip()]
    if custom_labels and len(custom_labels) != len(series):
        raise ValueError('Legend names must match the selected series count.')
    origin = str(params.get('_origin','Input'))
    title = str(params.get('title','')).strip()
    if not title or title == 'Custom plot':
        title = f'{origin} · {", ".join(series)}'
    return {'id': node_id, 'title': title, 'origin': origin, 'colors': colors,
            'labels': {key: custom_labels[i] if custom_labels else key for i,key in enumerate(series)},
            'chart_type': kind, 'source': mode, 'x_key': x_key, 'series': series,
            'rows': [{x_key: row[x_key], **{column: row[column] for column in series}} for row in rows]}
