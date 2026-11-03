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
    if kind not in ('line', 'area', 'bar', 'scatter', 'scatter_line', 'histogram'):
        raise ValueError('Plot chart type must be line, area, bar, scatter, scatter_line or histogram.')
    if isinstance(value, dict) and value.get('study'):
        index = int(params.get('study_chart', 0))
        charts = value['charts']
        if not 0 <= index < len(charts):
            raise ValueError(f'Experiment chart index must be 0–{len(charts)-1}.')
        chosen = charts[index]
        mode = 'study'
        kind = str(params.get('chart_type', chosen['chart_type']))
    if mode == 'auto':
        mode = 'columns' if isinstance(value, pd.DataFrame) else 'decomposition' if isinstance(value, PreparedData) else 'predictions' if isinstance(value, ModelResult) else 'asset_prices' if isinstance(value, PortfolioData) else 'equity' if isinstance(value, (PortfolioResult, RiskAnalysis)) else 'backtest'
    rows: list[dict[str, Any]]
    defaults: list[str]
    x_key = 'date'
    requested_x=str(params.get('x_column','')).strip()
    # Infer only the notebook coordinate; preserve explicit selections.
    if not requested_x and kind in ('scatter','scatter_line'):
        if isinstance(value,pd.DataFrame) and 'Coordinate' in value.columns:
            requested_x='Coordinate'
        elif mode=='study' and value.get('topic')=='curve_fit' and chosen['rows'] and 'coordinate' in chosen['rows'][0]:
            requested_x='coordinate'
    if mode == 'study' and isinstance(value, dict) and value.get('study'):
        rows, defaults, x_key = chosen['rows'], chosen['series'], chosen['x_key']
    elif mode == 'asset_prices' and isinstance(value, (PortfolioData, PortfolioResult)):
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
        if requested_x and requested_x!='date':
            if requested_x not in numeric: raise ValueError('Frame X column must be date or an available numeric column.')
            rows=[{'date':str(date.date()),requested_x:float(row[requested_x]),**{column:float(row[column]) for column in defaults}} for date,row in value.iterrows()]
        else:
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
    if requested_x:
        if requested_x not in rows[0]:raise ValueError(f'X column unavailable: {requested_x}. Available: {", ".join(rows[0])}.')
        x_key=requested_x
    skip=int(params.get('skip_rows',0))
    if skip<0 or skip>=len(rows):raise ValueError(f'Skip rows must be 0–{len(rows)-1}. Reduce skip_rows or increase epochs.')
    rows=rows[skip:]
    available = {key for key, number in rows[0].items() if key != x_key and isinstance(number, (int, float))}
    if not series or len(series) > 12 or len(series) != len(set(series)) or set(series) - available:
        raise ValueError(f'Choose 1–12 unique numeric series. Available: {", ".join(sorted(available))}.')
    if kind=='histogram':
        bins=int(params.get('hist_bins',20))
        if len(series)!=1 or not 5<=bins<=100:
            raise ValueError('Histogram requires exactly one series and 5–100 bins.')
        values=np.asarray([row[series[0]] for row in rows],dtype=float)
        if not np.isfinite(values).all():raise ValueError('Histogram observations must be finite; clean missing rows upstream.')
        counts,bounds=np.histogram(values,bins=bins,density=bool(params.get('hist_density',False)))
        x_key='bin';series=['density' if params.get('hist_density',False) else 'count']
        rows=[{'bin':float((bounds[i]+bounds[i+1])/2),series[0]:float(count)} for i,count in enumerate(counts)]
        kind='bar'
    y_scale=str(params.get('y_scale','linear'))
    if y_scale not in ('linear','log'):raise ValueError('Y scale must be linear or log.')
    if y_scale=='log' and any(not np.isfinite(row[column]) or row[column]<=0 for row in rows for column in series):
        raise ValueError('Logarithmic Y requires strictly positive finite values. Select positive loss series or use linear scale.')
    scatter_columns=[s.strip() for s in str(params.get('scatter_columns','actual')).split(',') if s.strip()]
    if kind=='scatter_line':
        if set(scatter_columns)-set(series) or not scatter_columns:raise ValueError('Scatter columns must be selected plot series; use actual for notebook observations.')
    if kind in ('scatter','scatter_line') and any(not isinstance(row[x_key],(int,float)) or not np.isfinite(row[x_key]) for row in rows):
        raise ValueError('Scatter plots need a numeric X column, e.g. coordinate. Set x_column accordingly.')
    line_rows=None
    if kind=='scatter_line' and isinstance(value,dict) and value.get('topic')=='curve_fit' and int(params.get('study_chart',0))==0 and x_key=='coordinate':
        smooth=value['charts'][1]['rows']
        line_series=[column for column in series if column not in scatter_columns]
        if line_series and all(column in smooth[0] for column in line_series):
            line_rows=[{x_key:row[x_key],**{column:row[column] for column in line_series}} for row in smooth]
    if y_scale=='log' and line_rows and any(not np.isfinite(row[column]) or row[column]<=0 for row in line_rows for column in row if column!=x_key):
        raise ValueError('Logarithmic Y requires positive finite values in the smooth curve too.')
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
    result={'id': node_id, 'title': title, 'origin': origin, 'colors': colors,
            'labels': {key: custom_labels[i] if custom_labels else key for i,key in enumerate(series)},
            'chart_type': kind, 'y_scale':y_scale,'scatter_columns':scatter_columns if kind=='scatter_line' else [],'source': mode, 'x_key': x_key, 'series': series,
            'rows': [{x_key: row[x_key], **{column: row[column] for column in series}} for row in rows]}

    if line_rows is not None: result['line_rows']=line_rows
    return result
