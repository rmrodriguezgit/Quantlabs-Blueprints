"""Typed data pins and Kahn DAG execution. No graph-supplied code is evaluated."""
from collections import deque
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
from .nodes.scaling_nodes import affine_audit
from .nodes.curve_nodes import monthly_curve, curve_fit
from .nodes.study_nodes import study
from .nodes.data_nodes import source, features, merge_features
from .nodes.stats_nodes import PreparedData, prepare
from .nodes.dl_nodes import ModelResult, NetworkSpec, train
from .nodes.backtest_nodes import backtest
from .nodes.plot_nodes import plot
from .nodes.metrics_nodes import RiskAnalysis, analyze, frame_returns
from .nodes.portfolio_nodes import PortfolioData, PortfolioResult, portfolio_data, build_portfolio

PINS = {'curve_data': ('frame','frame'), 'curve_fit': ('frame','tensor'), 'study': ('frame', 'tensor'), 'source': (None, 'frame'), 'features': ('frame', 'frame'), 'prepare': ('frame', 'tensor'),
        'model': ('tensor', 'tensor'), 'backtest': ('tensor', None), 'config': (None, 'config'),
        'network_input': (None, 'network'), 'layer': ('network', 'network'),
        'network_output': ('network', 'network'), 'merge': ('frame', 'frame'), 'plot': (None, None), 'portfolio_data': (None, 'portfolio'), 'portfolio': ('portfolio', 'portfolio'), 'risk': (None, 'portfolio')}
INPUTS = {kind: ({pin: pin} if pin else {}) for kind, (pin, _) in PINS.items()}
for kind in ('features', 'prepare', 'model', 'backtest', 'curve_fit'):
    INPUTS[kind]['config'] = 'config'
INPUTS['study'].update({'asset_b':'frame','asset_c':'frame'})
INPUTS['model']['network'] = 'network'
INPUTS['curve_fit']['network'] = 'network'
INPUTS['merge']['frame_b'] = 'frame'
INPUTS['plot'] = {'frame': 'frame', 'tensor': 'tensor', 'portfolio': 'portfolio'}
INPUTS['risk'] = {'frame': 'frame', 'tensor': 'tensor', 'portfolio': 'portfolio', 'benchmark': 'frame'}
PINS['backtest'] = ('tensor', 'tensor')


def serialize(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return {'rows': len(value), 'columns': list(value.columns)}
    if isinstance(value, (np.ndarray, torch.Tensor)):
        return serialize(value.tolist())
    if isinstance(value, np.generic):
        return serialize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize(v) for v in value]
    return value


class DAGRunner:
    def __init__(self, graph: dict[str, Any]) -> None:
        raw_nodes = graph.get('nodes', [])
        self.nodes = {node['id']: node for node in raw_nodes}
        self.edges = graph.get('edges', [])
        if not self.nodes or len(self.nodes) != len(raw_nodes) or len(self.nodes) > 50:
            raise ValueError('Graph requires 1–50 uniquely identified nodes.')
        self.outputs: dict[str, Any] = {}
        self.incoming: dict[str, list[dict[str, Any]]] = {key: [] for key in self.nodes}
        for node in self.nodes.values():
            if node['data']['kind'] not in PINS:
                raise ValueError('Unknown node kind.')
        for edge in self.edges:
            if edge['source'] not in self.nodes or edge['target'] not in self.nodes:
                raise ValueError('Edge refers to a missing node.')
            src, dst = self.nodes[edge['source']]['data']['kind'], self.nodes[edge['target']]['data']['kind']
            out_pin, in_pin = edge.get('sourceHandle'), edge.get('targetHandle')
            if out_pin is None or out_pin != PINS[src][1] or INPUTS[dst].get(in_pin) != out_pin:
                raise ValueError('Incompatible or missing pin handles.')
            if any(e.get('targetHandle') == in_pin for e in self.incoming[edge['target']]):
                raise ValueError('Only one connection per input pin is allowed.')
            self.incoming[edge['target']].append(edge)

    def topological_sort(self) -> list[str]:
        degrees = {key: len(edges) for key, edges in self.incoming.items()}
        outgoing: dict[str, list[str]] = {key: [] for key in self.nodes}
        for edge in self.edges:
            outgoing[edge['source']].append(edge['target'])
        queue = deque(key for key, count in degrees.items() if count == 0)
        order: list[str] = []
        while queue:
            key = queue.popleft()
            order.append(key)
            for target in outgoing[key]:
                degrees[target] -= 1
                if degrees[target] == 0:
                    queue.append(target)
        if len(order) != len(self.nodes):
            raise ValueError('Cycle detected. Blueprints must be directed acyclic graphs.')
        return order

    def execute(self, progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        report: dict[str, Any] = {'decomposition': [], 'predictions': [], 'backtest': [], 'metrics': {}, 'history': [], 'nodes': [], 'plots': [], 'portfolios': [], 'risk_reports': [], 'studies': [], 'scaling_audits': []}
        order = self.topological_sort()
        for key in order:
            node = self.nodes[key]
            kind, params = node['data']['kind'], dict(node['data'].get('params', {}))
            upstream: Any = None
            inputs: dict[str, Any] = {}
            for edge in self.incoming[key]:
                value = self.outputs[edge['source']]
                inputs[edge['targetHandle']] = value
                if edge['targetHandle'] == 'config':
                    params.update(value)
                elif edge['targetHandle'] == PINS[kind][0]:
                    upstream = value
            if progress:
                progress({'node': key, 'label': node['data'].get('label') or kind, 'status': 'running'})
            try:
                if kind == 'source':
                    output: Any = source(params)
                elif kind == 'portfolio_data':
                    output = portfolio_data(params, (lambda symbol: progress({'node': key, 'status': 'running', 'detail': f'Loading {symbol}'})) if progress else None)
                elif kind == 'portfolio' and isinstance(upstream, PortfolioData):
                    output = build_portfolio(upstream, params)
                    report['portfolios'].append({'id': key, **output.report})
                elif kind == 'risk':
                    primary = [value for pin, value in inputs.items() if pin != 'benchmark']
                    if len(primary) != 1:
                        raise ValueError('Risk Metrics needs exactly one primary input and an optional benchmark frame.')
                    value = primary[0]
                    benchmark = None
                    scope = 'historical sample'
                    if isinstance(value, PortfolioResult):
                        assets = value.data.prices.pct_change(fill_method=None).reindex(value.returns.index)
                        daily = pd.concat([assets, value.returns], axis=1)
                        benchmark, scope = value.benchmark, 'holdout / out of sample'
                    elif isinstance(value, PortfolioData):
                        daily = value.prices.pct_change(fill_method=None).dropna()
                        benchmark = value.benchmark.pct_change(fill_method=None).reindex(daily.index)
                    elif isinstance(value, RiskAnalysis):
                        daily, benchmark, scope = value.returns, value.benchmark, value.report['scope']
                    elif isinstance(value, pd.DataFrame):
                        daily = frame_returns(value, params).to_frame(params.get('series_name', 'Asset'))
                    elif isinstance(value, ModelResult):
                        daily = pd.DataFrame({'Underlying asset': np.expm1(value.prepared.returns)}, index=pd.to_datetime(value.prepared.dates))
                        scope = 'underlying asset on model test dates'
                    elif isinstance(value, dict) and 'backtest' in value:
                        daily = pd.DataFrame(value['backtest']).set_index('date')
                        daily.index = pd.to_datetime(daily.index)
                        benchmark = daily['market_return'].rename('Market')
                        daily = daily[['strategy_return','market_return']].rename(columns={'strategy_return':'Strategy','market_return':'Market'})
                        scope = 'strategy holdout / out of sample'
                    else:
                        raise ValueError('Risk Metrics input is not a return series, model, backtest or portfolio.')
                    if 'benchmark' in inputs:
                        benchmark = frame_returns(inputs['benchmark'], {'return_column':'Close','return_type':'price'}).rename(str(params.get('benchmark_name','Benchmark')))
                    output = analyze(daily, benchmark, params, str(params.get('title', 'Risk metrics')), scope)
                    report['risk_reports'].append({'id': key, **output.report})
                elif kind == 'plot':
                    if len(inputs) != 1:
                        raise ValueError('Plot requires exactly one frame or tensor input.')
                    ancestors: list[str] = []
                    visited: set[str] = set()
                    def origins(node_id: str) -> None:
                        if node_id in visited:
                            return
                        visited.add(node_id)
                        item = self.nodes[node_id]['data']
                        if item['kind'] in ('source','portfolio_data'):
                            title = str(item.get('params',{}).get('ticker' if item['kind']=='source' else 'tickers', item.get('label',node_id)))
                            ancestors.append(title)
                        for incoming in self.incoming[node_id]:
                            origins(incoming['source'])
                    origins(key)
                    params['_origin'] = ' / '.join(dict.fromkeys(ancestors)) or self.nodes[next(iter(self.incoming[key]))['source']]['data'].get('label','Input')
                    params['_plot_index'] = len(report['plots'])
                    output = plot(next(iter(inputs.values())), params, key)
                    report['plots'].append(output)
                elif kind == 'config':
                    output = params
                elif kind == 'network_input':
                    count, mode = int(params.get('input_features', 16)), str(params.get('mode', 'last'))
                    if not 1 <= count <= 512 or mode not in ('last', 'flatten'):
                        raise ValueError('Network input requires 1–512 features and last/flatten mode.')
                    output = NetworkSpec(count, mode, [])
                elif kind in ('layer', 'network_output') and isinstance(upstream, NetworkSpec):
                    if upstream.complete:
                        raise ValueError('No layers may follow Network Output; connect it to a trainer.')
                    if kind == 'network_output':
                        if int(params.get('output_units', 1)) != 1:
                            raise ValueError('Current forecasting target requires exactly one output.')
                        output = NetworkSpec(upstream.input_features, upstream.mode, list(upstream.layers), True)
                    else:
                        output = NetworkSpec(upstream.input_features, upstream.mode, [*upstream.layers, params])
                elif kind == 'merge' and isinstance(upstream, pd.DataFrame) and isinstance(inputs.get('frame_b'), pd.DataFrame):
                    output = merge_features(upstream, inputs['frame_b'])
                elif kind == 'curve_data' and isinstance(upstream, pd.DataFrame):
                    output = monthly_curve(upstream, params)
                elif kind == 'curve_fit' and isinstance(upstream, pd.DataFrame):
                    if 'network' in inputs: params['_network']=inputs['network']
                    output = curve_fit(upstream, params, (lambda event: progress({'node':key,'status':'epoch',**event})) if progress else None)
                    report['studies'].append({'id':key,'title':node['data'].get('label','Curve fit'),**output})
                elif kind == 'study' and isinstance(upstream, pd.DataFrame):
                    output = study(upstream, params, (lambda event: progress({'node': key, 'status': 'epoch', **event})) if progress else None, [inputs[k] for k in ('asset_b','asset_c') if k in inputs])
                    report['studies'].append({'id': key, 'title': node['data'].get('label', 'Experiment'), **output})
                elif kind == 'features'  and isinstance(upstream, pd.DataFrame):
                    output = features(upstream, params)
                elif kind == 'prepare' and isinstance(upstream, pd.DataFrame):
                    output = prepare(upstream, params)
                    report['decomposition'] = output.decomposition
                    report['features'] = output.feature_names
                    report['input_shape'] = list(output.x_train.shape[1:])
                elif kind == 'model' and isinstance(upstream, PreparedData):
                    if 'network' in inputs:
                        params['_network'] = inputs['network']
                        params['architecture'] = 'Custom'
                    output = train(upstream, params, (lambda event: progress({'node': key, 'status': 'epoch', **event})) if progress else None)
                    report['metrics'] = output.metrics
                    report['history'] = output.history
                    report['predictions'] = [{'date': d, 'actual': float(y), 'predicted': float(p)} for d, y, p in zip(upstream.dates, upstream.y_test, output.predicted)]
                    report['scaling_audits'].append({'id':key,'node_title':node['data'].get('label',kind),**affine_audit(upstream.y_train,float(upstream.y_train.mean()),max(float(upstream.y_train.std()),1e-8),'Regression target y')})
                    report['target'] = upstream.target
                    if output.latent is not None:
                        report['autoencoder'] = {'latent': output.latent, 'reconstructed': output.reconstructed, 'space': 'scaled features'}
                elif kind == 'backtest' and isinstance(upstream, ModelResult):
                    output = backtest(upstream, params)
                    report.update(output)
                else:
                    raise ValueError(f'{kind} has a missing or invalid input.')
            except (ValueError, TypeError, RuntimeError, KeyError) as exc:
                if progress:
                    progress({'node': key, 'status': 'error', 'detail': str(exc)})
                raise ValueError(f'{node["data"].get("label") or kind}: {exc}') from exc
            audits = output.scaling_audits if isinstance(output, PreparedData) else output.get('scaling_audits',[]) if isinstance(output,dict) else []
            report['scaling_audits'].extend({'id':key,'node_title':node['data'].get('label',kind),**audit} for audit in audits)
            self.outputs[key] = output
            summary = {'id': key, 'kind': kind, 'status': 'complete'}
            if isinstance(output, pd.DataFrame):
                summary.update(serialize(output))
                summary['source_rows'] = output.attrs.get('raw_rows', len(output))
            report['nodes'].append(summary)
            if progress:
                details = {k: v for k, v in summary.items() if k not in ('id', 'kind', 'status')}
                if isinstance(output, PortfolioData):
                    details.update({'rows': len(output.prices), 'detail': ', '.join(output.prices.columns)})
                if isinstance(output, PreparedData):
                    details.update({'features': len(output.feature_names), 'samples': {'train': len(output.y_train), 'validation': len(output.y_val), 'test': len(output.y_test)}})
                progress({'node': key, 'status': 'complete', **details})
        return serialize(report)
