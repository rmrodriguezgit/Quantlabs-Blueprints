"""Typed data pins and Kahn DAG execution. No graph-supplied code is evaluated."""
from collections import deque
from typing import Any, Callable
import numpy as np
import pandas as pd
import torch
from .nodes.data_nodes import source, features, merge_features
from .nodes.stats_nodes import PreparedData, prepare
from .nodes.dl_nodes import ModelResult, NetworkSpec, train
from .nodes.backtest_nodes import backtest
from .nodes.plot_nodes import plot

PINS = {'source': (None, 'frame'), 'features': ('frame', 'frame'), 'prepare': ('frame', 'tensor'),
        'model': ('tensor', 'tensor'), 'backtest': ('tensor', None), 'config': (None, 'config'),
        'network_input': (None, 'network'), 'layer': ('network', 'network'),
        'network_output': ('network', 'network'), 'merge': ('frame', 'frame'), 'plot': (None, None)}
INPUTS = {kind: ({pin: pin} if pin else {}) for kind, (pin, _) in PINS.items()}
for kind in ('features', 'prepare', 'model', 'backtest'):
    INPUTS[kind]['config'] = 'config'
INPUTS['model']['network'] = 'network'
INPUTS['merge']['frame_b'] = 'frame'
INPUTS['plot'] = {'frame': 'frame', 'tensor': 'tensor'}
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
        report: dict[str, Any] = {'decomposition': [], 'predictions': [], 'backtest': [], 'metrics': {}, 'history': [], 'nodes': [], 'plots': []}
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
                elif kind == 'plot':
                    if len(inputs) != 1:
                        raise ValueError('Plot requires exactly one frame or tensor input.')
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
            self.outputs[key] = output
            summary = {'id': key, 'kind': kind, 'status': 'complete'}
            if isinstance(output, pd.DataFrame):
                summary.update(serialize(output))
                summary['source_rows'] = output.attrs.get('raw_rows', len(output))
            report['nodes'].append(summary)
            if progress:
                details = {k: v for k, v in summary.items() if k not in ('id', 'kind', 'status')}
                if isinstance(output, PreparedData):
                    details.update({'features': len(output.feature_names), 'samples': {'train': len(output.y_train), 'validation': len(output.y_val), 'test': len(output.y_test)}})
                progress({'node': key, 'status': 'complete', **details})
        return serialize(report)
