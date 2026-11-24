"""Example: chain forecast → anomaly (on its residuals) and a classifier, then show memory was released.

Run:  .venv/bin/python -m backend.dl_tools.examples.run_blueprint
"""
import json
import logging

import numpy as np

from ..core import RunContext, registry
from ..core.graph import run
from ..core.memory import device_bytes

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(message)s')
rng = np.random.default_rng(0)
t = np.arange(600)
series = (np.sin(t / 8) + 0.1 * rng.normal(size=600))
series[520] += 2.5                       # a shock the anomaly tool should find in the forecast residuals
graph = {
    'inputs': {'series': series.tolist()},
    'nodes': [{'id': 'forecast', 'tool': 'dl_tabular_forecast', 'params': {'architecture': 'lstm', 'epochs': 25, 'horizon': 3}},
              {'id': 'classify', 'tool': 'dl_sequence_classifier', 'params': {'epochs': 15}},
              {'id': 'anomaly', 'tool': 'dl_anomaly_autoencoder', 'params': {'latent': 1, 'epochs': 60, 'fit_fraction': 0.5}}],
    'edges': [{'source': '$series', 'target': 'forecast', 'target_port': 'series'},
              {'source': '$series', 'target': 'classify', 'target_port': 'series'},
              {'source': 'forecast', 'source_port': 'residuals', 'target': 'anomaly', 'target_port': 'data'}],
}
print('Registered:', [s.name for s in registry.specs()])
report = run(graph, RunContext(device='auto', seed=42))
out = report['outputs']
for key, node in report['nodes'].items():
    print(f"{key:9s} {node['device']:4s} {node['weights']:16s} {node['inference_ms']:8.1f} ms")
print('forecast  next =', [round(v, 4) for v in out['forecast']['forecast']], 'confidence', round(out['forecast']['confidence'], 3))
print('classify  label =', out['classify']['label'], {k: round(v, 3) for k, v in out['classify']['probabilities'].items()})
flags = np.flatnonzero(out['anomaly']['is_anomaly'])
first_target = report['nodes']['forecast']['diagnostics']['fit_windows'] + 20   # residual row i holds errors for series[fit_windows + lookback + i … +2]
print('anomaly   flagged windows (series indices) =', [f'{i}–{i + 2}' for i in (flags + first_target)], '· injected shock at 520', 'threshold', round(out['anomaly']['threshold'], 4))
print('memory   ', json.dumps(report['memory']), '| now allocated:', {d: device_bytes(d) for d in ('mps', 'cuda', 'cpu')})
