"""Submenú 17 · Redes neuronales: el ejemplo multicapa se ejecuta sin internet y las validaciones fallan con mensajes claros."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault('QL_ARTIFACTS', tempfile.mkdtemp(prefix='nn-artifacts-'))

from backend.engine.dag_runner import DAGRunner, CONTRACTS  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
KINDS = ['nn_data', 'nn_guard', 'nn_input', 'nn_split', 'nn_scaler', 'nn_embedding', 'nn_hidden', 'nn_output', 'nn_train', 'nn_finetune', 'nn_predict']


def chain(steps, extra=()):
    nodes = [{'id': f'n{i}', 'type': 'blueprint', 'position': {'x': 0, 'y': 0}, 'data': {'kind': k, 'params': p}} for i, (k, p) in enumerate(steps)]
    edges = [{'id': f'e{i}', 'source': f'n{i}', 'target': f'n{i + 1}', 'sourceHandle': 'frame', 'targetHandle': 'frame'} for i in range(len(steps) - 1)]
    return {'nodes': nodes, 'edges': [*edges, *extra]}


SYN = ('nn_data', {'source': 'synthetic', 'ticker': 'SPY'})
HEAD = [SYN, ('nn_input', {'lags': 5}), ('nn_split', {}), ('nn_scaler', {})]


class NNBlueprint(unittest.TestCase):
    def test_contracts_and_catalog(self):
        ts = (ROOT / 'src/renderer/types/blueprint.ts').read_text()
        for k in KINDS:
            self.assertIn(k, CONTRACTS['nodes'])
            self.assertIn(f'  {k}: {{', ts)
        self.assertEqual(CONTRACTS['nodes']['nn_finetune']['extraInputs'], {'pretrained': 'frame'})

    def test_example_runs_offline(self):
        e = json.loads((ROOT / 'public/blueprints/nn-multicapa-yfinance.json').read_text())['entries'][0]
        g = e['graph']
        for n in g['nodes']:
            if n['data']['kind'] == 'nn_data':
                n['data']['params']['source'] = 'synthetic'
            if n['data']['kind'] in ('nn_train', 'nn_finetune'):
                n['data']['params']['epochs'] = 3
        r = DAGRunner(g).execute()
        topics = [s['topic'] for s in r['studies']]
        for k in KINDS:
            self.assertIn(k, topics)
        pred = [s for s in r['studies'] if s['topic'] == 'nn_predict']
        self.assertEqual(len(pred), 2)
        self.assertTrue(all('TE_RMSE' in s['metrics'] for s in pred))

    def test_variants(self):
        for task, extra in (('direction', {}), ('multiclass', {'mode': 'simple', 'features': 'LogReturn'})):
            g = chain([SYN, ('nn_input', {'lags': 5, **extra}), ('nn_split', {'mode': 'walk_forward', 'folds': 2}), ('nn_scaler', {'method': 'robust'}),
                       ('nn_hidden', {'units': '32', 'repeat': 2, 'norm': 'batch'}), ('nn_output', {'task': task}), ('nn_train', {'epochs': 2}), ('nn_predict', {})])
            self.assertIn('TE_accuracy', DAGRunner(g).execute()['studies'][-1]['metrics'])
        g = chain([SYN, ('nn_input', {'mode': 'polynomial', 'features': 'LogReturn,Range', 'degree': 3, 'horizon': 3}), ('nn_split', {}),
                   ('nn_scaler', {'method': 'minmax', 'feature_range': '-1,1'}), ('nn_hidden', {'units': 'custom', 'custom_units': 20, 'residual': True}),
                   ('nn_output', {'loss': 'mae'}), ('nn_train', {'epochs': 2}), ('nn_predict', {})])
        self.assertIn('TE_RMSE', DAGRunner(g).execute()['studies'][-1]['metrics'])

    def test_errors(self):
        cases = [
            (chain([*HEAD, ('nn_output', {'task': 'direction', 'loss': 'mse'})]), 'incompatible'),
            (chain([*HEAD, ('nn_output', {'activation': 'sigmoid'})]), 'Regresión con activación'),
            (chain([SYN, ('nn_input', {}), ('nn_output', {})]), 'Escalador'),
            (chain([*HEAD, ('nn_finetune', {})]), 'pretrained'),
            (chain([*HEAD, ('nn_scaler', {'operation': 'load_transform', 'save_as': 'no_existe'})]), 'No existe'),
            (chain([('nn_data', {'source': 'synthetic', 'synthetic_rows': 200}), ('nn_input', {'lags': 150})]), 'filas útiles'),
        ]
        for g, msg in cases:
            with self.assertRaisesRegex(ValueError, msg):
                DAGRunner(g).execute()


if __name__ == '__main__':
    unittest.main()
