"""Autoencoders (Unidad 8): las diez herramientas independientes."""
import json
import unittest

import numpy as np
import torch

from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes import ae_pipeline_nodes as A

KINDS = list(A.STEPS)
FEATURES = {'lags': 0, 'use_rsi': True, 'use_ema': True, 'use_sma': True, 'use_macd': True, 'use_vwap': True, 'use_bb': True}


def graph(skip: tuple[str, ...] = (), params: dict | None = None) -> dict:
    nodes = [{'id': 'SRC', 'data': {'kind': 'source', 'params': {'source': 'synthetic', 'years': 5}}},
             {'id': 'FT', 'data': {'kind': 'features', 'params': FEATURES}}]
    nodes += [{'id': k, 'data': {'kind': k, 'params': {'epochs': 15, **(params or {}).get(k, {})}}} for k in KINDS if k not in skip]
    ids = [n['id'] for n in nodes]
    return {'nodes': nodes, 'edges': [{'id': f'e{i}', 'source': a, 'target': b, 'sourceHandle': 'frame', 'targetHandle': 'frame'}
                                      for i, (a, b) in enumerate(zip(ids, ids[1:]))]}


class AEPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.result = DAGRunner(graph()).execute()

    def test_every_tool_reports(self) -> None:
        topics = [s['topic'] for s in self.result['studies']]
        self.assertEqual(topics, KINDS)
        json.dumps(self.result, allow_nan=False)
        for s in self.result['studies']:
            self.assertTrue(s['selected'], s['topic'])

    def test_split_and_scaling_have_no_leakage(self) -> None:
        split, scale = self.result['studies'][0], self.result['studies'][1]
        self.assertEqual(len(split['features']), 11)
        self.assertTrue(all(split['splits'][k] >= 8 for k in ('train', 'validation', 'test')))
        self.assertLess(scale['metrics']['max |media TR escalada|'], 1e-5)

    def test_selection_and_test_metrics(self) -> None:
        sel = next(s for s in self.result['studies'] if s['topic'] == 'ae_select')
        self.assertEqual(set(sel['metrics']), {'PCA', 'AE', 'DAE'})
        self.assertEqual(sel['selected'], min(sel['metrics'], key=sel['metrics'].get))
        test = self.result['studies'][-1]
        self.assertEqual({r['modelo'] for r in test['table']}, {'PCA', 'AE', 'DAE'})
        self.assertTrue(all(np.isfinite(r['test_reconstruction_MSE']) for r in test['table']))

    def test_dae_without_noise_fails_clearly(self) -> None:
        with self.assertRaisesRegex(Exception, 'Ruido'):
            DAGRunner(graph(skip=('ae_noise',))).execute()

    def test_ae_only_without_noise_runs(self) -> None:
        out = DAGRunner(graph(skip=('ae_noise',), params={'ae_train': {'train_dae': False}})).execute()
        sel = next(s for s in out['studies'] if s['topic'] == 'ae_select')
        self.assertEqual(set(sel['metrics']), {'PCA', 'AE'})

    def test_training_is_reproducible(self) -> None:
        a = DAGRunner(graph()).execute()['studies'][6]['metrics']
        self.assertEqual(a, self.result['studies'][6]['metrics'])


if __name__ == '__main__':
    unittest.main()
