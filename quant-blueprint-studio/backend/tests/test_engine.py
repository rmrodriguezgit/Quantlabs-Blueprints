"""Deterministic synthetic tests: DAG validation, leakage and executable models."""
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient
from backend.server import app
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.data_nodes import source, features
from backend.engine.nodes.stats_nodes import prepare
from backend.engine.nodes.dl_nodes import train
from backend.engine.nodes.backtest_nodes import backtest


def graph() -> dict:
    kinds = ['source', 'features', 'prepare', 'model', 'backtest']
    nodes = [{'id': str(i), 'data': {'kind': kind, 'params': {'years': 1} if i == 0 else {'epochs': 1, 'hidden': 8} if i == 3 else {}}} for i, kind in enumerate(kinds)]
    return {'nodes': nodes, 'edges': [{'source': str(i), 'target': str(i+1), 'sourceHandle': 'frame' if i < 2 else 'tensor', 'targetHandle': 'frame' if i < 2 else 'tensor'} for i in range(4)]}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.frame = features(source({'years': 1}), {})

    def test_unordered_dag_executes(self) -> None:
        g = graph()
        g['nodes'].reverse()
        result = DAGRunner(g).execute()
        self.assertEqual(len(result['nodes']), 5)
        self.assertEqual(len(result['predictions']), len(result['backtest']))
        self.assertTrue(np.isfinite(result['metrics']['MSE']))
        self.assertTrue(set(row['signal'] for row in result['backtest']) <= {-1, 0, 1})

    def test_cycles_missing_nodes_and_pins(self) -> None:
        g = graph()
        # A self-loop with compatible tensor pins still must fail cycle validation.
        g['edges'][2] = {'source': '3', 'target': '3', 'sourceHandle': 'tensor', 'targetHandle': 'tensor'}
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            DAGRunner(g).topological_sort()
        g = graph()
        g['edges'][0]['source'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'missing'):
            DAGRunner(g)
        g = graph()
        g['edges'][0]['sourceHandle'] = 'tensor'
        with self.assertRaisesRegex(ValueError, 'pin'):
            DAGRunner(g)

    def test_scaler_does_not_see_test_data(self) -> None:
        original = prepare(self.frame, {})
        changed = self.frame.copy()
        changed.iloc[-20:, changed.columns.get_loc('Close')] *= 100
        modified = prepare(changed, {})
        np.testing.assert_array_equal(original.x_train, modified.x_train)
        np.testing.assert_array_equal(original.scaler.mean_, modified.scaler.mean_)
        expected = self.frame['LogReturn'].to_numpy()[-len(original.y_test):]
        np.testing.assert_allclose(original.y_test, expected)

    def test_all_architectures(self) -> None:
        prepared = prepare(self.frame, {})
        for architecture in ['Dense', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'CNN-LSTM', 'Autoencoder', 'DAE', 'DQN']:
            with self.subTest(architecture=architecture):
                result = train(prepared, {'architecture': architecture, 'hidden': 8, 'epochs': 1})
                self.assertEqual(result.predicted.shape, prepared.y_test.shape)
                self.assertTrue(np.isfinite(result.predicted).all())

    def test_costs_and_forecast_alignment(self) -> None:
        data = prepare(self.frame, {})
        result = train(data, {'epochs': 1, 'hidden': 8})
        result.predicted = np.ones_like(result.predicted)
        zero = backtest(result, {'fee_bps': 0, 'slippage_bps': 0})
        np.testing.assert_allclose([r['strategy'] for r in zero['backtest']], [r['market'] for r in zero['backtest']])
        costly = backtest(result, {'fee_bps': 20})
        self.assertLess(costly['trading']['total_return'], zero['trading']['total_return'])
        self.assertEqual(costly['trading']['turnover'], 2)

    def test_api(self) -> None:
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/health').status_code, 200)
            self.assertEqual(client.post('/api/execute', json={'nodes': []}).status_code, 422)
            self.assertEqual(client.post('/api/execute', json=graph()).status_code, 200)
            with client.websocket_connect('/ws/execute') as ws:
                ws.send_json(graph())
                events = []
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get('event') in ('result', 'error'):
                        break
                self.assertEqual(events[-1]['event'], 'result')
                self.assertEqual(sum(e.get('status') == 'complete' for e in events), 5)


if __name__ == '__main__':
    unittest.main()
