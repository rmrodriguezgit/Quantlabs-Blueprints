"""«Cómo programar: LSTM y GRU»: each of the ten segments, checked."""
import unittest

import numpy as np
import torch

from backend.dl_tools.core import RunContext, registry
from backend.dl_tools.tools.dl.recurrent_compare.model import RecurrentRegressor
from backend.dl_tools.tools.dl.recurrent_compare.tool import make_sequences

CPU = RunContext(device='cpu', seed=1)


class Segments(unittest.TestCase):
    def test_01_sequences_shape_and_alignment(self):
        x = np.arange(30 * 11, dtype=float).reshape(30, 11)
        y = np.arange(30, dtype=float)
        sx, sy, ends = make_sequences(x, y, 20)
        self.assertEqual(sx.shape, (11, 20, 11))                 # (batch, 20, 11)
        self.assertTrue(np.array_equal(sx[0][-1], x[19]))        # window ends at session t …
        self.assertEqual(sy[0], y[19])                           # … and its target is the value stored on row t

    def test_02_to_07_shapes_states_and_head(self):
        x = torch.randn(4, 20, 11)
        for kind in ('LSTM', 'GRU'):
            for bi in (False, True):
                m = RecurrentRegressor(kind, 11, 32, bi)
                self.assertIsInstance(m.input_layer, torch.nn.LSTM if kind == 'LSTM' else torch.nn.GRU)
                self.assertEqual(m.input_layer.bidirectional, bi)
                self.assertEqual(m.summary(x).shape, (4, 64 if bi else 32))     # 06: cat(h[-2], h[-1])
                self.assertEqual(m.output_layer.in_features, 64 if bi else 32)  # 07: nn.Linear(64 if bi else 32, 1)
                self.assertEqual(m(x).shape, (4,))

    def test_06_bidirectional_summary_is_forward_and_backward(self):
        m = RecurrentRegressor('GRU', 11, 8, True)
        x = torch.randn(2, 20, 11)
        _, h = m.input_layer(x)
        self.assertTrue(torch.equal(m.summary(x), torch.cat([h[-2], h[-1]], dim=1)))

    def test_08_no_state_carried_between_windows(self):
        m = RecurrentRegressor('LSTM', 11, 16, False).eval()
        a, b = torch.randn(1, 20, 11), torch.randn(1, 20, 11)
        with torch.no_grad():
            first = m(a)
            m(b)
            again = m(a)
        self.assertTrue(torch.equal(first, again))

    def test_09_10_selection_ridge_and_units(self):
        rng = np.random.default_rng(0)
        n = 500
        x = rng.normal(size=(n, 11))
        y = np.full(n, np.nan)
        y[:-1] = 0.002 * np.tanh(x[:-1, 0] + 0.5 * x[:-1, 1]) + 0.001 * rng.normal(size=n - 1)
        with registry.get('dl_recurrent_compare')({'epochs': 30, 'candidates': ['LSTM', 'GRU']}) as tool:
            r = tool.execute({'features': x.tolist(), 'target': y.tolist()}, CPU)
        rows = r.outputs['comparison']
        rnn = [row for row in rows if row['modelo'] != 'Ridge (línea base)']
        self.assertEqual(r.outputs['selected'], min(rnn, key=lambda row: row['validation_MSE'])['modelo'])
        self.assertTrue(any(row['modelo'].startswith('Ridge') for row in rows))
        self.assertIsNotNone(r.outputs['next'])                  # last session has no target → live estimate
        self.assertGreater(np.std(r.outputs['actual']), 5)       # bps, not raw returns (≈ 0.002 × 10 000)
        self.assertLess(r.diagnostics['test']['selected_MSE'], r.diagnostics['test']['zero_MSE'])


if __name__ == '__main__':
    unittest.main()
