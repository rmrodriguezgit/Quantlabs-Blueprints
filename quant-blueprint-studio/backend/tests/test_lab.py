"""Offline tests for the ML lab core: contract, connectors (network mocked), splits, pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from backend.lab import (ConnectorError, ContractError, ExperimentConfig, build_features, chronological_split,
                         load, make_dataset, run_experiment, walk_forward)


def _ohlcv(n=300, tz=None):
    idx = pd.date_range('2025-01-01', periods=n, freq='h', tz=tz)
    c = 100 + np.cumsum(np.random.default_rng(0).normal(0, 1, n))
    return pd.DataFrame({'Open': c, 'High': c + 1, 'Low': c - 1, 'Close': c, 'Volume': 1.0}, index=idx)


class ContractTests(unittest.TestCase):
    def test_normalizes_tz_order_and_duplicates(self):
        f = _ohlcv(tz='America/New_York').iloc[::-1]
        f = pd.concat([f, f.iloc[:3]])
        ds = make_dataset(f, source='synthetic', symbol='X', freq='1h', params={})
        self.assertIsNone(ds.frame.index.tz)
        self.assertTrue(ds.frame.index.is_monotonic_increasing)
        self.assertFalse(ds.frame.index.has_duplicates)
        self.assertEqual(ds.meta.rows, 300)

    def test_rejects_unknown_source_and_missing_close(self):
        with self.assertRaises(ContractError):
            make_dataset(_ohlcv(), source='nope', symbol='X', freq='1h', params={})
        with self.assertRaises(ContractError):
            make_dataset(_ohlcv().drop(columns='Close'), source='synthetic', symbol='X', freq='1h', params={})


class ConnectorTests(unittest.TestCase):
    def test_synthetic_reproducible_by_seed(self):
        a, b, c = load('synthetic', {'n': 400, 'seed': 7}), load('synthetic', {'n': 400, 'seed': 7}), load('synthetic', {'n': 400, 'seed': 8})
        pd.testing.assert_frame_equal(a.frame, b.frame)
        self.assertFalse(a.frame['Close'].equals(c.frame['Close']))
        self.assertEqual(a.meta.fingerprint(), b.meta.fingerprint())

    def test_yfinance_wrapped_and_retried(self):
        calls = {'n': 0}

        def flaky(p, progress=None):
            calls['n'] += 1
            if calls['n'] == 1:
                raise ConnectionError('rate limited')
            return _ohlcv(tz='UTC'), {}
        with mock.patch('backend.engine.nodes.yfinance_nodes.yfinance_data', flaky):
            ds = load('yfinance', {'ticker': 'spy', 'backoff': 0})
        self.assertEqual((ds.meta.source, ds.meta.symbol, calls['n']), ('yfinance', 'SPY', 2))

    def test_validation_error_not_retried(self):
        with mock.patch('backend.engine.nodes.yfinance_nodes.yfinance_data', side_effect=ValueError('bad ticker')) as m:
            with self.assertRaises(ConnectorError):
                load('yfinance', {'ticker': 'X', 'backoff': 0})
        self.assertEqual(m.call_count, 1)

    def test_polymarket_drops_snapshot_columns(self):
        f = _ohlcv()
        f['P_up'] = .5
        f['Book_spread'] = np.nan
        with mock.patch('backend.engine.nodes.polymarket_nodes.polymarket_source', return_value=(f, {})):
            ds = load('polymarket', {'market': 'btc-updown-5m'})
        self.assertNotIn('Book_spread', ds.frame.columns)
        self.assertEqual(ds.meta.source, 'polymarket')
        with self.assertRaises(ConnectorError):
            load('polymarket', {})


class SplitTests(unittest.TestCase):
    def test_chronological_with_embargo(self):
        f = _ohlcv(100)
        s = chronological_split(f, .6, .2, embargo=3)
        self.assertLess(s.train.index[-1], s.validation.index[0])
        self.assertLess(s.validation.index[-1], s.test.index[0])
        self.assertEqual(s.sizes(), {'train': 57, 'validation': 17, 'test': 20})

    def test_walk_forward_train_before_test(self):
        for tr, te in walk_forward(_ohlcv(200), folds=4, embargo=2):
            self.assertLess(tr.index[-1], te.index[0])


class PipelineTests(unittest.TestCase):
    def test_features_are_trailing(self):
        f = load('synthetic', {'n': 300, 'seed': 1}).frame
        full = build_features(f, 1)
        cut = build_features(f.iloc[:200], 1)
        common = cut.index[:-1]
        pd.testing.assert_frame_equal(full.loc[common].drop(columns='target'), cut.loc[common].drop(columns='target'))

    def test_run_persists_artifacts_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = ExperimentConfig(source_params={'n': 800, 'seed': 3}, out_dir=d)
            r1, r2 = run_experiment(cfg), run_experiment(cfg)
            self.assertEqual(r1['metrics'], r2['metrics'])
            out = Path(r1['artifacts'])
            for name in ('config.json', 'dataset.json', 'metrics.json', 'model.joblib'):
                self.assertTrue((out / name).exists(), name)
            self.assertEqual(json.loads((out / 'dataset.json').read_text())['source'], 'synthetic')
            self.assertLess(r1['metrics']['split_bounds']['train'][1], r1['metrics']['split_bounds']['validation'][0])


if __name__ == '__main__':
    unittest.main()
