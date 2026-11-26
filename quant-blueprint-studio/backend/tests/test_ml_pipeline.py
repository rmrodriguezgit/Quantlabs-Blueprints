"""ML Pipeline modules: chain, leakage rules and errors (offline, synthetic prices)."""
import numpy as np
import pandas as pd
import unittest
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes import ml_pipeline_nodes as M


def prices(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.bdate_range('2022-01-03', periods=n)
    f = pd.DataFrame({'Open': close, 'High': close * 1.01, 'Low': close * .99, 'Close': close, 'Volume': rng.integers(1e5, 1e6, n).astype(float)}, index=idx)
    f.loc[f.index[10], 'Volume'] = np.nan
    return f


def chain(task_mode='next', algorithm='Compare', **split):
    f, _ = M.select_columns(prices(), {'columns': 'Close,Volume', 'mode': 'keep'})
    f, _ = M.returns(f, {'columns': 'Close', 'return_type': 'both', 'periods': 1})
    f['Lag1'] = f['LogRet_Close'].shift(1)
    f, _ = M.clean(f, {'missing': 'ffill', 'outliers': 'clip_zscore'})
    f, rs = M.split(f, {'target': 'LogRet_Close', 'target_mode': task_mode, 'features': 'Lag1,Ret_Close,Volume', **split})
    f, rt = M.train(f, {'algorithm': algorithm, 'algorithm_cls': algorithm, 'estimators': 30})
    return f, rs, rt


def test_full_chain_regression():
    f, rs, rt = chain()
    assert set(f['Split']) == {'TR', 'VA', 'TE', 'FUT'}
    assert rs['splits']['train'] > rs['splits']['validation'] > 0
    f, re = M.evaluate(f, {})
    assert 'RMSE' in re['metrics'] and any(r['conjunto'] == 'TE' for r in re['table'])
    pr, rp = M.predict(f, {'scope': 'test_future'})
    assert (pr['Split'] == 'FUT').sum() == 1 and 'Lower' in pr
    _, rv = M.visualize(pr, {})
    assert rv['charts']


def test_classification_direction():
    f, _, rt = chain('direction', 'Logistic')
    assert f.attrs['ml']['task'] == 'classification' and 'Proba' in f
    _, re = M.evaluate(f, {})
    assert 'Accuracy' in re['metrics']


def test_scaler_fitted_on_train_only():
    f, *_ = chain()
    tr = f[f['Split'] == 'TR']
    assert abs(tr['Lag1'].mean()) < 1e-9 and abs(tr['Lag1'].std(ddof=0) - 1) < 1e-9


def test_chronological_and_gap():
    f, *_ = chain(gap=5)
    order = {k: f.index[f['Split'] == k] for k in ('TR', 'VA', 'TE')}
    assert order['TR'].max() < order['VA'].min() < order['VA'].max() < order['TE'].min()


def test_errors_are_explicit():
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Dividir'):
        M.train(prices(), {})
    with unittest.TestCase().assertRaisesRegex(ValueError, 'Limpieza'):
        M.split(prices(), {'target': 'Close'})
    with unittest.TestCase().assertRaisesRegex(ValueError, 'inexistentes'):
        M.select_columns(prices(), {'columns': 'Nope'})


def test_dag_runs_pipeline():
    kinds = ['source', 'ml_select', 'ml_returns', 'ml_clean', 'ml_split', 'ml_train', 'ml_evaluate', 'ml_predict', 'ml_visualize']
    params = {'source': {'source': 'synthetic', 'years': 2}, 'ml_select': {'columns': 'Close,Volume'}, 'ml_returns': {'return_type': 'log'},
              'ml_split': {'target': 'LogRet_Close'}, 'ml_train': {'algorithm': 'Ridge'}}
    nodes = [{'id': k, 'data': {'kind': k, 'label': k, 'params': params.get(k, {})}} for k in kinds]
    edges = [{'id': f'e{i}', 'source': a, 'target': b, 'sourceHandle': 'frame', 'targetHandle': 'frame'} for i, (a, b) in enumerate(zip(kinds, kinds[1:]))]
    report = DAGRunner({'nodes': nodes, 'edges': edges}).execute()
    assert len(report['studies']) == 8


class MLPipeline(unittest.TestCase):
    pass


for _name, _fn in list(globals().items()):
    if _name.startswith('test_') and callable(_fn):
        setattr(MLPipeline, _name, staticmethod(_fn))

if __name__ == '__main__':
    unittest.main()
