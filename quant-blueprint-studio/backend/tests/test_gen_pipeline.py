"""Unidad 9 · Generativos en 10 herramientas: cadena, reglas sin fuga y errores (sin internet)."""
import numpy as np
import pandas as pd
import pytest
from backend.engine.nodes import gen_pipeline_nodes as G


def prices(n=900, seed=0):
    rng = np.random.default_rng(seed)
    vol = 0.01 * np.exp(np.cumsum(rng.normal(0, .05, n)).clip(-1, 1))
    close = 100 * np.exp(np.cumsum(rng.standard_t(5, n) * vol / 1.3))
    return pd.DataFrame({'Close': close, 'Volume': 1e6}, index=pd.bdate_range('2015-01-02', periods=n))


def chain(upto='gen_evaluate', **over):
    f, reps = prices(), {}
    for k in G.ORDER:
        f, reps[k] = G.STEPS[k](f, {'epochs': 4, **over.get(k, {})})
        if k == upto:
            break
    return f, reps


def test_windows_and_causal_condition():
    f, r = chain('gen_condition')
    st = G._state(f, 'gen_condition')
    ret = np.log(prices()['Close']).diff().dropna().to_numpy()
    i = st['starts'][0]
    assert i == 20 and np.allclose(st['W'][0], ret[i:i + 16], atol=1e-6)
    assert np.isclose(st['C'][0, 0], np.log(np.std(ret[0:20], ddof=1)), atol=1e-5)


def test_split_purges_and_scaler_uses_train():
    f, r = chain('gen_scale')
    st = G._state(f, 'gen_scale')
    d, s, h = st['dates'], st['starts'], st['horizon']
    vs, ts = (pd.Timestamp(st['cuts'][k]) for k in ('VAL_START', 'TEST_START'))
    assert (d[s[st['group'] == 0] + h - 1] < vs).all()
    assert (d[s[st['group'] == 1]] >= vs).all() and (d[s[st['group'] == 1] + h - 1] < ts).all()
    assert (d[s[st['group'] == 2]] >= ts).all() and r['gen_split']['metrics']['purgadas'] > 0
    tr = st['GX'][st['group'] == 0]
    assert abs(tr.mean()) < 1e-4 and abs(tr.std() - 1) < 1e-3
    assert r['gen_scale']['scaling_audits']


def test_full_chain_selects_on_validation_and_reports_test():
    f, r = chain()
    assert r['gen_select']['selected'].split(' ')[0] in ('CVAE', 'cGAN')
    m = r['gen_evaluate']['metrics']
    assert np.isfinite(m['W1_TE']) and 0 <= m['excesos_VaR5_acum'] <= 1
    assert {'Gen_VaR_cum', 'Real_cum'} <= set(f.columns) and f['Gen_VaR_cum'].notna().sum() > 8
    assert any(t['fuente'] == 'Bootstrap TR' for t in r['gen_evaluate']['table'])


def test_reproducible():
    a, _ = chain()
    b, _ = chain()
    assert np.allclose(a['Gen_VaR_cum'].dropna(), b['Gen_VaR_cum'].dropna())


def test_order_and_errors():
    with pytest.raises(ValueError, match='01'):
        G.condition(prices(), {})
    f, _ = chain('gen_scale')
    with pytest.raises(ValueError, match='modelos entrenados'):
        G.select(f, {})
    with pytest.raises(ValueError, match='0.95'):
        chain('gen_split', gen_split={'train_ratio': .85, 'val_ratio': .12})
    with pytest.raises(ValueError, match='retornos'):
        G.windows(prices(), {'return_column': 'Close'})


def test_gan_only_branch():
    f, _ = chain('gen_scale')
    f, _ = G.gan_train(f, {'epochs': 3})
    f, rs = G.select(f, {})
    assert rs['selected'].startswith('cGAN')


def test_dag_runner_executes_library_entry():
    import json
    from pathlib import Path
    from backend.engine.dag_runner import DAGRunner
    lib = json.loads((Path(__file__).resolve().parents[2] / 'public/blueprints/unit-9-tools.json').read_text())
    g = lib['entries'][0]['graph']
    for n in g['nodes']:
        if n['data']['kind'] in ('gen_vae_train', 'gen_gan_train'):
            n['data']['params']['epochs'] = 3
    report = DAGRunner(g).execute()
    topics = [s.get('topic') for s in report['studies']]
    assert all(k in topics for k in G.ORDER)
