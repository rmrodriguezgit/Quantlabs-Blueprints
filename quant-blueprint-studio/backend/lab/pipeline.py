"""Traceable train -> validate -> evaluate pipeline with persisted artifacts.

Each run writes ``<out_dir>/<run_id>/`` containing config.json, dataset.json (source
metadata), metrics.json and model.joblib, so any number in a report can be traced back
to the exact data request, split and seed that produced it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .connectors import load
from .contracts import Dataset
from .splits import chronological_split

FEATURES = ('ret_1', 'ret_5', 'vol_10', 'mom_20', 'range_1')


@dataclass
class ExperimentConfig:
    source: str = 'synthetic'
    source_params: dict[str, Any] = field(default_factory=lambda: {'n': 1500, 'seed': 42})
    model: str = 'logistic'            # logistic | random_forest
    horizon: int = 1                   # predict sign of the return `horizon` bars ahead
    train: float = 0.6
    validation: float = 0.2
    seed: int = 42
    out_dir: str = 'outputs/lab_runs'

    def run_id(self) -> str:
        key = json.dumps({k: v for k, v in asdict(self).items() if k != 'out_dir'}, sort_keys=True, default=str)
        return hashlib.sha256(key.encode()).hexdigest()[:12]


def build_features(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Strictly trailing features; target uses the future and is aligned per row."""
    close = frame['Close'].astype(float)
    logret = np.log(close / close.shift(1))
    out = pd.DataFrame(index=frame.index)
    out['ret_1'] = logret
    out['ret_5'] = np.log(close / close.shift(5))
    out['vol_10'] = logret.rolling(10).std()
    out['mom_20'] = close / close.rolling(20).mean() - 1
    if {'High', 'Low'} <= set(frame.columns):
        out['range_1'] = (frame['High'] - frame['Low']) / close
    else:
        out['range_1'] = logret.abs()
    fwd = np.log(close.shift(-horizon) / close)
    out['target'] = (fwd > 0).astype(float).where(fwd.notna())
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _model(name: str, seed: int):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20, random_state=seed, n_jobs=1)
    raise ValueError(f'Modelo desconocido {name!r}: logistic | random_forest.')


def _score(model, part: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score
    x, y = part[list(FEATURES)].to_numpy(), part['target'].to_numpy()
    proba = model.predict_proba(x)[:, 1]
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) == 2 else float('nan')
    return {'accuracy': float(accuracy_score(y, proba > 0.5)), 'auc': auc,
            'base_rate': float(y.mean()), 'rows': int(len(y))}


def run_experiment(cfg: ExperimentConfig, dataset: Dataset | None = None) -> dict[str, Any]:
    import joblib
    ds = dataset or load(cfg.source, cfg.source_params)
    table = build_features(ds.frame, cfg.horizon)
    split = chronological_split(table, cfg.train, cfg.validation, embargo=cfg.horizon)
    model = _model(cfg.model, cfg.seed)
    model.fit(split.train[list(FEATURES)].to_numpy(), split.train['target'].to_numpy())
    metrics = {'validation': _score(model, split.validation), 'test': _score(model, split.test),
               'train': _score(model, split.train), 'splits': split.sizes(),
               'split_bounds': {k: [str(v.index[0]), str(v.index[-1])] for k, v in
                                (('train', split.train), ('validation', split.validation), ('test', split.test))}}
    run_id = cfg.run_id()
    out = Path(cfg.out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.json').write_text(json.dumps(asdict(cfg), indent=2, default=str))
    (out / 'dataset.json').write_text(json.dumps({**ds.meta.to_dict(), 'fingerprint': ds.meta.fingerprint()}, indent=2, default=str))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, out / 'model.joblib')
    return {'run_id': run_id, 'artifacts': str(out), 'metrics': metrics}
