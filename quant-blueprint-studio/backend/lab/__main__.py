"""CLI: python -m backend.lab --source synthetic --model logistic [--param seed=7 --param n=2000]."""
import argparse
import json

from .pipeline import ExperimentConfig, run_experiment


def _value(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def main() -> None:
    ap = argparse.ArgumentParser(prog='python -m backend.lab')
    ap.add_argument('--source', default='synthetic', choices=['synthetic', 'yfinance', 'polymarket'])
    ap.add_argument('--model', default='logistic', choices=['logistic', 'random_forest'])
    ap.add_argument('--horizon', type=int, default=1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out-dir', default='outputs/lab_runs')
    ap.add_argument('--param', action='append', default=[], help='source param key=value (repeatable)')
    a = ap.parse_args()
    params = dict(kv.split('=', 1) for kv in a.param)
    params = {k: _value(v) for k, v in params.items()}
    if a.source == 'synthetic':
        params = {'n': 1500, 'seed': a.seed, **params}
    cfg = ExperimentConfig(source=a.source, source_params=params, model=a.model, horizon=a.horizon,
                           seed=a.seed, out_dir=a.out_dir)
    print(json.dumps(run_experiment(cfg), indent=2))


if __name__ == '__main__':
    main()
