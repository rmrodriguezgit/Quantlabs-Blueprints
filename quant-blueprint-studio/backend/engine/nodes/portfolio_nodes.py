"""Aligned multi-asset data and long-only Markowitz evaluated on untouched dates."""
from dataclasses import dataclass
import hashlib
from typing import Any, Callable
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from .data_nodes import source
from .metrics_nodes import analyze, RiskAnalysis, settings


@dataclass
class PortfolioData:
    prices: pd.DataFrame
    benchmark: pd.Series
    benchmark_name: str
    source: str
    alignment: dict[str, Any]


@dataclass
class PortfolioResult:
    data: PortfolioData
    returns: pd.DataFrame
    benchmark: pd.Series
    report: dict[str, Any]


def portfolio_data(params: dict[str, Any], progress: Callable[[str], None] | None = None) -> PortfolioData:
    tickers = [t.strip().upper() for t in str(params.get('tickers', 'SPY,QQQ,TLT')).split(',') if t.strip()]
    if not 2 <= len(tickers) <= 20 or len(set(tickers)) != len(tickers) or any(len(t) > 20 for t in tickers):
        raise ValueError('Choose 2–20 unique tickers, separated by commas.')
    name = str(params.get('benchmark', 'SPY')).strip().upper()
    if not name or len(name) > 20:
        raise ValueError('Choose one valid benchmark ticker.')
    years = int(params.get('years', 3))
    mode = str(params.get('source', 'synthetic'))
    if mode not in ('synthetic', 'yfinance') or not 1 <= years <= 10:
        raise ValueError('Source must be synthetic/yfinance, history 1–10 years.')
    symbols = list(dict.fromkeys([*tickers, name]))
    series: dict[str, pd.Series] = {}
    if mode == 'synthetic':
        index = pd.bdate_range(end='2025-12-31', periods=252 * years)
        factor = np.random.default_rng(42).normal(.00025, .008, len(index))
        for ticker in symbols:
            seed = int.from_bytes(hashlib.sha256(ticker.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(seed)
            beta = rng.uniform(.2, 1.4)
            log_returns = beta * factor + rng.normal(rng.uniform(-.0001, .0004), rng.uniform(.003, .012), len(index))
            series[ticker] = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=index, name=ticker)
            if progress:
                progress(ticker)
    else:
        for ticker in symbols:
            if progress:
                progress(ticker)
            series[ticker] = source({'source': 'yfinance', 'years': years, 'ticker': ticker})['Close'].rename(ticker)
    before = {name: len(values) for name, values in series.items()}
    joined = pd.concat(series.values(), axis=1).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if len(joined) < 100 or not joined.index.is_unique or (joined <= 0).any().any():
        raise ValueError('Portfolio requires at least 100 common dates of finite positive adjusted closes; increase history or change assets.')
    benchmark = joined[name].copy().rename(name)
    return PortfolioData(joined[tickers], benchmark, name, mode,
                         {'available_rows': before, 'common_price_rows': len(joined), 'calendar': 'intersection; no forward fill'})


def project(weights: np.ndarray, cap: float) -> np.ndarray:
    lo, hi = float(weights.min() - cap), float(weights.max())
    for _ in range(80):
        middle = (lo + hi) / 2
        if np.clip(weights - middle, 0, cap).sum() > 1:
            lo = middle
        else:
            hi = middle
    return np.clip(weights - (lo + hi) / 2, 0, cap)


def optimize_weights(train: pd.DataFrame, annual: int, rf: float, cap: float, covariance: str) -> dict[str, np.ndarray]:
    n = train.shape[1]
    if cap * n < 1 - 1e-10 or not 0 < cap <= 1:
        raise ValueError(f'Maximum weight {cap:g} is infeasible for {n} assets; cap × asset count must be ≥1.')
    if covariance not in ('shrinkage', 'sample'):
        raise ValueError('Covariance must be shrinkage or sample.')
    values = train.to_numpy()
    cov = (LedoitWolf().fit(values).covariance_ if covariance == 'shrinkage' else np.cov(values, rowvar=False, ddof=1)) * annual
    # Small ridge only stabilizes numerical null spaces; report the estimator selected above.
    cov = cov + np.eye(n) * 1e-10
    mean = values.mean(axis=0) * annual
    excess = mean - np.expm1(np.log1p(rf) / annual) * annual
    if np.max(np.diag(cov)) <= 1.01e-10:
        raise ValueError('Asset returns have no variance; maximum Sharpe is undefined.')
    initial = np.full(n, 1 / n)
    if abs(cap * n - 1) < 1e-12:
        return {'min_variance': initial.copy(), 'max_sharpe': initial.copy()}
    constraints = [{'type': 'eq', 'fun': lambda w: float(w.sum() - 1), 'jac': lambda w: np.ones(n)}]
    bounds = [(0., cap)] * n
    def variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)
    def negative_sharpe(w: np.ndarray) -> float:
        return -float(excess @ w) / np.sqrt(max(variance(w), 1e-16))
    output: dict[str, np.ndarray] = {}
    for name, objective, starts in [('min_variance', variance, [initial]),
                                    ('max_sharpe', negative_sharpe, [initial, *[project(np.eye(n)[i], cap) for i in range(n)]])]:
        candidates = []
        for start in starts:
            result = minimize(objective, start, method='SLSQP', bounds=bounds, constraints=constraints,
                              options={'ftol': 1e-11, 'maxiter': 1000})
            if result.success and np.isfinite(result.fun) and abs(result.x.sum() - 1) < 1e-6 and result.x.min() >= -1e-7 and result.x.max() <= cap + 1e-7:
                candidates.append(result)
        if not candidates:
            raise ValueError(f'Markowitz {name} failed to converge; change assets, covariance estimator or weight cap.')
        best = min(candidates, key=lambda candidate: candidate.fun)
        output[name] = project(best.x, cap)
    return output


def weighted_returns(asset_returns: pd.DataFrame, weights: np.ndarray, rebalance: str, cost_bps: float) -> pd.Series:
    values = asset_returns.to_numpy()
    cost = cost_bps / 10000
    if rebalance == 'daily':
        gross = values @ weights
        drift = weights[None, :] * (1 + values[:-1]) / (1 + gross[:-1, None])
        turnover = np.r_[np.abs(weights).sum(), np.abs(weights[None, :] - drift).sum(axis=1)]
    elif rebalance == 'buy_hold':
        wealth = np.cumprod(1 + values, axis=0) @ weights
        gross = wealth / np.r_[1., wealth[:-1]] - 1
        turnover = np.zeros(len(gross))
        turnover[0] = 1
    else:
        raise ValueError('Rebalance must be daily or buy_hold.')
    turnover[-1] += 1  # Liquidate the terminal allocation.
    net = gross - turnover * cost
    if np.any(net <= -1) or not np.isfinite(net).all():
        raise ValueError('Portfolio equity is exhausted after trading costs; reduce costs or change assets.')
    return pd.Series(net, index=asset_returns.index)


def build_portfolio(data: PortfolioData, params: dict[str, Any]) -> PortfolioResult:
    annual, rf, _ = settings(params)
    fraction = float(params.get('train_fraction', .7))
    cap = float(params.get('max_weight', 1))
    fee = float(params.get('fee_bps', 0))
    slippage = float(params.get('slippage_bps', 0))
    if not .5 <= fraction <= .9 or not 0 <= fee <= 1000 or not 0 <= slippage <= 1000:
        raise ValueError('Train fraction .5–.9 and fee/slippage 0–1000 bps required.')
    returns = data.prices.pct_change(fill_method=None).dropna()
    benchmark = data.benchmark.pct_change(fill_method=None).reindex(returns.index)
    split = int(len(returns) * fraction)
    train, test = returns.iloc[:split], returns.iloc[split:]
    if len(train) < 60 or len(test) < 20:
        raise ValueError('Need at least 60 estimation returns and 20 holdout returns; increase history or reduce train fraction.')
    covariance = str(params.get('covariance', 'shrinkage'))
    weights = optimize_weights(train, annual, rf, cap, covariance)
    weights = {'equal_weight': np.full(test.shape[1], 1 / test.shape[1]), **weights}
    custom = str(params.get('weights', '')).strip()
    if custom:
        try:
            allocation = np.array([float(v.strip()) for v in custom.split(',')])
        except ValueError as exc:
            raise ValueError('Custom weights must be decimal numbers separated by commas.') from exc
        if len(allocation) != test.shape[1] or not np.isfinite(allocation).all() or np.any(allocation < 0) or abs(allocation.sum() - 1) > 1e-6:
            raise ValueError('Custom weights must match ticker order, be nonnegative and sum to 1.')
        weights['custom'] = allocation
    rebalance = str(params.get('rebalance', 'daily'))
    strategy_returns = pd.DataFrame({name: weighted_returns(test, w, rebalance, fee + slippage) for name, w in weights.items()})
    benchmark_test = benchmark.reindex(test.index).rename(data.benchmark_name)
    analysis = analyze(pd.concat([test, strategy_returns], axis=1), benchmark_test, params,
                       str(params.get('title', 'Portfolio comparison')), 'holdout / out of sample')
    report = {**analysis.report, 'assets': list(test.columns), 'benchmark': data.benchmark_name,
              'weights': {name: {ticker: float(w) for ticker, w in zip(test.columns, allocation)} for name, allocation in weights.items()},
              'estimation': {'start': str(train.index[0].date()), 'end': str(train.index[-1].date()), 'observations': len(train),
                             'covariance': covariance, 'long_only': True, 'max_weight': cap},
              'rebalance': rebalance, 'cost_bps_per_traded_notional': fee + slippage, 'alignment': data.alignment,
              'source': data.source,
              'curves': [{'date': str(date.date()), **{name: float(value) for name, value in row.items()}}
                         for date, row in ((1 + strategy_returns).cumprod() - 1).iterrows()]}
    return PortfolioResult(data, strategy_returns, benchmark_test, report)
