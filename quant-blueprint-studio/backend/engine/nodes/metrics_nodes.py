"""Consistent daily-return analytics for assets, strategies and portfolios."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RiskAnalysis:
    returns: pd.DataFrame
    benchmark: pd.Series | None
    report: dict[str, Any]


def settings(params: dict[str, Any]) -> tuple[int, float, float]:
    annual = int(params.get('periods_per_year', 252))
    rf = float(params.get('risk_free_rate', 0))
    confidence = float(params.get('confidence', .95))
    if not 1 <= annual <= 365 or not np.isfinite(rf) or not -.99 < rf <= 1 or not .8 <= confidence < 1:
        raise ValueError('Annual periods 1–365, annual risk-free rate >−.99 and ≤1, confidence .8–<1 required.')
    return annual, rf, confidence


def performance(returns: pd.Series, benchmark: pd.Series | None, params: dict[str, Any]) -> dict[str, Any]:
    annual, rf, confidence = settings(params)
    values = returns.to_numpy(dtype=np.float64)
    if len(values) < 3 or not np.isfinite(values).all() or np.any(values <= -1):
        raise ValueError('Metrics require at least 3 finite simple returns strictly greater than −100%.')
    equity = np.cumprod(1 + values)
    if not np.isfinite(equity).all():
        raise ValueError('Compounded equity overflowed; check return units (decimal, not percent).')
    daily_rf = np.expm1(np.log1p(rf) / annual)
    excess = values - daily_rf
    std = float(np.std(values, ddof=1))
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)))
    losses = -values
    var = float(np.quantile(losses, confidence, method='linear'))
    tail = losses[losses >= var]
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    result: dict[str, Any] = {
        'roi': float(equity[-1] - 1),
        'cagr': float(np.expm1(np.log1p(values).sum() * annual / len(values))),
        'var': var, 'cvar': float(tail.mean()),
        'volatility': std * np.sqrt(annual),
        'sharpe': float(excess.mean() / std * np.sqrt(annual)) if std > 1e-12 else None,
        'sortino': float(excess.mean() / downside * np.sqrt(annual)) if downside > 1e-12 else None,
        'alpha': None, 'alpha_daily': None, 'beta': None, 'ols_r2': None,
        'max_drawdown': float(np.min(equity / peak - 1)), 'observations': len(values),
    }
    if benchmark is not None:
        aligned = pd.concat([returns.rename('asset'), benchmark.rename('benchmark')], axis=1).dropna()
        if len(aligned) >= 3 and float(aligned['benchmark'].std(ddof=1)) > 1e-12:
            fit = sm.OLS(aligned['asset'].to_numpy() - daily_rf,
                         sm.add_constant(aligned['benchmark'].to_numpy() - daily_rf)).fit()
            result.update({'alpha_daily': float(fit.params[0]), 'alpha': float(fit.params[0] * annual),
                           'beta': float(fit.params[1]), 'ols_r2': float(fit.rsquared) if float(aligned['asset'].std(ddof=1)) > 1e-12 and np.isfinite(fit.rsquared) else None})
    notes: dict[str, str] = {}
    if result['sharpe'] is None:
        notes['sharpe'] = 'Retornos constantes: la volatilidad del denominador es cero.'
    if result['sortino'] is None:
        notes['sortino'] = 'No hay desviación negativa respecto a la tasa libre de riesgo.'
    if result['beta'] is None:
        reason = 'No hay benchmark conectado.' if benchmark is None else 'El benchmark no tiene suficientes observaciones comunes o su varianza es cero.'
        notes.update({key: reason for key in ('alpha','alpha_daily','beta','ols_r2')})
    elif result['ols_r2'] is None:
        notes['ols_r2'] = 'La serie evaluada tiene varianza cero; R² no está definido.'
    result['metric_notes'] = notes
    return result


def analyze(returns: pd.DataFrame, benchmark: pd.Series | None, params: dict[str, Any], title: str, scope: str) -> RiskAnalysis:
    annual, rf, confidence = settings(params)
    if returns.empty or not returns.index.is_unique:
        raise ValueError('Returns require unique dates and nonempty observations.')
    aligned = returns.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if benchmark is not None:
        benchmark_name = benchmark.name
        aligned = aligned.join(benchmark.rename('__benchmark__'), how='inner').dropna()
        benchmark = aligned.pop('__benchmark__').rename(benchmark_name)
    if len(aligned) < 3:
        raise ValueError('At least 3 common observations are required for risk analysis.')
    metrics = [{'name': str(name), **performance(aligned[name], benchmark, params)} for name in aligned.columns]
    report = {'title': title, 'scope': scope, 'start': str(aligned.index[0].date()), 'end': str(aligned.index[-1].date()),
              'confidence': confidence, 'var_horizon': '1 trading period', 'risk_free_rate': rf,
              'periods_per_year': annual, 'benchmark': str(benchmark.name) if benchmark is not None else None,
              'metrics': metrics}
    return RiskAnalysis(aligned, benchmark, report)


def frame_returns(frame: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    column = str(params.get('return_column', 'LogReturn'))
    kind = str(params.get('return_type', 'log'))
    if column == 'CumulativeLogReturn':
        raise ValueError('Risk metrics need per-period returns: choose LogReturn, or Close with return type price; cumulative returns are not daily returns.')
    if column == 'LogReturn' and column not in frame and 'Close' in frame:
        values = frame['Close'].pct_change(fill_method=None)
    elif column not in frame:
        raise ValueError(f'Return column {column} is missing.')
    elif kind == 'log':
        values = np.expm1(frame[column])
    elif kind == 'simple':
        values = frame[column]
    elif kind == 'price':
        values = frame[column].pct_change(fill_method=None)
    else:
        raise ValueError('Return type must be log, simple or price.')
    return values.replace([np.inf, -np.inf], np.nan).dropna()
