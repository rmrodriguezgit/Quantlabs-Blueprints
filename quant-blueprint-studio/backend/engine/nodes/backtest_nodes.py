"""Forecast at t uses data through t-1; trade that same target period."""
from typing import Any
import numpy as np
from .dl_nodes import ModelResult


def backtest(result: ModelResult, params: dict[str, Any]) -> dict[str, Any]:
    if result.prepared.target != 'LogReturn':
        raise ValueError('Trading requires LogReturn forecasts; Close/Volatility are regression-only.')
    threshold = float(params.get('threshold', .001))
    fee = float(params.get('fee_bps', 2)) / 10000
    slippage = float(params.get('slippage_bps', 1)) / 10000
    if threshold < 0 or not 0 <= fee <= .1 or not 0 <= slippage <= .1:
        raise ValueError('Threshold/costs must be nonnegative; costs at most 1000 bps.')
    signals = result.signals if result.signals is not None else np.where(result.predicted > threshold, 1, np.where(result.predicted < -threshold, -1, 0))
    market = np.expm1(result.prepared.returns)
    turnover = np.abs(np.diff(signals, prepend=0)).astype(float)
    turnover[-1] += abs(signals[-1])  # Liquidate final position.
    returns = signals * market - turnover * (fee + slippage)
    equity = np.cumprod(1 + returns)
    benchmark = np.cumprod(1 + market)
    peak = np.maximum.accumulate(np.r_[1., equity])[1:]
    return {'backtest': [{'date': date, 'strategy': float(eq - 1), 'market': float(bm - 1), 'signal': int(signal), 'strategy_return': float(net), 'market_return': float(raw)}
                         for date, eq, bm, signal, net, raw in zip(result.prepared.dates, equity, benchmark, signals, returns, market)],
            'trading': {'total_return': float(equity[-1] - 1), 'max_drawdown': float(np.min(equity / peak - 1)),
                        'sharpe': float(np.sqrt(252) * returns.mean() / returns.std()) if returns.std() else 0.,
                        'turnover': float(turnover.sum())}}
