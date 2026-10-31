"""OHLCV ingestion and strictly trailing feature engineering."""
from typing import Any
import numpy as np
import pandas as pd


def source(params: dict[str, Any]) -> pd.DataFrame:
    years = int(params.get('years', 3))
    if not 1 <= years <= 10:
        raise ValueError('Years must be between 1 and 10.')
    if params.get('source', 'synthetic') not in ('synthetic', 'yfinance'):
        raise ValueError('Source must be synthetic or yfinance.')
    if params.get('source', 'synthetic') == 'synthetic':
        rng = np.random.default_rng(int(params.get('seed', 42)))
        index = pd.bdate_range(end='2025-12-31', periods=252 * years)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(index))))
        opening = close * np.exp(rng.normal(0, 0.003, len(index)))
        return pd.DataFrame({'Open': opening, 'High': np.maximum(opening, close) * 1.005,
                             'Low': np.minimum(opening, close) * 0.995, 'Close': close,
                             'Volume': rng.integers(100000, 5000000, len(index))}, index=index)
    import yfinance as yf
    ticker = str(params.get('ticker', 'SPY')).strip().upper()
    if not ticker or len(ticker) > 20:
        raise ValueError('Supply a single ticker (up to 20 characters).')
    end = pd.Timestamp.now().normalize()
    frame = yf.download(ticker, start=(end - pd.DateOffset(years=years)).date(),
                        end=end.date(), interval='1d', auto_adjust=True, progress=False, timeout=20)
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        raise ValueError(f'No quotes returned for {ticker}. Check symbol/network.')
    return frame[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sort_index()


def features(frame: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    close = result['Close']
    result['LogReturn'] = np.log(close / close.shift(1))
    lags = int(params.get('lags', 5))
    window = int(params.get('rsi_window', 14))
    if not 1 <= lags <= 60 or not 2 <= window <= 100:
        raise ValueError('Lags must be 1–60 and RSI window 2–100.')
    for lag in range(1, lags + 1):
        result[f'Lag_{lag}'] = result['LogReturn'].shift(lag)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/window, adjust=False).mean()
    result['RSI'] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss.ne(0), 100)
    result['EMA'] = close.ewm(span=20, adjust=False).mean()
    result['MACD'] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    result['MACDSignal'] = result['MACD'].ewm(span=9, adjust=False).mean()
    result['Volatility'] = result['LogReturn'].rolling(20).std()
    return result.replace([np.inf, -np.inf], np.nan).dropna()
