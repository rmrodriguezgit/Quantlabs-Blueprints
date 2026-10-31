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
    """Append only enabled indicators. VWAP is trailing N-day typical-price VWAP."""
    result = frame.copy()
    close = result['Close']
    result['LogReturn'] = np.log(close / close.shift(1))
    lags = int(params.get('lags', 5))
    if not 0 <= lags <= 60:
        raise ValueError('Lags must be 0–60.')
    def window(key: str, default: int) -> int:
        value = int(params.get(key, default))
        if not 2 <= value <= 252:
            raise ValueError(f'{key} must be 2–252.')
        return value
    for lag in range(1, lags + 1):
        result[f'Lag_{lag}'] = result['LogReturn'].shift(lag)
    if params.get('use_rsi', True):
        period = window('rsi_window', 14)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
        result['RSI'] = (100 - 100 / (1 + gain / loss.replace(0, np.nan))).where(loss.ne(0), 100).where(gain.ne(0) | loss.ne(0), 50)
    if params.get('use_ema', True):
        result['EMA'] = close.ewm(span=window('ema_window', 20), adjust=False).mean()
    if params.get('use_sma', True):
        result['SMA'] = close.rolling(window('sma_window', 20)).mean()
    if params.get('use_macd', True):
        fast, slow, signal = window('macd_fast', 12), window('macd_slow', 26), window('macd_signal', 9)
        if fast >= slow:
            raise ValueError('MACD fast period must be smaller than slow period.')
        result['MACD'] = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
        result['MACDSignal'] = result['MACD'].ewm(span=signal, adjust=False).mean()
        result['MACDHistogram'] = result['MACD'] - result['MACDSignal']
    if params.get('use_vwap', True):
        if not result['Volume'].fillna(0).gt(0).any():
            raise ValueError('VWAP is unavailable: this asset has no positive reported volume (common for Yahoo FX symbols such as EURUSD=X). Disable Enable VWAP in Technical Features, and deselect VWAP from input/plot columns, or use an asset with reported trading volume. No rows were silently discarded.')
        period = window('vwap_window', 20)
        typical = (result['High'] + result['Low'] + close) / 3
        volume = result['Volume'].rolling(period).sum().replace(0, np.nan)
        result['VWAP'] = (typical * result['Volume']).rolling(period).sum() / volume
    if params.get('use_bb', True):
        period = window('bb_window', 20)
        deviations = float(params.get('bb_std', 2))
        if not 0 < deviations <= 5:
            raise ValueError('Bollinger standard deviations must be >0 and ≤5.')
        middle = close.rolling(period).mean()
        spread = deviations * close.rolling(period).std(ddof=0)
        result['BBMiddle'], result['BBUpper'], result['BBLower'] = middle, middle + spread, middle - spread
    result['Volatility'] = result['LogReturn'].rolling(20).std()
    cleaned = result.replace([np.inf, -np.inf], np.nan).dropna()
    cleaned.attrs['raw_rows'] = frame.attrs.get('raw_rows', len(frame))
    return cleaned


def merge_features(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Join feature branches on dates. Conflicting columns require explicit renaming."""
    common = left.index.intersection(right.index)
    if common.empty:
        raise ValueError('Feature branches have no overlapping dates.')
    duplicate = left.columns.intersection(right.columns)
    for column in duplicate:
        if not np.allclose(left.loc[common, column], right.loc[common, column], equal_nan=True):
            raise ValueError(f'Conflicting column {column}; use distinct indicators/parameters across branches.')
    return left.loc[common].join(right.loc[common].drop(columns=duplicate)).dropna()
