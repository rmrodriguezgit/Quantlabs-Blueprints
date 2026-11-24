"""yfinance Data: one ticker with every yfinance.download option exposed and validated.

Verified against yfinance 0.2.66 (``yf.download`` signature and ``history`` docs):
periods 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max; intervals 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo.
Yahoo silently returns 0 rows beyond its intraday windows (measured: 1m ≈ 7 days per request and
only the last 30 days; 2m–90m ≤ 60 days; 60m/1h ≤ 730 days), so the limits are checked before
downloading and an empty answer is reported with the reason instead of failing downstream.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd

PERIODS = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
INTERVALS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
TIMEZONES = ['America/New_York', 'UTC', 'Europe/London', 'America/Mexico_City', 'Asia/Tokyo']
PERIOD_DAYS = {'1d': 1, '5d': 5, '1mo': 31, '3mo': 92, '6mo': 183, '1y': 366, '2y': 731, '5y': 1827, '10y': 3653}
MAX_DAYS = {'1m': 7, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '60m': 730, '90m': 60, '1h': 730}
LOOKBACK_DAYS = {'1m': 30, '2m': 60, '5m': 60, '15m': 60, '30m': 60, '60m': 730, '90m': 60, '1h': 730}
# bars per year for annualising volatility (US regular session: 6.5 h, 252 days)
BARS_PER_YEAR = {'1m': 252 * 390, '2m': 252 * 195, '5m': 252 * 78, '15m': 252 * 26, '30m': 252 * 13, '60m': 252 * 7, '1h': 252 * 7,
                 '90m': 252 * 5, '1d': 252, '5d': 52, '1wk': 52, '1mo': 12, '3mo': 4}
_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
Progress = Callable[[dict[str, Any]], None] | None


def _flag(p: dict[str, Any], key: str, default: bool) -> bool:
    v = p.get(key, default)
    return v if isinstance(v, bool) else str(v).strip().lower() in ('1', 'true', 'sí', 'si', 'yes')


def _span_days(p: dict[str, Any]) -> tuple[float | None, pd.Timestamp | None]:
    """Requested span in days and its start (None = open-ended: ytd/max)."""
    today = pd.Timestamp.now().normalize()
    if p['range_mode'] == 'period':
        if p['period'] == 'ytd':
            start = pd.Timestamp(year=today.year, month=1, day=1)
            return float((today - start).days + 1), start
        if p['period'] == 'max':
            return None, None
        days = PERIOD_DAYS[p['period']]
        return float(days), today - pd.Timedelta(days=days)
    start = pd.Timestamp(p['start'])
    end = pd.Timestamp(p['end']) if p['end'] else today + pd.Timedelta(days=1)
    return float((end - start).days), start


def validate(p: dict[str, Any]) -> dict[str, Any]:
    q = {
        'ticker': str(p.get('ticker', 'SPY')).strip().upper(),
        'range_mode': str(p.get('range_mode', 'period')),
        'period': str(p.get('period', '1y')),
        'start': str(p.get('start', '') or '').strip(),
        'end': str(p.get('end', '') or '').strip(),
        'interval': str(p.get('interval', '1d')),
        'auto_adjust': _flag(p, 'auto_adjust', True),
        'back_adjust': _flag(p, 'back_adjust', False),
        'actions': _flag(p, 'actions', False),
        'prepost': _flag(p, 'prepost', False),
        'repair': _flag(p, 'repair', False),
        'keepna': _flag(p, 'keepna', False),
        'rounding': _flag(p, 'rounding', False),
        'drop_incomplete': _flag(p, 'drop_incomplete', True),
        'timezone': str(p.get('timezone', 'America/New_York')),
        'timeout': float(p.get('timeout', 20)),
        'cache_seconds': float(p.get('cache_seconds', 300)),
    }
    if not q['ticker'] or len(q['ticker']) > 20 or any(c.isspace() or c == ',' for c in q['ticker']):
        raise ValueError('Ticker: un solo símbolo de Yahoo (máx. 20 caracteres), p. ej. SPY, BTC-USD, EURUSD=X, ^GSPC.')
    if q['range_mode'] not in ('period', 'start_end'):
        raise ValueError('Rango: period o start_end.')
    if q['interval'] not in INTERVALS:
        raise ValueError(f'Intervalo no válido. Opciones: {", ".join(INTERVALS)}.')
    if q['range_mode'] == 'period' and q['period'] not in PERIODS:
        raise ValueError(f'Periodo no válido. Opciones: {", ".join(PERIODS)}.')
    if q['range_mode'] == 'start_end':
        try:
            s = pd.Timestamp(q['start'])
            e = pd.Timestamp(q['end']) if q['end'] else None
        except (ValueError, TypeError) as exc:
            raise ValueError('Fechas en formato AAAA-MM-DD (fin opcional; es exclusivo).') from exc
        if not q['start']:
            raise ValueError('Con rango start_end indica la fecha de inicio (AAAA-MM-DD).')
        if e is not None and e <= s:
            raise ValueError('La fecha de fin (exclusiva) debe ser posterior a la de inicio.')
        if s > pd.Timestamp.now():
            raise ValueError('La fecha de inicio está en el futuro.')
    if q['back_adjust'] and q['auto_adjust']:
        raise ValueError('back_adjust sólo aplica con auto_adjust desactivado (ajusta Open/High/Low y deja Close sin ajustar).')
    if q['prepost'] and q['interval'] not in MAX_DAYS:
        raise ValueError('Pre/post mercado sólo existe en intervalos intradía (1m–90m, 1h).')
    if q['timezone'] not in TIMEZONES:
        raise ValueError(f'Zona horaria: {", ".join(TIMEZONES)}.')
    if not (3 <= q['timeout'] <= 120 and 0 <= q['cache_seconds'] <= 86400):
        raise ValueError('Timeout 3–120 s y caché 0–86400 s.')
    # Yahoo's intraday windows (it returns nothing beyond them)
    iv = q['interval']
    if iv in MAX_DAYS:
        span, start = _span_days(q)
        if span is None:
            raise ValueError(f'Con intervalo {iv} no se puede pedir period=max: Yahoo sólo guarda {LOOKBACK_DAYS[iv]} días de datos {iv}.')
        if span > MAX_DAYS[iv] + 1:
            raise ValueError(f'Intervalo {iv}: Yahoo permite como máximo {MAX_DAYS[iv]} días por descarga y pediste ~{span:.0f}. '
                             f'Usa un periodo más corto (p. ej. {"5d" if iv == "1m" else "1mo" if MAX_DAYS[iv] == 60 else "1y"}) o un intervalo mayor.')
        if start is not None and (pd.Timestamp.now() - start).days > LOOKBACK_DAYS[iv] + 1:
            raise ValueError(f'Intervalo {iv}: Yahoo sólo conserva los últimos {LOOKBACK_DAYS[iv]} días; el inicio pedido es anterior.')
    return q


def download(q: dict[str, Any]) -> pd.DataFrame:
    import yfinance as yf
    key = repr(sorted((k, v) for k, v in q.items() if k not in ('cache_seconds', 'drop_incomplete', 'timezone')))
    hit = _CACHE.get(key)
    ttl = q['cache_seconds'] if q['interval'] not in MAX_DAYS else min(q['cache_seconds'], 60)
    if hit and time.time() - hit[0] < ttl:
        return hit[1].copy()
    kwargs: dict[str, Any] = {'interval': q['interval'], 'auto_adjust': q['auto_adjust'], 'back_adjust': q['back_adjust'], 'actions': q['actions'],
                              'prepost': q['prepost'], 'repair': q['repair'], 'keepna': q['keepna'], 'rounding': q['rounding'],
                              'timeout': q['timeout'], 'progress': False, 'threads': False, 'multi_level_index': False}
    if q['range_mode'] == 'period':
        kwargs['period'] = q['period']
    else:
        kwargs['start'] = q['start']
        if q['end']:
            kwargs['end'] = q['end']
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        frame = yf.download(q['ticker'], **kwargs)
    if isinstance(frame.columns, pd.MultiIndex):           # defensive: older/newer yfinance layouts
        frame.columns = frame.columns.get_level_values(0)
    _CACHE[key] = (time.time(), frame.copy())
    return frame


def yfinance_data(p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    q = validate(p)
    started = time.perf_counter()
    if progress: progress({'detail': f'yfinance {q["ticker"]} · {q["interval"]} · ' + (q['period'] if q['range_mode'] == 'period' else f'{q["start"]} → {q["end"] or "hoy"}')})
    raw = download(q)
    if raw is None or raw.empty or 'Close' not in raw or raw['Close'].dropna().empty:
        raise ValueError(f'Yahoo no devolvió datos para {q["ticker"]} ({q["interval"]}, '
                         + (q['period'] if q['range_mode'] == 'period' else f'{q["start"]}–{q["end"] or "hoy"}')
                         + '). Revisa el símbolo (p. ej. BTC-USD, EURUSD=X, ^GSPC), el rango o la conexión.')
    frame = raw.rename(columns={'Adj Close': 'AdjClose', 'Stock Splits': 'StockSplits', 'Capital Gains': 'CapitalGains'}).copy()
    keep = [c for c in ('Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume', 'Dividends', 'StockSplits', 'CapitalGains') if c in frame]
    frame = frame[keep]
    idx = pd.DatetimeIndex(frame.index)
    tz_note = 'fechas diarias sin zona horaria'
    if idx.tz is not None:                                   # intraday comes in UTC: express it in the chosen zone, then drop tz
        idx = idx.tz_convert(q['timezone']).tz_localize(None)
        tz_note = f'horas en {q["timezone"]}'
    frame.index = idx
    frame.index.name = 'Date'
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    if not q['keepna']:
        frame = frame.dropna(subset=['Close'])
    dropped = None
    if q['drop_incomplete'] and len(frame) > 1:
        # the newest bar may still be forming (current minute / today's session): it is not a finished observation
        step = {'1m': '1min', '2m': '2min', '5m': '5min', '15m': '15min', '30m': '30min', '60m': '60min', '1h': '60min', '90m': '90min',
                '1d': '1D', '5d': '5D', '1wk': '7D', '1mo': '31D', '3mo': '92D'}[q['interval']]
        now = pd.Timestamp.now(tz=q['timezone']).tz_localize(None) if q['interval'] in MAX_DAYS else pd.Timestamp.now(tz='America/New_York').tz_localize(None)
        if frame.index[-1] + pd.Timedelta(step) > now:
            dropped = frame.index[-1]
            frame = frame.iloc[:-1]
    frame['Volume'] = frame['Volume'].fillna(0) if 'Volume' in frame else 0.0
    frame.attrs.update({'raw_rows': len(frame), 'ticker': q['ticker'], 'interval': q['interval'], 'timezone': q['timezone']})
    close = frame['Close'].astype(float)
    ret = np.log(close / close.shift(1)).dropna()
    per_year = BARS_PER_YEAR.get(q['interval'], 252)
    eq = close / close.iloc[0]
    dd = float((eq / eq.cummax() - 1).min()) if len(eq) > 1 else 0.0
    gaps = int((frame[['Open', 'High', 'Low', 'Close']].isna().any(axis=1)).sum()) if q['keepna'] else 0
    zero_vol = int((frame['Volume'] == 0).sum()) if 'Volume' in frame else 0
    adj = ('precios ajustados por dividendos y splits (auto_adjust)' if q['auto_adjust'] else
           'Open/High/Low ajustados y Close sin ajustar (back_adjust)' if q['back_adjust'] else 'precios sin ajustar + columna AdjClose')
    span = f'{frame.index[0]:%Y-%m-%d %H:%M} → {frame.index[-1]:%Y-%m-%d %H:%M}' if q['interval'] in MAX_DAYS else f'{frame.index[0]:%Y-%m-%d} → {frame.index[-1]:%Y-%m-%d}'
    notes = [
        f'yfinance.download("{q["ticker"]}", interval="{q["interval"]}", '
        + (f'period="{q["period"]}"' if q['range_mode'] == 'period' else f'start="{q["start"]}"' + (f', end="{q["end"]}"' if q['end'] else ''))
        + f', auto_adjust={q["auto_adjust"]}, back_adjust={q["back_adjust"]}, actions={q["actions"]}, prepost={q["prepost"]}, repair={q["repair"]}, keepna={q["keepna"]}, rounding={q["rounding"]}).',
        f'{len(frame)} barras {q["interval"]} ({span}); {tz_note}; {adj}. {time.perf_counter() - started:.1f}s.',
        'La fecha de fin es exclusiva en yfinance. El intradía sólo cubre la sesión regular salvo con prepost.' if q['interval'] in MAX_DAYS else
        'La fecha de fin es exclusiva en yfinance.',
    ]
    if dropped is not None:
        notes.append(f'Se descartó la última barra ({dropped}) porque aún no había cerrado (drop_incomplete).')
    if zero_vol and q['prepost'] and zero_vol < len(frame):
        notes.append(f'{zero_vol} barras con volumen 0: Yahoo no reporta volumen en pre/post mercado; las métricas de volumen sólo valen en sesión regular.')
    elif zero_vol and zero_vol > .5 * len(frame):
        notes.append(f'{zero_vol} barras con volumen 0: típico de divisas (=X) e índices (^); VWAP y métricas de volumen no tienen sentido aquí.')
    if q['actions'] and 'Dividends' in frame:
        notes.append(f'Acciones corporativas: {int((frame["Dividends"] > 0).sum())} dividendos, {int((frame.get("StockSplits", pd.Series(dtype=float)) > 0).sum())} splits en el rango.')
    rows = frame.reset_index()
    chart_rows = [{'fecha': d.strftime('%Y-%m-%d %H:%M') if q['interval'] in MAX_DAYS else d.strftime('%Y-%m-%d'), 'Close': round(float(c), 6),
                   'Volume': float(v)} for d, c, v in zip(rows['Date'], rows['Close'], rows['Volume'])][-1500:]
    report = {
        'topic': 'yfinance_data', 'selected': f'{q["ticker"]} · {q["interval"]} · ' + (q['period'] if q['range_mode'] == 'period' else f'{q["start"]} → {q["end"] or "hoy"}'),
        'features': list(frame.columns), 'splits': {'train': 0, 'validation': 0, 'test': len(frame)},
        'notes': notes,
        'table': [{'fecha': str(d), **{k: (None if pd.isna(v) else round(float(v), 6)) for k, v in r.items()}} for d, r in frame.tail(12).iloc[::-1].iterrows()],
        'metrics': {'barras': len(frame), 'retorno_total': float(close.iloc[-1] / close.iloc[0] - 1), 'volatilidad_anual': float(ret.std() * np.sqrt(per_year)) if len(ret) > 2 else None,
                    'max_drawdown': dd, 'ultimo_close': float(close.iloc[-1]), 'maximo': float(frame['High'].max()), 'minimo': float(frame['Low'].min()),
                    'barras_volumen_cero': zero_vol, 'barras_incompletas': gaps},
        'charts': [{'title': f'{q["ticker"]} · Close ({q["interval"]})', 'rows': chart_rows, 'series': ['Close'], 'x_key': 'fecha', 'chart_type': 'line'},
                   {'title': f'{q["ticker"]} · Volumen', 'rows': chart_rows, 'series': ['Volume'], 'x_key': 'fecha', 'chart_type': 'bar'}],
    }
    return frame, report
