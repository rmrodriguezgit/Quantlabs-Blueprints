"""Polymarket data source and a Laya-weighted UP/DOWN decision.

Endpoints used (verified against live responses, 2026-11):

* Gamma  ``GET https://gamma-api.polymarket.com/markets?slug=|condition_ids=|clob_token_ids=``
         ``GET https://gamma-api.polymarket.com/events?slug=``
         ``GET https://gamma-api.polymarket.com/public-search?q=&limit_per_type=``
  Markets carry ``outcomes``/``outcomePrices``/``clobTokenIds`` as JSON-encoded strings,
  ``eventStartTime``/``endDate`` (ISO) and tags such as ``5M``/``15M``/``1H``.
* CLOB   ``GET https://clob.polymarket.com/prices-history?market=<token>&interval=&fidelity=``
         → ``{"history": [{"t": unix, "p": float}]}``
         ``GET https://clob.polymarket.com/book?token_id=`` → bids/asks ``[{price, size}]`` (strings)

Recurring crypto markets use slugs ``<asset>-updown-<5m|15m|1h>-<unix start>``.
Network calls are read-only, time-limited, cached and never block the rest of the graph
longer than ``timeout`` seconds per request.
"""
from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
WINDOWS = {'5m': 300, '15m': 900, '30m': 1800, '1h': 3600}
UP_WORDS = ('up', 'yes', 'higher', 'above', 'over')
DOWN_WORDS = ('down', 'no', 'lower', 'below', 'under')
Progress = Callable[[dict[str, Any]], None] | None

_CACHE: dict[str, tuple[float, Any]] = {}
_LOCK = threading.Lock()


# ---------------- HTTP with cache, retries and timeouts ----------------

def _get(url: str, params: dict[str, Any] | None = None, *, ttl: float = 15, timeout: float = 8, retries: int = 2) -> Any:
    full = url + ('?' + urllib.parse.urlencode(params, doseq=True) if params else '')
    now = time.time()
    with _LOCK:
        hit = _CACHE.get(full)
        if hit and now - hit[0] < ttl:
            return hit[1]
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(full, headers={'User-Agent': 'quant-blueprint-studio', 'Accept': 'application/json'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                value = json.loads(resp.read().decode('utf-8'), strict=False)
            with _LOCK:
                _CACHE[full] = (time.time(), value)
                if len(_CACHE) > 500:
                    for key in sorted(_CACHE, key=lambda k: _CACHE[k][0])[:100]:
                        _CACHE.pop(key, None)
            return value
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code < 500 and exc.code != 429:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
        time.sleep(0.4 * (attempt + 1))
    if hit:  # stale but usable: keep the graph running
        return hit[1]
    raise ValueError(f'Polymarket no responde ({type(last).__name__}: {last}). Revisa la conexión o reintenta.')


def _jsonlist(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith('['):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


# ---------------- market resolution ----------------

def _parse_ref(ref: str) -> tuple[str, str]:
    ref = ref.strip()
    if not ref:
        raise ValueError('Indica un mercado de Polymarket (slug, URL, condition_id, token_id o texto).')
    if ref.startswith('http'):
        path = [p for p in urllib.parse.urlparse(ref).path.split('/') if p]
        if not path:
            raise ValueError('URL de Polymarket sin slug.')
        # /event/<event-slug>[/<market-slug>] or /market/<slug>
        return 'slug', path[-1]
    if re.fullmatch(r'0x[0-9a-fA-F]{64}', ref):
        return 'condition_id', ref.lower()
    if re.fullmatch(r'\d{30,}', ref):
        return 'token_id', ref
    if re.fullmatch(r'[a-z0-9]+(-[a-z0-9]+)+', ref):
        return 'slug', ref
    return 'text', ref


def _resolve_updown_alias(slug: str, ttl: float) -> str:
    """``btc-updown-15m`` (no timestamp) → the market whose window contains now."""
    m = re.fullmatch(r'([a-z]+)-updown-(5m|15m|30m|1h)', slug)
    if not m:
        return slug
    step = WINDOWS[m.group(2)]
    now = int(time.time())
    for start in (now - now % step, now - now % step - step):
        candidate = f'{slug}-{start}'
        if _get(f'{GAMMA}/events', {'slug': candidate}, ttl=ttl):
            return candidate
    raise ValueError(f'No hay mercado activo para {slug}; Polymarket no publica esa ventana para ese activo.')


def _markets(params: dict[str, Any], ttl: float, timeout: float) -> list[dict[str, Any]]:
    """Gamma hides closed markets unless asked: try open first, then closed=true."""
    return _get(f'{GAMMA}/markets', params, ttl=ttl, timeout=timeout) or _get(f'{GAMMA}/markets', {**params, 'closed': 'true'}, ttl=ttl, timeout=timeout)


def find_market(ref: str, ttl: float = 30, timeout: float = 8) -> dict[str, Any]:
    kind, value = _parse_ref(ref)
    if kind == 'slug':
        value = _resolve_updown_alias(value, ttl)
        found = _markets({'slug': value}, ttl=ttl, timeout=timeout)
        if not found:  # event slug: take its first open market
            events = _get(f'{GAMMA}/events', {'slug': value}, ttl=ttl, timeout=timeout)
            markets = events[0].get('markets', []) if events else []
            if markets:
                open_ = [mk for mk in markets if not mk.get('closed')] or markets
                found = _markets({'slug': open_[0]['slug']}, ttl=ttl, timeout=timeout)
    elif kind == 'condition_id':
        found = _markets({'condition_ids': value}, ttl=ttl, timeout=timeout)
    elif kind == 'token_id':
        found = _markets({'clob_token_ids': value}, ttl=ttl, timeout=timeout)
    else:
        hits = search_markets(value, 1, timeout=timeout)
        found = _markets({'slug': hits[0]['slug']}, ttl=ttl, timeout=timeout) if hits else []
    if not found:
        raise ValueError(f'Mercado no encontrado en Polymarket: {ref!r}.')
    return found[0]


def search_markets(query: str, limit: int = 10, timeout: float = 8) -> list[dict[str, Any]]:
    data = _get(f'{GAMMA}/public-search', {'q': query, 'limit_per_type': max(1, min(limit, 20))}, ttl=30, timeout=timeout)
    out: list[dict[str, Any]] = []
    for event in data.get('events', []) if isinstance(data, dict) else []:
        for mk in event.get('markets', [])[:3]:
            out.append({'slug': mk.get('slug'), 'question': mk.get('question') or event.get('title'),
                        'closed': bool(mk.get('closed')), 'end': mk.get('endDate'),
                        'outcomes': _jsonlist(mk.get('outcomes')), 'window': detect_window(mk, event)})
    out.sort(key=lambda r: r['closed'])
    return out[:limit]


def detect_window(market: dict[str, Any], event: dict[str, Any] | None = None) -> str | None:
    slug = str(market.get('slug', ''))
    m = re.search(r'-updown-(5m|15m|30m|1h)-', slug)
    if m:
        return m.group(1)
    tags = {str(t.get('slug', t.get('label', ''))).lower() for t in (event or {}).get('tags', []) + market.get('tags', []) if isinstance(t, dict)}
    for w in WINDOWS:
        if w in tags:
            return w
    start, end = market.get('eventStartTime'), market.get('endDate')
    if start and end:
        secs = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds()
        for w, s in WINDOWS.items():
            if abs(secs - s) < 1:
                return w
    return None


def map_outcomes(outcomes: list[str], up_index: str = 'auto') -> tuple[int, int]:
    """Canonical UP/DOWN indices. YES/Up/Higher → UP; NO/Down/Lower → DOWN."""
    if len(outcomes) != 2:
        raise ValueError(f'Se requieren exactamente 2 outcomes para UP/DOWN; el mercado tiene {len(outcomes)}.')
    if up_index in ('0', '1'):
        u = int(up_index)
        return u, 1 - u
    low = [str(o).strip().lower() for o in outcomes]
    for i, name in enumerate(low):
        if name in UP_WORDS or any(name.startswith(w) for w in UP_WORDS):
            return i, 1 - i
        if name in DOWN_WORDS:
            return 1 - i, i
    return 0, 1  # e.g. "Team A" vs "Team B": first outcome = UP unless overridden


def market_meta(ref: str, up_index: str = 'auto', timeout: float = 8) -> dict[str, Any]:
    mk = find_market(ref, timeout=timeout)
    outcomes = [str(o) for o in _jsonlist(mk.get('outcomes'))]
    tokens = [str(t) for t in _jsonlist(mk.get('clobTokenIds'))]
    prices = [float(p) for p in _jsonlist(mk.get('outcomePrices'))]
    u, d = map_outcomes(outcomes, up_index)
    events = mk.get('events') or []
    return {
        'slug': mk.get('slug'), 'question': mk.get('question'), 'condition_id': mk.get('conditionId'),
        'event_slug': events[0].get('slug') if events else None, 'event_title': events[0].get('title') if events else None,
        'tokens': tokens, 'outcome_prices': prices, 'spread_gamma': _f(mk.get('spread')), 'change_1h': _f(mk.get('oneHourPriceChange')),
        'change_24h': _f(mk.get('oneDayPriceChange')), 'change_7d': _f(mk.get('oneWeekPriceChange')), 'description': (mk.get('description') or '')[:1200],
        'resolution_source': mk.get('resolutionSource') or (events[0].get('resolutionSource') if events else None),
        'outcomes': outcomes, 'up_outcome': outcomes[u], 'down_outcome': outcomes[d],
        'up_token': tokens[u] if len(tokens) == 2 else None, 'down_token': tokens[d] if len(tokens) == 2 else None,
        'price_up': prices[u] if len(prices) == 2 else None, 'price_down': prices[d] if len(prices) == 2 else None,
        'window': detect_window(mk, events[0] if events else None), 'start': mk.get('eventStartTime') or mk.get('startDate'),
        'end': mk.get('endDate'), 'closed': bool(mk.get('closed')), 'active': bool(mk.get('active')),
        'accepting_orders': bool(mk.get('acceptingOrders')), 'volume': float(mk.get('volumeNum') or 0),
        'liquidity': float(mk.get('liquidityNum') or 0), 'volume24h': float(mk.get('volume24hr') or 0),
    }


# ---------------- underlying crypto price (resolution source: Chainlink BTC/USD etc.) ----------------

PRICE_VARIANTS = {'5m': 'fiveminute', '15m': 'fifteen', '1h': 'onehour'}   # verified; others return a generic placeholder
RTDS = 'wss://ws-live-data.polymarket.com'


def _iso(ts: Any) -> str:
    return pd.Timestamp(ts).tz_convert('UTC').strftime('%Y-%m-%dT%H:%M:%SZ')


def price_to_beat(asset: str, window: str, start: str, end: str, timeout: float = 8) -> dict[str, Any] | None:
    """Opening price of the window as shown by Polymarket (Chainlink), from polymarket.com/api/crypto/crypto-price."""
    variant = PRICE_VARIANTS.get(window)
    if not variant:
        return None
    try:
        data = _get('https://polymarket.com/api/crypto/crypto-price',
                    {'symbol': asset.upper(), 'eventStartTime': _iso(start), 'variant': variant, 'endDate': _iso(end)}, ttl=30, timeout=timeout, retries=1)
    except ValueError:
        return None
    if not isinstance(data, dict) or data.get('openPrice') is None:
        return None
    return {'open': float(data['openPrice']), 'close': None if data.get('closePrice') is None else float(data['closePrice']), 'completed': bool(data.get('completed'))}


def chainlink_spot(asset: str, timeout: float = 6) -> list[dict[str, float]]:
    """Recent per-second Chainlink prices from Polymarket's real-time data socket (first snapshot only)."""
    key = f'rtds:{asset}'
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and time.time() - hit[0] < 3:
            return hit[1]
    try:
        import asyncio
        import websockets

        async def snap() -> list[dict[str, float]]:
            async with websockets.connect(RTDS, open_timeout=timeout) as ws:
                await ws.send(json.dumps({'action': 'subscribe', 'subscriptions': [{'topic': 'crypto_prices_chainlink', 'type': '*', 'filters': json.dumps({'symbol': f'{asset.lower()}/usd'})}]}))
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout)
                    try:
                        msg = json.loads(raw)
                    except (TypeError, json.JSONDecodeError):
                        continue   # the socket first sends an empty acknowledgement
                    if not isinstance(msg, dict):
                        continue
                    pay = msg.get('payload') or {}
                    rows = pay.get('data') or ([{'timestamp': pay['timestamp'], 'value': pay['value']}] if 'value' in pay else [])
                    if rows:
                        return [{'t': float(r['timestamp']) / 1000, 'p': float(r['value'])} for r in rows]
        rows = asyncio.run(snap())
    except Exception:   # network, timeout, schema: the price panel is optional
        return hit[1] if hit else []
    with _LOCK:
        _CACHE[key] = (time.time(), rows)
    return rows


BINANCE = 'https://api.binance.com/api/v3/klines'
BINANCE_SYMBOLS = {'btc': 'BTCUSDT', 'eth': 'ETHUSDT', 'sol': 'SOLUSDT', 'xrp': 'XRPUSDT', 'doge': 'DOGEUSDT', 'bnb': 'BNBUSDT'}


def _klines(asset: str, t0: int, t1: int, historical: bool, timeout: float) -> list[tuple[int, float, float]] | None:
    symbol = BINANCE_SYMBOLS.get(asset.lower())
    if not symbol:
        return None
    try:
        rows = _get(BINANCE, {'symbol': symbol, 'interval': '1m', 'startTime': int(t0) * 1000, 'endTime': int(t1) * 1000, 'limit': 1000},
                    ttl=86400 if historical else 5, timeout=timeout, retries=1)
    except ValueError:
        return None
    if not isinstance(rows, list):
        return None
    return [(int(r[0]) // 1000, float(r[1]), float(r[4])) for r in rows]   # open_time, open, close


def target_from_klines(k: list[Any], start: int, t_dec: float, end: int, historical: bool, vol_minutes: int = 30, source: str = 'Binance 1m') -> dict[str, Any] | None:
    """Distance to the price to beat as a volatility-scaled z-score, known at t_dec.

    z = Δ% since the window open / (σ of 1-minute returns · √minutes left). Under a random walk
    P(UP) ≈ Φ(z). Binance 1m klines are used both live and for history, so calibration and live
    decisions see the same quantity; the Chainlink level differs by a few dollars but its in-window
    move tracks Binance closely (Chainlink is what the dashboard shows and what resolves the market).
    """
    if not k or len(k) < 10 or t_dec < start:
        return None
    k = [tuple(x) for x in k]
    opens = [o for ts, o, _c in k if ts == start]
    if not opens:
        return None
    # price known at t_dec: close of the last kline finished by t_dec (live: the current partial one)
    done = [c for ts, _o, c in k if ts + 60 <= t_dec] if historical else [k[-1][2]]
    if not done:
        return None
    p_open, p_now = opens[0], done[-1]
    closes = np.array([c for ts, _o, c in k if ts < start][-vol_minutes:])
    sigma = float(np.std(np.diff(np.log(closes))) * 100) if len(closes) > 5 else 0.0
    sigma = max(sigma, 0.005)                      # floor: 0.005 % per minute
    minutes_left = max((end - t_dec) / 60, 0.25)
    diff_pct = (p_now / p_open - 1) * 100
    z = float(np.clip(diff_pct / (sigma * np.sqrt(minutes_left)), -6, 6))
    return {'z': z, 'diff_pct': diff_pct, 'sigma_pct_min': sigma, 'minutes_left': minutes_left, 'source': source}


def target_signal(asset: str, start: int, t_dec: float, end: int, vol_minutes: int = 30, timeout: float = 8, historical: bool = False) -> dict[str, Any] | None:
    if t_dec < start or asset.lower() not in BINANCE_SYMBOLS:
        return None
    k = _klines(asset, int(start - vol_minutes * 60), int(t_dec), historical, timeout)
    return target_from_klines(k or [], start, t_dec, end, historical, vol_minutes, f'Binance {BINANCE_SYMBOLS[asset.lower()]} 1m')


def underlying(meta: dict[str, Any], window: str, timeout: float) -> dict[str, Any] | None:
    m = re.match(r'([a-z]+)-updown-', str(meta.get('slug', '')))
    if not m or not meta.get('start') or not meta.get('end'):
        return None
    asset = m.group(1)
    target = price_to_beat(asset, window, meta['start'], meta['end'], timeout)
    spot_rows = chainlink_spot(asset)
    spot = spot_rows[-1]['p'] if spot_rows else None
    if target is None and spot is None:
        return None
    open_ = target['open'] if target else None
    diff = None if spot is None or open_ is None else spot - open_
    start_ts, end_ts = int(pd.Timestamp(meta['start']).timestamp()), int(pd.Timestamp(meta['end']).timestamp())
    signal = target_signal(asset, start_ts, time.time(), end_ts, timeout=timeout)
    return {'asset': asset.upper(), 'source': 'Chainlink (Polymarket)', 'price_to_beat': open_, 'spot': spot, 'signal': signal,
            'spot_ts': spot_rows[-1]['t'] if spot_rows else None, 'diff': diff, 'diff_pct': None if diff is None else diff / open_ * 100,
            'leading': None if diff is None else ('UP' if diff >= 0 else 'DOWN'), 'close': target['close'] if target else None,
            'completed': target['completed'] if target else False, 'history': spot_rows[-180:]}


def _mid_or(book: dict[str, Any], fallback: float) -> float:
    """Same rule as the decision: book mid when the spread is ≤ 0.10, else the last history point."""
    bb, ba = book.get('best_bid'), book.get('best_ask')
    if bb is not None and ba is not None and np.isfinite(bb) and np.isfinite(ba) and ba - bb <= .1:
        return float((bb + ba) / 2)
    return fallback


def _book(token: str, timeout: float, ttl: float) -> dict[str, float]:
    book = _get(f'{CLOB}/book', {'token_id': token}, ttl=ttl, timeout=timeout)
    bids = sorted(((float(b['price']), float(b['size'])) for b in book.get('bids', [])), reverse=True)
    asks = sorted((float(a['price']), float(a['size'])) for a in book.get('asks', []))
    bb, ba = (bids[0][0] if bids else np.nan), (asks[0][0] if asks else np.nan)
    near = lambda side, ref, sign: sum(s for p, s in side if abs(p - ref) <= .05) if np.isfinite(ref) else 0.0
    bd, ad = near(bids, bb, 1), near(asks, ba, -1)
    return {'best_bid': bb, 'best_ask': ba, 'spread': ba - bb if bids and asks else np.nan,
            'bid_depth': bd, 'ask_depth': ad, 'imbalance': (bd - ad) / (bd + ad) if bd + ad > 0 else 0.0}


def polymarket_source(p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    timeout = float(p.get('timeout', 8))
    ttl = float(p.get('cache_seconds', 15))
    interval = str(p.get('history_interval', '1d'))
    fidelity = int(p.get('fidelity', 1))
    window_choice = str(p.get('window', 'auto'))
    if interval not in ('1h', '6h', '1d', '1w', 'max') or not 1 <= fidelity <= 1440 or not 1 <= timeout <= 30 or not 0 <= ttl <= 3600:
        raise ValueError('Historial 1h/6h/1d/1w/max, fidelidad 1–1440 min, timeout 1–30 s, caché 0–3600 s.')
    if window_choice not in ('auto', *WINDOWS):
        raise ValueError('Ventana: auto, 5m, 15m, 30m o 1h.')
    if progress: progress({'detail': 'Resolviendo mercado en Gamma API'})
    meta = market_meta(str(p.get('market', '')), str(p.get('up_outcome', 'auto')), timeout)
    if not meta['up_token']:
        raise ValueError('El mercado no tiene tokens CLOB (orderbook desactivado).')
    window = meta['window'] if window_choice == 'auto' else window_choice
    if window is None:
        window = str(p.get('fallback_window', '1h'))
        window_note = f'Ventana no detectable en el mercado; se usa {window} (cámbiala en el Inspector).'
    else:
        window_note = f'Ventana {window} ' + ('detectada del mercado.' if window_choice == 'auto' else 'fijada manualmente.')
    if progress: progress({'detail': 'Descargando historial CLOB'})
    hist = _get(f'{CLOB}/prices-history', {'market': meta['up_token'], 'interval': interval, 'fidelity': fidelity}, ttl=ttl, timeout=timeout)
    points = hist.get('history', []) if isinstance(hist, dict) else []
    if len(points) < 3:
        raise ValueError('Historial CLOB vacío o muy corto; prueba interval=max o un mercado con más actividad.')
    s = pd.Series({pd.Timestamp(pt['t'], unit='s', tz='UTC'): float(pt['p']) for pt in points}).sort_index()
    s = s[~s.index.duplicated(keep='last')]
    frame = pd.DataFrame({'P_up': s.clip(0, 1)})
    frame.index = frame.index.tz_convert(None)
    frame.index.name = 'Date'
    frame['P_down'] = 1 - frame['P_up']
    frame['Close'] = frame['P_up']                         # lets generic nodes (Laya Signal, Plot…) consume it
    frame['Open'] = frame['Close'].shift(1).fillna(frame['Close'])
    frame['High'] = frame[['Open', 'Close']].max(axis=1)
    frame['Low'] = frame[['Open', 'Close']].min(axis=1)
    frame['Volume'] = 0.0
    try:
        book = _book(meta['up_token'], timeout, min(ttl, 10))
    except ValueError as exc:
        book = {'best_bid': np.nan, 'best_ask': np.nan, 'spread': np.nan, 'bid_depth': 0.0, 'ask_depth': 0.0, 'imbalance': 0.0}
        window_note += f' Orderbook no disponible ({exc}).'
    for k, v in book.items():
        frame['Book_' + k] = np.nan
        frame.iloc[-1, frame.columns.get_loc('Book_' + k)] = v   # snapshot: known only at the last row
    if progress: progress({'detail': 'Precio subyacente (Chainlink) y precio a vencer'})
    under = underlying(meta, window, timeout)
    end = pd.Timestamp(meta['end']).tz_convert(None) if meta['end'] else None
    frame['SecondsToClose'] = np.clip((end - frame.index).total_seconds().to_numpy(), 0, None) if end is not None else np.nan
    frame.attrs.update({'raw_rows': len(frame), 'polymarket': {**meta, 'window': window, 'window_seconds': WINDOWS[window], 'book': book, 'underlying': under},
                        'laya_ticker': meta['slug']})
    last = frame.iloc[-1]
    report = {
        'topic': 'polymarket_source', 'selected': meta['slug'], 'features': ['P_up', 'P_down', 'Close', 'SecondsToClose'] + ['Book_' + k for k in book],
        'splits': {'train': 0, 'validation': 0, 'test': len(frame)},
        'notes': [
            f'Mercado: {meta["question"]}  ·  UP = «{meta["up_outcome"]}», DOWN = «{meta["down_outcome"]}».',
            window_note,
            f'Historial CLOB prices-history interval={interval} fidelity={fidelity} min sobre el token UP. Close = P(UP) implícita.',
            'Orderbook (spread, profundidad ±5¢, desequilibrio) es un snapshot actual: sólo se escribe en la última fila para no mirar al futuro.',
            f'Estado: {"cerrado" if meta["closed"] else "abierto"}; acepta órdenes: {meta["accepting_orders"]}. Caché {ttl:.0f}s, timeout {timeout:.0f}s.',
        ],
        'table': [{'campo': k, 'valor': (round(v, 4) if isinstance(v, float) else v)} for k, v in
                  [('slug', meta['slug']), ('condition_id', meta['condition_id']), ('ventana', window), ('inicio', meta['start']), ('cierre', meta['end']),
                   ('P_up (Gamma)', meta['price_up']), ('P_up (último CLOB)', float(last['P_up'])), ('spread', book['spread']),
                   ('desequilibrio libro', book['imbalance']),
                   (f'{under["asset"]} actual (Chainlink)' if under else 'subyacente', under and under['spot']), ('precio a vencer', under and under['price_to_beat']), ('volumen', meta['volume']), ('liquidez', meta['liquidity'])]],
        'metrics': {'filas': len(frame), 'p_up': _mid_or(book, float(last['P_up'])), 'p_up_historial': float(last['P_up']), 'spread': None if not np.isfinite(book['spread']) else float(book['spread']),
                    'imbalance': float(book['imbalance']), 'liquidez': meta['liquidity'], 'volumen_24h': meta['volume24h'],
                    'precio_a_vencer': under and under['price_to_beat'], 'subyacente': under and under['spot'],
                    'diferencia_pct': under and under['diff_pct'], 'z_objetivo': under and (under.get('signal') or {}).get('z')},
        'market': {**meta, 'window': window, 'book': {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in book.items()},
                   'seconds_to_close': None if end is None else max(0.0, (end - pd.Timestamp.utcnow().tz_convert(None)).total_seconds()),
                   'last_history': str(frame.index[-1]), 'underlying': under, 'fetched_at': pd.Timestamp.utcnow().isoformat()},
        'charts': [{'title': f'P(UP) implícita · {meta["slug"]}', 'rows': [{'date': d.strftime('%Y-%m-%d %H:%M'), 'P_up': float(v), 'P_down': float(1 - v)} for d, v in frame['P_up'].items()][-1500:],
                    'series': ['P_up', 'P_down'], 'x_key': 'date', 'chart_type': 'line'}],
    }
    return frame, report


# ---------------- shared signal features (live and historical use the same code) ----------------

def market_signals(prices: pd.Series, p_mkt: float, imbalance: float, lookback: int, target: dict[str, Any] | None = None) -> dict[str, float]:
    """z_market, z_momentum, z_imbalance, z_target from data known at decision time (z_laya added separately)."""
    look = prices.astype(float).iloc[-lookback:]
    mom = float(look.iloc[-1] - look.iloc[0]) if len(look) > 1 else 0.0
    vol = float(look.diff().std() or 0) * np.sqrt(max(len(look), 1)) + 1e-6
    imb = float(imbalance) if np.isfinite(imbalance) else 0.0
    return {'market': float(_logit(p_mkt)), 'momentum': float(np.tanh(mom / vol) * 2), 'laya': 0.0, 'imbalance': imb * 2,
            'target': float(target['z']) if target else 0.0, '_momentum_raw': mom, '_target_pct': target['diff_pct'] if target else None}


def laya_state(question: str, window: str, p_mkt: float, z: dict[str, float], spread: float | None, minutes_to_close: float | None) -> dict[str, Any]:
    return {'market': question, 'horizon': window, 'implied_probability_up': round(p_mkt, 4),
            'momentum_over_lookback': round(z['_momentum_raw'], 4), 'normalized_momentum': round(z['momentum'] / 2, 3),
            'orderbook_imbalance': round(z['imbalance'] / 2, 3), 'spread': None if spread is None or not np.isfinite(spread) else round(float(spread), 4),
            'minutes_to_close': None if minutes_to_close is None or not np.isfinite(minutes_to_close) else round(float(minutes_to_close), 1),
            'change_since_window_open_pct': None if z.get('_target_pct') is None else round(float(z['_target_pct']), 4),
            'distance_to_price_to_beat_z': round(z['target'], 3)}


def laya_question(window: str, text: str | None = None) -> dict[str, Any]:
    return {'direction': {'type': 'choice', 'instructions': text or f'Given this prediction-market state, will the outcome resolve UP or DOWN at the end of the {window} window?',
                          'criteria': {'up': f'the asset finishes the {window} window higher (UP resolves)', 'down': f'the asset finishes the {window} window lower (DOWN resolves)'}}}


def bucket_seconds(window: str, setting: Any = 'auto') -> int:
    """Minute buckets inside a window: 60 s for 5m/15m, 300 s for 30m/1h; '0' = a single decision point."""
    if setting in (None, '', 'auto'):
        return 60 if WINDOWS[window] <= 900 else 300
    value = int(setting)
    if value and (value < 30 or WINDOWS[window] % value):
        raise ValueError(f'Tramo {value}s no divide la ventana {window}.')
    return value


def decision_offsets(window: str, bucket_s: int, single_offset: int) -> list[int]:
    """Decision point of each bucket = its midpoint (live decisions fall anywhere inside it)."""
    step = WINDOWS[window]
    if bucket_s <= 0:
        return [single_offset]
    return [b * bucket_s + bucket_s // 2 for b in range(step // bucket_s)]


def bucket_of(elapsed: float, bucket_s: int, step: int) -> int | None:
    if bucket_s <= 0:
        return None
    return int(np.clip(elapsed // bucket_s, 0, step // bucket_s - 1))


def _window_raw(slug: str, start: int, step: int, lookback: int, up_index: str, timeout: float) -> dict[str, Any] | None:
    """Everything needed to rebuild any decision point of one settled window (kept on disk: it never changes)."""
    cache = _learning_dir() / 'windows' / f'v3_{slug}_{lookback}_{up_index}.json'
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    try:
        mk = find_market(slug, ttl=60, timeout=timeout)
    except ValueError:
        return None
    prices = [float(x) for x in _jsonlist(mk.get('outcomePrices'))]
    if not mk.get('closed') or len(prices) != 2 or max(prices) < .99:
        return None
    outcomes, tokens = [str(o) for o in _jsonlist(mk.get('outcomes'))], [str(t) for t in _jsonlist(mk.get('clobTokenIds'))]
    u, _d = map_outcomes(outcomes, up_index)
    hist = _get(f'{CLOB}/prices-history', {'market': tokens[u], 'startTs': start - lookback * 60 - 60, 'endTs': start + step, 'fidelity': 1}, ttl=86400, timeout=timeout)
    pts = [[int(pt['t']), float(pt['p'])] for pt in (hist.get('history', []) if isinstance(hist, dict) else [])]
    if not pts:
        return None
    k = _klines(slug.split('-updown-')[0], start - 30 * 60, start + step, True, timeout) or []
    raw = {'slug': slug, 'start': start, 'step': step, 'question': mk.get('question'), 'y': 1.0 if prices[u] > .5 else 0.0,
           'prices': pts, 'klines': [list(x) for x in k]}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(raw))
    return raw


def window_at(raw: dict[str, Any], offset: int, lookback: int) -> dict[str, Any] | None:
    """State of a settled window as known at start+offset (nothing later is used)."""
    start, step = raw['start'], raw['step']
    t_dec = start + offset
    pts = [p for t, p in raw['prices'] if t_dec - lookback * 60 - 60 <= t <= t_dec]
    if not pts:
        return None
    k = [x for x in raw['klines'] if x[0] < t_dec]
    target = target_from_klines(k, start, t_dec, start + step, True)
    return {'slug': raw['slug'], 'start': start, 'offset': offset, 'decision_ts': t_dec, 'question': raw['question'], 'prices': pts,
            'p_market': pts[-1], 'y': raw['y'], 'minutes_to_close': (step - offset) / 60, 'target': target}


def resolved_windows(series: str, count: int, offsets: list[int] | int, lookback: int, up_index: str = 'auto', timeout: float = 8, progress: Progress = None) -> list[dict[str, Any]]:
    """Last `count` resolved windows of a recurring series, one record per window and decision offset."""
    m = re.fullmatch(r'([a-z]+-updown-(5m|15m|30m|1h))', series)
    if not m:
        raise ValueError(f'La calibración por ventanas requiere una serie recurrente <activo>-updown-<5m|15m|1h>; {series!r} no lo es.')
    step = WINDOWS[m.group(2)]
    offsets = [offsets] if isinstance(offsets, int) else list(offsets)
    if not offsets or any(not 0 <= o < step for o in offsets):
        raise ValueError(f'Desfase de decisión 0–{step - 1} s para ventanas de {m.group(2)}.')
    from concurrent.futures import ThreadPoolExecutor
    now = int(time.time())
    starts = [now - now % step - k * step for k in range(1, count + 25)]   # extra: the newest ones may not be settled yet
    raws: list[dict[str, Any]] = []
    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(_window_raw, f'{series}-{st}', st, step, lookback, up_index, timeout) for st in starts]
        for i, fut in enumerate(futures):
            raw = fut.result()
            if raw:
                raws.append(raw)
            if progress and i % 20 == 0:
                progress({'detail': f'Ventanas resueltas {len(raws)}/{count} (revisadas {i + 1})'})
            if len(raws) >= count:
                break
    if len(raws) < 20:
        raise ValueError(f'Sólo {len(raws)} ventanas resueltas de {series}; se necesitan ≥ 20.')
    out = []
    for raw in sorted(raws, key=lambda r: r['start']):
        for b, off in enumerate(offsets):
            rec = window_at(raw, off, lookback)
            if rec:
                out.append({**rec, 'bucket': b})
    return out


# ---------------- online learning of the decision weights ----------------

def _learning_dir() -> 'Path':
    import os
    from pathlib import Path
    path = Path(os.environ.get('QL_LEARNING_DIR') or Path(__file__).resolve().parents[4] / '.quantlabs' / 'polymarket')
    path.mkdir(parents=True, exist_ok=True)
    return path


def series_key(slug: str | None) -> str:
    """btc-updown-5m-1795375800 → btc-updown-5m: all windows of one series share weights."""
    return re.sub(r'-\d{9,}$', '', slug or 'market')


def _state_path(key: str) -> 'Path':
    return _learning_dir() / (re.sub(r'[^a-z0-9_-]', '_', key.lower())[:80] + '.json')


def load_learning(key: str) -> dict[str, Any] | None:
    path = _state_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def reset_learning(key: str) -> bool:
    path = _state_path(key)
    existed = path.exists()
    path.unlink(missing_ok=True)
    return existed


def _save_learning(key: str, state: dict[str, Any]) -> None:
    path = _state_path(key)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, ensure_ascii=False))
    tmp.replace(path)


def resolved_outcome(slug: str, up_index: str = 'auto', timeout: float = 8) -> float | None:
    """1.0 if UP won, 0.0 if DOWN won, None while the market is still open or unresolved."""
    mk = find_market(slug, ttl=20, timeout=timeout)
    if not mk.get('closed'):
        return None
    prices = [float(p) for p in _jsonlist(mk.get('outcomePrices'))]
    if len(prices) != 2 or max(prices) < .99:
        return None   # closed but not settled yet (or 50-50 refund): skip until it settles
    u, _d = map_outcomes([str(o) for o in _jsonlist(mk.get('outcomes'))], up_index)
    return 1.0 if prices[u] > .5 else 0.0


def _weights_for(state: dict[str, Any], bucket: Any) -> dict[str, float] | None:
    if bucket is None:
        return state['weights']
    entry = state.get('buckets', {}).get(str(bucket))
    return entry['weights'] if entry else None


def online_update(state: dict[str, Any], current_slug: str | None, lr: float, l2: float, up_index: str, timeout: float) -> list[dict[str, Any]]:
    """Settle pending decisions whose market resolved; SGD on log-loss: w ← w − lr·((p − y)·z + l2·w).

    Pending keys are ``slug`` or ``slug#bucket``; each bucket updates its own weight set.
    """
    learned, outcomes = [], {}
    for key in sorted(state['pending']):
        rec = state['pending'][key]
        slug = key.split('#')[0]   # the key is authoritative: slug or slug#bucket
        if slug == current_slug:
            continue
        if slug not in outcomes:
            try:
                outcomes[slug] = resolved_outcome(slug, up_index, timeout)
            except ValueError:
                outcomes[slug] = None
        y = outcomes[slug]
        if y is None:
            if time.time() - rec['t'] > 6 * 3600:   # never settled: drop it
                state['pending'].pop(key)
            continue
        weights = _weights_for(state, rec.get('bucket'))
        if weights is None:                          # bucket layout changed since it was recorded
            state['pending'].pop(key)
            continue
        p = rec['p']
        before = dict(weights)
        grad = p - y
        for k in SIGNALS:
            weights[k] = float(np.clip(weights.get(k, 0.0) - lr * (grad * rec['z'].get(k, 0.0) + l2 * weights.get(k, 0.0)), -5, 5))
        weights['bias'] = float(np.clip(weights['bias'] - lr * grad, -5, 5))
        loss = -float(y * np.log(max(p, 1e-6)) + (1 - y) * np.log(max(1 - p, 1e-6)))
        entry = {'slug': slug, 'bucket': rec.get('bucket'), 'decided_at': rec['as_of'], 'p_up': round(p, 4), 'decision': rec['decision'],
                 'outcome': 'UP' if y else 'DOWN', 'hit': int((p >= .5) == (y == 1.0)), 'logloss': round(loss, 4), 'p_market': rec.get('p_market'),
                 'market_hit': None if rec.get('p_market') is None else int((rec['p_market'] >= .5) == (y == 1.0)),
                 'weights_before': before, 'weights': dict(weights)}
        state['history'].append(entry)
        state['history'] = state['history'][-1500:]
        state['pending'].pop(key)
        learned.append(entry)
    return learned


def learning_summary(state: dict[str, Any], bucket: Any = None, window: int = 50) -> dict[str, Any]:
    h = state['history']
    mine = [e for e in h if e.get('bucket') == bucket] if bucket is not None else h
    recent = h[-window:]
    mh = [e['market_hit'] for e in recent if e.get('market_hit') is not None]
    by_bucket = []
    for b, entry in sorted(state.get('buckets', {}).items(), key=lambda kv: int(kv[0])):
        es = [e for e in h if str(e.get('bucket')) == b][-window:]
        ms = [e['market_hit'] for e in es if e.get('market_hit') is not None]
        by_bucket.append({'bucket': int(b), 'n': len(es), 'hit_rate': float(np.mean([e['hit'] for e in es])) if es else None,
                          'market_hit_rate': float(np.mean(ms)) if ms else None,
                          'logloss': float(np.mean([e['logloss'] for e in es])) if es else None, 'weights': entry['weights']})
    cur = state['buckets'][str(bucket)] if bucket is not None and str(bucket) in state.get('buckets', {}) else state
    return {'key': state['key'], 'updates': len(h), 'pending': sorted(state['pending']), 'weights': cur['weights'],
            'initial_weights': cur['initial_weights'], 'hit_rate': float(np.mean([e['hit'] for e in recent])) if recent else None,
            'market_hit_rate': float(np.mean(mh)) if mh else None, 'bucket': bucket, 'bucket_updates': len(mine), 'by_bucket': by_bucket,
            'logloss': float(np.mean([e['logloss'] for e in recent])) if recent else None, 'window': window,
            'history': [{k: e.get(k) for k in ('slug', 'bucket', 'decided_at', 'p_up', 'decision', 'outcome', 'hit', 'logloss', 'weights')} for e in mine[-60:]]}


# ---------------- UP/DOWN decision ----------------

SIGNALS = ['market', 'momentum', 'laya', 'imbalance', 'target']


def _logit(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(x / (1 - x))


def _sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-np.clip(z, -30, 30))))


def laya_updown(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """P(UP) = σ(bias + Σ wᵢ·zᵢ) over log-odds signals known at the last row."""
    if 'Close' not in frame or len(frame) < 5:
        raise ValueError('LAYA UP/DOWN necesita una serie con Close (Polymarket Market u otra fuente).')
    meta = frame.attrs.get('polymarket', {})
    window = str(p.get('window', 'inherit'))
    if window == 'inherit':
        window = meta.get('window') or '1h'
    if window not in WINDOWS:
        raise ValueError('Horizonte: inherit, 5m, 15m, 30m o 1h.')
    lookback_bars = int(p.get('lookback_bars', 30))
    threshold = float(p.get('min_confidence', .55))
    source = str(p.get('weight_source', 'manual'))
    use_laya = bool(p.get('use_laya', True))
    if not (3 <= lookback_bars <= 2000 and .5 <= threshold < 1 and source in ('manual', 'calibrated', 'online')):
        raise ValueError('Lookback 3–2000 barras, confianza mínima 0.5–0.99, pesos manual|calibrated|online.')
    online_lr, online_l2 = float(p.get('online_lr', .05)), float(p.get('online_l2', .001))
    commit = str(p.get('commit_at', 'first'))
    if not (0 < online_lr <= 1 and 0 <= online_l2 <= .1 and commit in ('first', 'last')):
        raise ValueError('lr online 0–1, L2 online 0–0.1, registro first|last.')
    w = {k: float(p.get(f'w_{k}', d)) for k, d in (('market', 1.0), ('momentum', .5), ('laya', 1.0), ('imbalance', .3), ('target', 1.0))}
    bias = float(p.get('bias', 0.0))
    origin = 'Inspector (manual)'
    by_bucket = frame.attrs.get('updown_weights_by_bucket') or {}
    if by_bucket:
        bucket_s = int(frame.attrs.get('updown_bucket_s', 0))
    else:
        bucket_s = bucket_seconds(window) if bool(p.get('minute_buckets', True)) else 0
    step = WINDOWS[window]
    elapsed = max(0.0, time.time() - pd.Timestamp(meta['start']).timestamp()) if meta.get('start') else 0.0
    bucket = bucket_of(elapsed, bucket_s, step)
    n_buckets = step // bucket_s if bucket_s else 1
    # Laya Calibrate on resolved Polymarket windows: per-minute set if available, else the single one
    if bucket is not None and by_bucket and str(bucket) not in by_bucket:   # a minute without enough data: use the nearest calibrated one
        bucket = min((int(k) for k in by_bucket), key=lambda k: abs(k - bucket))
    full = by_bucket.get(str(bucket)) if bucket is not None and by_bucket else frame.attrs.get('updown_weights')
    if source == 'calibrated':
        if full:
            w, bias = {k: float(full.get(k, w[k])) for k in SIGNALS}, float(full['bias'])
            origin = f'Laya Calibrate · {frame.attrs.get("updown_calibration", {}).get("windows", "?")} ventanas resueltas' + (f' · pesos del minuto {bucket + 1}/{n_buckets}' if bucket is not None and by_bucket else '')
        else:
            cw = frame.attrs.get('laya_weights')
            if not cw:
                raise ValueError('weight_source=calibrated requiere Laya Calibrate aguas arriba.')
            # daily Laya Calibrate weights: [bias, log(Pup/Pdown), logit P(higher), risk-1] → laya log-odds channel
            bias, w['laya'] = float(cw[0]), float(cw[1])
            origin = 'Laya Calibrate (TR/VA diario)'
    learning: dict[str, Any] | None = None
    learned: list[dict[str, Any]] = []
    if source == 'online':
        key = str(p.get('learning_key') or '').strip() or series_key(meta.get('slug'))
        seed = {**({k: float(full.get(k, w[k])) for k in SIGNALS} if full else w), 'bias': float(full['bias']) if full else bias}
        lstate = load_learning(key) or {'key': key, 'weights': dict(seed), 'initial_weights': dict(seed), 'pending': {}, 'history': [],
                                        'seeded_from': 'Laya Calibrate' if full else 'Inspector'}
        for k in SIGNALS:
            lstate['weights'].setdefault(k, w[k]); lstate['initial_weights'].setdefault(k, w[k])
        if bucket is not None and (lstate.get('bucket_s') != bucket_s or len(lstate.get('buckets', {})) != n_buckets):
            # one weight set per minute bucket, each seeded from its calibrated set (or the shared seed)
            lstate['bucket_s'] = bucket_s
            lstate['buckets'] = {}
            for bb in range(n_buckets):
                cb = by_bucket.get(str(bb))
                sb = {**({k: float(cb.get(k, w[k])) for k in SIGNALS} if cb else {k: seed[k] for k in SIGNALS}), 'bias': float(cb['bias']) if cb else seed['bias']}
                lstate['buckets'][str(bb)] = {'weights': dict(sb), 'initial_weights': dict(sb)}
            lstate['pending'] = {k: v for k, v in lstate['pending'].items() if v.get('bucket') is None}
            lstate['seeded_from'] = 'Laya Calibrate por minuto' if by_bucket else lstate.get('seeded_from', 'Inspector')
        if progress and lstate['pending']: progress({'detail': f'Aprendizaje en línea: comprobando {len(lstate["pending"])} ventana(s) pendientes'})
        learned = online_update(lstate, meta.get('slug'), online_lr, online_l2, str(p.get('up_outcome', 'auto')), 8)
        cw = _weights_for(lstate, bucket) or lstate['weights']
        w = {k: float(cw.get(k, w[k])) for k in SIGNALS}
        bias = float(cw['bias'])
        origin = f'Online ({len(lstate["history"])} decisiones resueltas · {key} · inicio {lstate.get("seeded_from", "Inspector")})' + (f' · minuto {bucket + 1}/{n_buckets}' if bucket is not None else '')
    close = frame['Close'].astype(float)
    p_mkt = float(close.iloc[-1]) if meta else 0.5
    if {'Book_best_bid', 'Book_best_ask'} <= set(frame.columns):
        bb, ba = float(frame['Book_best_bid'].iloc[-1]), float(frame['Book_best_ask'].iloc[-1])
        if np.isfinite(bb) and np.isfinite(ba) and ba - bb <= .1:
            p_mkt = (bb + ba) / 2   # live mid beats the last traded history point
    last = lambda col: float(frame[col].iloc[-1]) if col in frame else np.nan
    z = market_signals(close, p_mkt, last('Book_imbalance'), lookback_bars, (meta.get('underlying') or {}).get('signal'))
    laya_detail: dict[str, Any] = {'used': False}
    if 'Laya_P_up' in frame and np.isfinite(last('Laya_P_up')):  # Laya Signal upstream already answered
        up, dn = last('Laya_P_up'), last('Laya_P_down')
        z['laya'] = float(np.log(max(up, 1e-4) / max(dn, 1e-4)))
        laya_detail = {'used': True, 'via': 'Laya Signal', 'P_up': up, 'P_down': dn}
    elif use_laya:
        from .laya_nodes import _agent
        sec = last('SecondsToClose')
        state = laya_state(meta.get('question', 'market'), window, p_mkt, z, last('Book_spread'), sec / 60 if np.isfinite(sec) else None)
        if progress: progress({'detail': 'Consultando Laya-MLX'})
        res = _agent(str(p.get('checkpoint', 'laya-mlx'))).predict(state, laya_question(window, p.get('question')))['answers']['direction']
        up, dn = float(res['probabilities']['up']), float(res['probabilities']['down'])
        z['laya'] = float(np.log(max(up, 1e-4) / max(dn, 1e-4)))
        laya_detail = {'used': True, 'via': 'laya-mlx directo', 'P_up': up, 'P_down': dn, 'state': state}
    z.pop('_momentum_raw', None); z.pop('_target_pct', None)
    contrib = {k: w[k] * z[k] for k in SIGNALS}
    score = bias + sum(contrib.values())
    p_up = _sigmoid(score)
    conf = max(p_up, 1 - p_up)
    decision = 'UP' if p_up >= .5 else 'DOWN'
    actionable = conf >= threshold
    trace = [{'señal': k, 'z (log-odds)': round(z[k], 4), 'peso': round(w[k], 4), 'contribución': round(contrib[k], 4)} for k in SIGNALS]
    trace.append({'señal': 'bias', 'z (log-odds)': 1.0, 'peso': round(bias, 4), 'contribución': round(bias, 4)})
    if source == 'online':
        slug = meta.get('slug')
        pkey = slug if bucket is None else f'{slug}#{bucket}'
        if slug and not meta.get('closed') and (commit == 'last' or pkey not in lstate['pending']):
            lstate['pending'][pkey] = {'t': time.time(), 'slug': slug, 'bucket': bucket, 'as_of': str(frame.index[-1]), 'z': z, 'p': p_up, 'decision': decision, 'p_market': p_mkt}
        _save_learning(key, lstate)
        learning = {**learning_summary(lstate, bucket), 'learned_now': [{k: e.get(k) for k in ('slug', 'bucket', 'p_up', 'outcome', 'hit')} for e in learned], 'lr': online_lr}
    out = frame.copy()
    out['UpDown_P_up'] = np.nan
    out['UpDown_Decision'] = np.nan
    out.iloc[-1, out.columns.get_loc('UpDown_P_up')] = p_up
    out.iloc[-1, out.columns.get_loc('UpDown_Decision')] = 1.0 if decision == 'UP' else -1.0
    out.attrs.update(frame.attrs)
    result = {'decision': decision, 'probability_up': round(p_up, 4), 'confidence': round(conf, 4), 'actionable': actionable,
              'horizon': window, 'market': meta.get('slug'), 'as_of': str(frame.index[-1]), 'weights': {**w, 'bias': bias},
              'weight_source': origin, 'signals': z, 'contributions': {**contrib, 'bias': bias}, 'laya': laya_detail,
              'question': meta.get('question'), 'p_market': p_mkt, 'threshold': threshold, 'learning': learning,
              'bucket': bucket, 'bucket_s': bucket_s, 'buckets_total': n_buckets, 'elapsed_s': round(elapsed, 1), 'window_start': meta.get('start'),
              'bucket_stats': frame.attrs.get('updown_calibration', {}).get('buckets')}
    out.attrs['updown'] = result
    report = {
        'topic': 'laya_updown', 'selected': f'{decision} · {conf:.1%} · {window}', 'features': SIGNALS,
        'splits': {'train': 0, 'validation': 0, 'test': 1},
        'notes': [
            f'Decisión {decision} para la ventana {window}: P(UP) = σ(bias + Σ wᵢ·zᵢ) = {p_up:.4f}. ' + ('Supera' if actionable else 'No supera') + f' la confianza mínima {threshold}.',
            (f'Minuto {bucket + 1} de {n_buckets} ({elapsed:.0f}s transcurridos, tramos de {bucket_s}s): se usan los pesos de ese tramo.' if bucket is not None else 'Un único juego de pesos para toda la ventana.'),
            ('Aprendizaje en línea: al resolverse cada ventana, w ← w − lr·((p − y)·z + L2·w) con la decisión registrada antes del cierre. '
             + (f'Acierto últimas {learning["window"]}: {learning["hit_rate"]:.1%} vs mercado {learning["market_hit_rate"]:.1%}.' if learning and learning['hit_rate'] is not None and learning['market_hit_rate'] is not None else 'Aún sin ventanas resueltas.')) if source == 'online' else
            f'Pesos: {origin}. Cámbialos en el Inspector o conecta Laya Calibrate y usa weight_source=calibrated.',
            'z_market = logit P(UP) implícita; z_momentum = tanh(Δp/σ·√n)·2; z_imbalance = 2·desequilibrio del libro; z_laya = log(P_up/P_down) de Laya; z_target = Δ% desde la apertura / (σ 1m · √min restantes) (≈ probit de P(UP) en paseo aleatorio).',
            ('Laya: ' + laya_detail['via'] + f' P_up={laya_detail["P_up"]:.3f}.') if laya_detail['used'] else 'Laya desactivado: z_laya = 0.',
            'JSON de decisión: ' + json.dumps({k: result[k] for k in ('decision', 'probability_up', 'confidence', 'actionable', 'horizon', 'market', 'as_of')}, ensure_ascii=False),
            'No es asesoría financiera: es una señal de investigación sin garantía de acierto.',
        ],
        'table': trace + ([{'señal': f'aprendido · {e["slug"]}', 'z (log-odds)': e['p_up'], 'peso': e['outcome'], 'contribución': e['hit']} for e in learned]),
        'metrics': {'p_up': p_up, 'confidence': conf, 'actionable': float(actionable), 'score': score, 'decision_up': 1.0 if decision == 'UP' else 0.0},
        'charts': [{'title': 'Contribución por señal (log-odds)', 'rows': [{'señal': r['señal'], 'contribución': r['contribución']} for r in trace], 'series': ['contribución'], 'x_key': 'señal', 'chart_type': 'bar'}],
        'decision': result,
    }
    return out, report


# ---------------- Laya Signal / Laya Calibrate on resolved Polymarket windows ----------------

_LAYA_CACHE: dict[str, dict[str, float]] = {}


def _ask_laya(states: list[dict[str, Any]], questions: dict[str, Any], checkpoint: str, progress: Progress) -> list[dict[str, float]]:
    from .laya_nodes import _agent
    qkey = json.dumps(questions, sort_keys=True) + checkpoint
    keys = [qkey + json.dumps(s, sort_keys=True) for s in states]
    todo = [i for i, k in enumerate(keys) if k not in _LAYA_CACHE]
    if todo:
        agent = _agent(checkpoint)
        for j in range(0, len(todo), 64):
            part = todo[j:j + 64]
            res = agent.predict_batch([states[i] for i in part], questions, batch_size=64) if hasattr(agent, 'predict_batch') else [agent.predict(states[i], questions) for i in part]
            for i, r in zip(part, res):
                pr = r['answers']['direction']['probabilities']
                _LAYA_CACHE[keys[i]] = {'up': float(pr['up']), 'down': float(pr['down'])}
            if progress: progress({'detail': f'Laya {min(j + 64, len(todo))}/{len(todo)} ventanas'})
    return [_LAYA_CACHE[k] for k in keys]


def laya_signal_polymarket(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One Laya question per resolved window and decision minute, plus the current state."""
    meta = frame.attrs['polymarket']
    series, window = series_key(meta.get('slug')), meta['window']
    n, single, lookback = int(p.get('train_windows', 120)), int(p.get('decision_offset_s', 60)), int(p.get('pm_lookback', 5))
    checkpoint = str(p.get('checkpoint', 'laya-mlx'))
    if not (20 <= n <= 500 and 3 <= lookback <= 60):
        raise ValueError('Ventanas de entrenamiento 20–500 y lookback 3–60.')
    bucket_s = bucket_seconds(window, p.get('bucket_s', 'auto'))
    offsets = decision_offsets(window, bucket_s, single)
    started = time.perf_counter()
    recs_raw = resolved_windows(series, n, offsets, lookback, str(p.get('up_outcome', 'auto')), progress=progress)
    questions = laya_question(window)
    hist_z = [market_signals(pd.Series(r['prices']), r['p_market'], 0.0, lookback, r.get('target')) for r in recs_raw]
    states = [laya_state(r['question'], window, r['p_market'], z, None, r['minutes_to_close']) for r, z in zip(recs_raw, hist_z)]
    # current window: same features, plus the live book
    close = frame['Close'].astype(float)
    last = lambda col: float(frame[col].iloc[-1]) if col in frame else np.nan
    p_now = float(close.iloc[-1])
    bb, ba = last('Book_best_bid'), last('Book_best_ask')
    if np.isfinite(bb) and np.isfinite(ba) and ba - bb <= .1:
        p_now = (bb + ba) / 2
    z_now = market_signals(close, p_now, last('Book_imbalance'), lookback, (meta.get('underlying') or {}).get('signal'))
    sec = last('SecondsToClose')
    states.append(laya_state(meta['question'], window, p_now, z_now, last('Book_spread'), sec / 60 if np.isfinite(sec) else None))
    answers = _ask_laya(states, questions, checkpoint, progress)
    records = []
    for r, z, a in zip(recs_raw, hist_z, answers):
        z = {**z, 'laya': float(np.log(max(a['up'], 1e-4) / max(a['down'], 1e-4)))}
        z.pop('_momentum_raw', None); z.pop('_target_pct', None)
        records.append({'slug': r['slug'], 'start': r['start'], 'offset': r['offset'], 'bucket': r['bucket'], 'y': r['y'],
                        'p_market': r['p_market'], 'laya_p_up': a['up'], 'z': z})
    cur = answers[-1]
    out = frame.copy()
    for col, val in (('Laya_P_up', cur['up']), ('Laya_P_down', cur['down']), ('Laya_P_flat', 0.0), ('Laya_P_higher', cur['up']), ('Laya_Risk', 1.0), ('Laya_Confidence', max(cur['up'], cur['down']))):
        out[col] = np.nan
        out.iloc[-1, out.columns.get_loc(col)] = val
    out.attrs.update(frame.attrs)
    out.attrs['polymarket_windows'] = records
    out.attrs['polymarket_windows_meta'] = {'series': series, 'offsets': offsets, 'bucket_s': bucket_s, 'lookback': lookback, 'window': window}
    windows_n = len({r['start'] for r in records})
    per = []
    for b, off in enumerate(offsets):
        rs = [r for r in records if r['bucket'] == b]
        if not rs:
            continue
        y = np.array([r['y'] for r in rs])
        hit = lambda v: float(((np.array(v) >= .5) == (y == 1)).mean())
        per.append({'minuto': b + 1, 'decisión a (s)': off, 'ventanas': len(rs), 'acierto mercado': round(hit([r['p_market'] for r in rs]), 3),
                    'acierto Laya': round(hit([r['laya_p_up'] for r in rs]), 3), 'acierto distancia': round(float(((np.array([r['z']['target'] for r in rs]) >= 0) == (y == 1)).mean()), 3)})
    y_all = np.array([r['y'] for r in records if r['bucket'] == 0])
    report = {
        'topic': 'laya_signal', 'selected': f'{checkpoint} · {windows_n} ventanas × {len(offsets)} tramos', 'features': list(states[-1].keys()),
        'splits': {'train': 0, 'validation': 0, 'test': len(records)},
        'notes': [
            f'Modo Polymarket por minuto: cada ventana resuelta de {series} se reconstruye en {len(offsets)} momentos ({", ".join(f"{o}s" for o in offsets)}), '
            f'con lookback {lookback} puntos de 1 min y su resultado real. Tramos de {bucket_s or "—"}s (punto medio de cada tramo).' if bucket_s else
            f'Modo Polymarket: una pregunta por ventana resuelta de {series}, con el estado a {single}s del inicio.',
            'Mismas funciones de señales que LAYA UP/DOWN en vivo. El histórico no tiene libro de órdenes: desequilibrio = 0 y spread vacío.',
            f'Pregunta: «{questions["direction"]["instructions"]}». Última fila = estado actual ({meta["slug"]}).',
            f'Tasa base UP: {y_all.mean():.3f} en {windows_n} ventanas. {len(records)} preguntas a Laya, {time.perf_counter() - started:.1f}s.',
            'La tabla muestra, por minuto, el acierto del mercado, de Laya y de la distancia al precio a vencer (signo de z).',
        ],
        'table': per,
        'metrics': {'ventanas': windows_n, 'tramos': len(offsets), 'tasa_base_up': float(y_all.mean()) if len(y_all) else None, 'laya_actual_p_up': cur['up']},
        'charts': [{'title': 'Acierto por minuto de la ventana', 'rows': per, 'series': ['acierto mercado', 'acierto Laya', 'acierto distancia'], 'x_key': 'minuto', 'chart_type': 'line'}],
    }
    return out, report


def _fit_logistic(x: np.ndarray, y: np.ndarray, tr_idx: np.ndarray, va_idx: np.ndarray, lr: float, epochs: int, l2: float, progress: Progress = None) -> tuple[np.ndarray, int, list[dict[str, float]], bool]:
    from .laya_nodes import _scores
    sig = lambda m: 1 / (1 + np.exp(-np.clip(m, -30, 30)))
    w = np.zeros(x.shape[1]); w[1] = 1.0                       # epoch 0 = follow the market
    best, history = (np.inf, w.copy(), 0), []
    xt, yt, xv, yv = x[tr_idx], y[tr_idx], x[va_idx], y[va_idx]
    for epoch in range(1, epochs + 1):
        w = w - lr * (xt.T @ (sig(xt @ w) - yt) / len(yt) + l2 * np.r_[0, w[1:]])
        lt, lv = _scores(sig(xt @ w), yt)['logloss'], _scores(sig(xv @ w), yv)['logloss']
        if lv < best[0]: best = (lv, w.copy(), epoch)
        if epoch % max(1, epochs // 100) == 0 or epoch == 1:
            history.append({'epoch': epoch, 'train': lt, 'validation': lv})
            if progress: progress({'epoch': epoch, 'epochs': epochs, 'train_loss': lt, 'val_loss': lv})
    grad = float(np.linalg.norm(xt.T @ (sig(xt @ w) - yt) / len(yt) + l2 * np.r_[0, w[1:]]))
    # converged: VA stopped early (best before the last 10 %) or the TR gradient has vanished
    converged = best[2] < epochs * .9 or grad < 1e-3
    return best[1], best[2], history, converged


def laya_calibrate_polymarket(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Logistic P(UP) = σ(bias + Σ wᵢ zᵢ) per decision minute. Splits are by window, so one window never lands in two sets."""
    from .laya_nodes import _ece, _scores
    recs = frame.attrs['polymarket_windows']
    wmeta = frame.attrs.get('polymarket_windows_meta', {})
    bucket_s = int(wmeta.get('bucket_s', 0))
    tr, va = float(p.get('train_ratio', .6)), float(p.get('val_ratio', .2))
    lr, epochs, l2 = float(p.get('lr', .1)), int(p.get('epochs', 300)), float(p.get('l2', .01))
    if not (.3 <= tr <= .8 and .05 <= va <= .3 and tr + va < .95 and 1e-4 <= lr <= 5 and 10 <= epochs <= 5000 and 0 <= l2 <= 10):
        raise ValueError('Fracciones TR 0.3–0.8, VA 0.05–0.3 (TR+VA < 0.95); lr 0.0001–5, épocas 10–5000, L2 0–10.')
    fit = ['market', 'momentum', 'laya', 'target']   # no historical book → imbalance cannot be fitted
    starts = sorted({r['start'] for r in recs})
    nw = len(starts)
    a, b = int(nw * tr), int(nw * (tr + va))
    if a < 12 or b - a < 4 or nw - b < 4:
        raise ValueError('Muy pocas ventanas para TR/VA/TE; sube «Ventanas de entrenamiento» en Laya Signal.')
    split = {s: ('tr' if i < a else 'va' if i < b else 'te') for i, s in enumerate(starts)}
    sig = lambda m: 1 / (1 + np.exp(-np.clip(m, -30, 30)))
    imb = float(p.get('imbalance_weight', 0.0))
    by_bucket, rows, wrows, curves, history, first_epoch = {}, [], [], [], [], 0
    pooled: dict[str, list[np.ndarray]] = {'y': []}
    buckets = sorted({r.get('bucket', 0) for r in recs})
    for bk in buckets:
        rs = [r for r in recs if r.get('bucket', 0) == bk]
        x = np.array([[1.0] + [r['z'][k] for k in fit] for r in rs])
        y = np.array([r['y'] for r in rs])
        part = np.array([split[r['start']] for r in rs])
        tr_i, va_i, te_i = np.where(part == 'tr')[0], np.where(part == 'va')[0], np.where(part == 'te')[0]
        if len(tr_i) < 10 or not len(va_i) or not len(te_i):
            continue
        if progress: progress({'detail': f'Calibrando minuto {bk + 1}/{len(buckets)}'})
        w, epoch, hist, converged = _fit_logistic(x, y, tr_i, va_i, lr, epochs, l2, progress if bk == buckets[0] else None)
        if bk == buckets[0]: history, first_epoch = hist, epoch
        cands = {'mercado': sig(x[te_i] @ np.array([0, 1.0, 0, 0, 0])), 'Laya cruda': sig(x[te_i] @ np.array([0, 0, 0, 1.0, 0])),
                 'distancia objetivo': sig(x[te_i] @ np.array([0, 0, 0, 0, 1.7])), 'calibrada': sig(x[te_i] @ w)}
        sc = {k: _scores(v, y[te_i]) for k, v in cands.items()}
        for k, v in cands.items(): pooled.setdefault(k, []).append(v)
        pooled['y'].append(y[te_i])
        weights = {'bias': float(w[0]), 'market': float(w[1]), 'momentum': float(w[2]), 'laya': float(w[3]), 'target': float(w[4]), 'imbalance': imb}
        by_bucket[str(bk)] = weights
        off = rs[0].get('offset')
        rows.append({'minuto': bk + 1, 'decisión a (s)': off, 'TE n': int(len(te_i)), 'época VA': epoch, 'convergió': 'sí' if converged else 'no',
                     **{f'acierto {k}': round(s['accuracy'], 3) for k, s in sc.items() if k != 'Laya cruda'},
                     'logloss mercado': round(sc['mercado']['logloss'], 4), 'logloss distancia': round(sc['distancia objetivo']['logloss'], 4),
                     'logloss calibrada': round(sc['calibrada']['logloss'], 4), 'ventaja vs mercado': round(sc['mercado']['logloss'] - sc['calibrada']['logloss'], 4)})
        wrows.append({'minuto': bk + 1, **{k: round(v, 4) for k, v in weights.items() if k != 'imbalance'}})
        curves.append({'minuto': bk + 1, 'mercado': sc['mercado']['logloss'], 'distancia objetivo': sc['distancia objetivo']['logloss'], 'calibrada': sc['calibrada']['logloss']})
    if not by_bucket:
        raise ValueError('Ningún tramo tiene datos suficientes para calibrar.')
    yp = np.concatenate(pooled['y'])
    total = {k: _scores(np.concatenate(v), yp) for k, v in pooled.items() if k != 'y'}
    rel: dict[float, dict[str, Any]] = {}
    for name in ('mercado', 'distancia objetivo', 'calibrada'):
        for row in _ece(np.concatenate(pooled[name]), yp)[1]:
            rel.setdefault(row['bin'], {'bin': row['bin'], 'perfecta': row['bin']})[name] = row['observada']
    out = frame.copy()
    out.attrs.update(frame.attrs)
    first = by_bucket[min(by_bucket, key=int)]
    out.attrs['updown_weights'] = first
    out.attrs['updown_weights_by_bucket'] = by_bucket if bucket_s else {}
    out.attrs['updown_bucket_s'] = bucket_s
    out.attrs['updown_calibration'] = {'windows': nw, 'test': nw - b, 'scores': total, 'buckets': rows if bucket_s else None}
    multi = bucket_s > 0
    report = {
        'topic': 'laya_calibrate', 'selected': (f'{len(by_bucket)} juegos de pesos (uno por minuto) · {nw} ventanas' if multi else f'mejor época VA = {first_epoch} · {nw} ventanas'),
        'features': ['bias'] + fit,
        'splits': {'train': a, 'validation': b - a, 'test': nw - b},
        'notes': [
            'Modo Polymarket: objetivo = resultado real de cada ventana (UP=1). TR/VA/TE se reparten por ventana en orden cronológico: '
            'los tramos de una misma ventana nunca caen en conjuntos distintos.',
            ('Un juego de pesos por minuto de la ventana; LAYA UP/DOWN usa el del minuto en curso. Época 0 = seguir al mercado.' if multi else
             'Época 0 = seguir al mercado (w_market=1). Si la calibrada no mejora al mercado en TE, las otras señales no aportan información nueva.'),
            '«ventaja vs mercado» = logloss mercado − logloss calibrada en TE: positivo = la calibrada es más precisa que el precio del mercado en ese minuto.',
            'z_target: Δ% del subyacente desde la apertura / (σ 1m · √min restantes), con Binance 1m en histórico y en vivo. «distancia objetivo» = sólo esa señal, P = σ(1.7·z).',
            f'El desequilibrio del libro no tiene histórico: su peso se fija en {imb} (el aprendizaje en línea lo ajusta en vivo).',
            *([f'Minutos {", ".join(str(r["minuto"]) for r in rows if r["convergió"] == "no")}: VA seguía mejorando al final y el gradiente TR no se anuló (> 0.001): el ajuste no convergió; sube épocas o lr.']
              if any(r['convergió'] == 'no' for r in rows) else []),
            'Totales TE (todos los minutos): ' + ', '.join(f'{k} acierto {v["accuracy"]:.3f} logloss {v["logloss"]:.4f}' for k, v in total.items()),
        ],
        'table': rows + [{'minuto': f'pesos {r["minuto"]}', **{k: v for k, v in r.items() if k != 'minuto'}} for r in wrows],
        'metrics': {**{f'te_{m}_{name.replace(" ", "_")}': sc[m] for name, sc in total.items() for m in sc}, 'tramos': len(by_bucket), 'epocas': epochs},
        'charts': ([
            {'title': 'Log-loss TE por minuto (menor = mejor)', 'rows': curves, 'series': ['mercado', 'distancia objetivo', 'calibrada'], 'x_key': 'minuto', 'chart_type': 'line'},
            {'title': 'Pesos calibrados por minuto', 'rows': wrows, 'series': ['market', 'momentum', 'laya', 'target', 'bias'], 'x_key': 'minuto', 'chart_type': 'line'},
        ] if multi else [{'title': 'Curva de aprendizaje (logloss)', 'rows': history, 'series': ['train', 'validation'], 'x_key': 'epoch', 'chart_type': 'line'}])
        + [{'title': 'Diagrama de fiabilidad TE', 'rows': sorted(rel.values(), key=lambda r: r['bin']), 'series': ['perfecta', 'mercado', 'distancia objetivo', 'calibrada'], 'x_key': 'bin', 'chart_type': 'scatter_line'}],
    }
    return out, report


# ---------------- Polymarket Dashboard: the whole active market, organised ----------------

DASH_SORTS = {'volumen_24h': 'volume24hr', 'liquidez': 'liquidity', 'nuevos': 'createdAt'}
DASH_CATEGORIES = ['todas', 'politics', 'crypto', 'sports', 'finance', 'economy', 'tech', 'ai', 'geopolitics', 'elections', 'culture', 'weather']
_GENERIC_TAGS = {'Recurring', 'Hide From New', 'Games', 'All', 'Featured', 'Up or Down', 'Crypto Prices', 'Global Elections', 'World Elections'}


def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def fetch_events(category: str, sort: str, pages: int, timeout: float, ttl: float, progress: Progress = None) -> list[dict[str, Any]]:
    """Active events from Gamma, 50 per page (larger pages are served gzip-only)."""
    out: list[dict[str, Any]] = []
    for page in range(pages):
        params: dict[str, Any] = {'active': 'true', 'closed': 'false', 'order': DASH_SORTS[sort], 'ascending': 'false', 'limit': 50, 'offset': page * 50}
        if category != 'todas':
            params['tag_slug'] = category
        if progress: progress({'detail': f'Gamma: página {page + 1}/{pages}'})
        rows = _get(f'{GAMMA}/events', params, ttl=ttl, timeout=timeout)
        if not isinstance(rows, list) or not rows:
            break
        out.extend(rows)
        if len(rows) < 50:
            break
    return out


def _market_row(mk: dict[str, Any], ev: dict[str, Any], category: str, now: pd.Timestamp) -> dict[str, Any] | None:
    if mk.get('closed') or not mk.get('active', True):
        return None
    outcomes = [str(o) for o in _jsonlist(mk.get('outcomes'))]
    prices = [_f(x) for x in _jsonlist(mk.get('outcomePrices'))]
    if len(outcomes) != len(prices) or not outcomes or any(v is None for v in prices):
        return None
    lead = int(np.argmax(prices))
    end = pd.Timestamp(mk.get('endDate') or ev.get('endDate')) if (mk.get('endDate') or ev.get('endDate')) else None
    hours = None if end is None else (end.tz_convert('UTC') - now).total_seconds() / 3600
    yes = prices[0]
    title = str(ev.get('title') or '')
    label = f"{title[:60]}: {mk['groupItemTitle']}" if mk.get('groupItemTitle') else (mk.get('question') or title)
    return {'mercado': label, 'pregunta': mk.get('question'), 'evento': ev.get('title'), 'slug': mk.get('slug'),
            'evento_slug': ev.get('slug'), 'categoría': category, 'outcome_líder': outcomes[lead], 'prob_líder': prices[lead], 'prob_primero': yes,
            'outcomes': ' / '.join(outcomes), 'bid': _f(mk.get('bestBid')), 'ask': _f(mk.get('bestAsk')), 'spread': _f(mk.get('spread')),
            'último': _f(mk.get('lastTradePrice')), 'cambio_1h': _f(mk.get('oneHourPriceChange')), 'cambio_24h': _f(mk.get('oneDayPriceChange')),
            'cambio_7d': _f(mk.get('oneWeekPriceChange')), 'volumen_24h': _f(mk.get('volume24hr')) or 0.0, 'volumen': _f(mk.get('volumeNum')) or 0.0,
            'liquidez': _f(mk.get('liquidityNum')) or 0.0, 'cierre': None if end is None else end.isoformat(), 'horas_al_cierre': hours,
            'acepta_órdenes': bool(mk.get('acceptingOrders')), 'url': f'https://polymarket.com/event/{ev.get("slug")}'}


def _event_category(ev: dict[str, Any], wanted: str) -> str:
    labels = [t.get('label') for t in ev.get('tags', []) if isinstance(t, dict) and t.get('label') and t.get('label') not in _GENERIC_TAGS]
    if wanted != 'todas':   # inside one category, break it down by its sub-tags (Bitcoin, NBA, …)
        sub = [l for l in labels if l.lower() != wanted.lower()]
        return sub[0] if sub else wanted.capitalize()
    for main in ('Politics', 'Crypto', 'Sports', 'Finance', 'Economy', 'Tech', 'AI', 'Geopolitics', 'Culture', 'Weather', 'Elections', 'Business', 'Science'):
        if main in labels:
            return main
    return labels[0] if labels else 'Otros'


def polymarket_dashboard(p: dict[str, Any], progress: Progress = None, market: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if market is not None:
        if not market.attrs.get('polymarket'):
            raise ValueError('Polymarket Dashboard: la entrada debe venir de Polymarket Market (o de un nodo que lo conserve, como LAYA UP/DOWN).')
        return polymarket_market_dashboard(market, p, progress)
    category, sort = str(p.get('category', 'todas')), str(p.get('sort', 'volumen_24h'))
    pages, top_n = int(p.get('pages', 2)), int(p.get('top_n', 15))
    min_vol, timeout, ttl = float(p.get('min_volume_24h', 0)), float(p.get('timeout', 10)), float(p.get('cache_seconds', 30))
    query = str(p.get('query', '')).strip().lower()
    if category not in DASH_CATEGORIES or sort not in DASH_SORTS or not (1 <= pages <= 10 and 5 <= top_n <= 50 and min_vol >= 0 and 1 <= timeout <= 30 and 0 <= ttl <= 3600):
        raise ValueError(f'Categoría {"/".join(DASH_CATEGORIES)}; orden {"/".join(DASH_SORTS)}; páginas 1–10; top 5–50; volumen mínimo ≥ 0; timeout 1–30 s; caché 0–3600 s.')
    started = time.perf_counter()
    events = fetch_events(category, sort, pages, timeout, ttl, progress)
    if not events:
        raise ValueError('Gamma no devolvió eventos activos para ese filtro.')
    now = pd.Timestamp.now(tz='UTC')
    erows, mrows = [], []
    for ev in events:
        cat = _event_category(ev, category)
        if query and query not in str(ev.get('title', '')).lower():
            continue
        ms = [r for r in (_market_row(mk, ev, cat, now) for mk in ev.get('markets', [])) if r]
        end = pd.Timestamp(ev['endDate']) if ev.get('endDate') else None
        erows.append({'evento': ev.get('title'), 'slug': ev.get('slug'), 'categoría': cat, 'volumen_24h': _f(ev.get('volume24hr')) or 0.0,
                      'volumen': _f(ev.get('volume')) or 0.0, 'interés_abierto': _f(ev.get('openInterest')) or 0.0,
                      'liquidez': _f(ev.get('liquidity')) or float(sum(m['liquidez'] for m in ms)), 'mercados': len(ms),
                      'cierre': None if end is None else end.isoformat(), 'horas_al_cierre': None if end is None else (end.tz_convert('UTC') - now).total_seconds() / 3600,
                      'comentarios': int(ev.get('commentCount') or 0), 'recurrente': bool(ev.get('seriesSlug')), 'url': f'https://polymarket.com/event/{ev.get("slug")}'})
        mrows.extend(ms)
    ev_df = pd.DataFrame(erows)
    mk_df = pd.DataFrame(mrows)
    if mk_df.empty:
        raise ValueError('Ningún mercado abierto con precios válidos para ese filtro.')
    mk_df = mk_df[mk_df['volumen_24h'] >= min_vol].reset_index(drop=True)
    ev_df = ev_df[ev_df['volumen_24h'] >= min_vol].reset_index(drop=True)
    pick = lambda df, cols: [{k: (None if isinstance(v, float) and not np.isfinite(v) else (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items()} for r in df[cols].to_dict('records')]
    mcols = ['mercado', 'evento', 'categoría', 'outcome_líder', 'prob_líder', 'cambio_24h', 'volumen_24h', 'liquidez', 'spread', 'horas_al_cierre', 'url']
    active = mk_df[(mk_df['prob_primero'] > .03) & (mk_df['prob_primero'] < .97)]     # leave out near-settled markets
    traded = active[active['volumen_24h'] >= 100]   # a price move without trades is not information
    movers = traded.reindex(traded['cambio_24h'].abs().sort_values(ascending=False).index).dropna(subset=['cambio_24h'])
    soon = mk_df[(mk_df['horas_al_cierre'] > 0) & (mk_df['horas_al_cierre'] <= 24)].sort_values('volumen_24h', ascending=False)
    contested = active[active['volumen_24h'] > 0].assign(_d=lambda d: (d['prob_primero'] - .5).abs()).sort_values(['_d', 'volumen_24h'], ascending=[True, False])
    contested = contested[contested['_d'] <= .1].sort_values('volumen_24h', ascending=False)
    wide = active[active['volumen_24h'] > 1000].sort_values('spread', ascending=False)
    cats = ev_df.groupby('categoría').agg(volumen_24h=('volumen_24h', 'sum'), eventos=('evento', 'count'), interés_abierto=('interés_abierto', 'sum')).sort_values('volumen_24h', ascending=False).reset_index()
    updown = mk_df[mk_df['slug'].fillna('').str.contains(r'-updown-(?:5m|15m|30m|1h)-')]
    ud = updown.assign(ventana=updown['slug'].str.extract(r'-updown-(5m|15m|30m|1h)-')[0], activo=updown['slug'].str.extract(r'^([a-z]+)-updown')[0]) if not updown.empty else updown
    hist = np.histogram(mk_df['prob_primero'].clip(0, 1), bins=10, range=(0, 1))[0]
    total_vol, total_oi = float(ev_df['volumen_24h'].sum()), float(ev_df['interés_abierto'].sum())
    kpis = {'eventos': int(len(ev_df)), 'mercados': int(len(mk_df)), 'volumen_24h': total_vol, 'interés_abierto': total_oi,
            'liquidez': float(mk_df['liquidez'].sum()), 'spread_mediano': None if active['spread'].dropna().empty else float(active['spread'].median()),
            'cierran_24h': int(len(soon)), 'eventos_cierran_24h': int(soon['evento_slug'].nunique()) if not soon.empty else 0,
            'mercados_por_evento_max': int(mk_df.groupby('evento_slug').size().max()), 'mercados_disputados': int(len(contested)), 'categorías': int(cats.shape[0]),
            'top_categoría': None if cats.empty else str(cats.iloc[0]['categoría']),
            'concentración_top10': float(ev_df.nlargest(10, 'volumen_24h')['volumen_24h'].sum() / total_vol) if total_vol > 0 else None}
    dashboard = {
        'kpis': kpis, 'filters': {'category': category, 'sort': sort, 'pages': pages, 'query': query, 'min_volume_24h': min_vol},
        'fetched_at': now.isoformat(), 'seconds': round(time.perf_counter() - started, 1),
        'categories': pick(cats.head(12), list(cats.columns)) if not cats.empty else [],
        'top_events': pick(ev_df.sort_values('volumen_24h', ascending=False).head(top_n), ['evento', 'categoría', 'volumen_24h', 'interés_abierto', 'liquidez', 'mercados', 'horas_al_cierre', 'url']),
        'top_markets': pick(mk_df.sort_values('volumen_24h', ascending=False).head(top_n), mcols),
        'movers': pick(movers.head(top_n), mcols),
        'ending_soon': pick(soon.head(top_n), mcols),
        'contested': pick(contested.head(top_n), mcols),
        'most_liquid': pick(mk_df.sort_values('liquidez', ascending=False).head(top_n), mcols),
        'wide_spreads': pick(wide.head(top_n), mcols),
        'probability_histogram': [{'tramo': f'{i * 10}–{i * 10 + 10}%', 'mercados': int(c)} for i, c in enumerate(hist)],
        'updown': [] if updown.empty else pick(ud.groupby(['activo', 'ventana']).agg(mercados=('slug', 'count'), volumen_24h=('volumen_24h', 'sum'), prob_up_media=('prob_primero', 'mean')).reset_index(), ['activo', 'ventana', 'mercados', 'volumen_24h', 'prob_up_media']),
    }
    out = mk_df.copy()
    out.attrs['polymarket_dashboard'] = dashboard
    def money(v: float) -> str:
        return f'${v / 1e6:.2f} M' if v >= 1e6 else f'${v / 1e3:.1f} k' if v >= 1e3 else f'${v:.0f}'
    top_ev = dashboard['top_events'][0] if dashboard['top_events'] else None
    report = {
        'topic': 'polymarket_dashboard', 'selected': f'{kpis["eventos"]} eventos · {kpis["mercados"]} mercados · {category}', 'features': mcols,
        'splits': {'train': 0, 'validation': 0, 'test': kpis['mercados']},
        'notes': [
            f'Gamma API /events activos (closed=false), orden {DASH_SORTS[sort]}, {pages} página(s) de 50, categoría {category}' + (f', texto «{query}»' if query else '') + f'. {dashboard["seconds"]}s.',
            f'Volumen 24 h {money(total_vol)} · interés abierto {money(total_oi)} · {kpis["cierran_24h"]} mercados de {kpis["eventos_cierran_24h"]} eventos cierran en 24 h.',
            f'Hay eventos con hasta {kpis["mercados_por_evento_max"]} mercados (p. ej. apuestas por jugador en deportes): los conteos de mercados están dominados por ellos; compara también por eventos.',
            (f'Evento más activo: «{top_ev["evento"]}» ({money(top_ev["volumen_24h"])} en 24 h).' if top_ev else 'Sin eventos.'),
            'Movimientos y disputados excluyen mercados casi resueltos (precio < 3 % o > 97 %); movimientos exigen ≥ $100 negociados en 24 h. Disputado = probabilidad entre 40 % y 60 %. «Por resolver» = pasó la fecha de cierre pero aún no se ha resuelto.',
            'Los datos son del momento de la consulta; Gamma puede tardar unos segundos en reflejar operaciones recientes.',
        ],
        'table': dashboard['top_markets'],
        'metrics': {k: v for k, v in kpis.items() if isinstance(v, (int, float)) or v is None},
        'charts': [
            {'title': 'Volumen 24 h por categoría', 'rows': dashboard['categories'], 'series': ['volumen_24h'], 'x_key': 'categoría', 'chart_type': 'bar'},
            {'title': 'Distribución de probabilidades (primer outcome)', 'rows': dashboard['probability_histogram'], 'series': ['mercados'], 'x_key': 'tramo', 'chart_type': 'bar'},
        ],
        'dashboard': dashboard,
    }
    return out, report


# ---------------- Polymarket Dashboard of ONE market (input from Polymarket Market) ----------------

DATA_API = 'https://data-api.polymarket.com/v2'


def _book_levels(token: str, timeout: float, ttl: float) -> dict[str, list[dict[str, float]]]:
    book = _get(f'{CLOB}/book', {'token_id': token}, ttl=ttl, timeout=timeout)
    bids = sorted(((float(b['price']), float(b['size'])) for b in book.get('bids', [])), reverse=True)
    asks = sorted((float(a['price']), float(a['size'])) for a in book.get('asks', []))
    return {'bids': [{'precio': p, 'acciones': s, 'usd': p * s} for p, s in bids], 'asks': [{'precio': p, 'acciones': s, 'usd': p * s} for p, s in asks]}


def _depth_curve(book: dict[str, list[dict[str, float]]], width: float = .2) -> list[dict[str, Any]]:
    """Cumulative $ depth around the mid, the classic depth chart."""
    if not book['bids'] or not book['asks']:
        return []
    mid = (book['bids'][0]['precio'] + book['asks'][0]['precio']) / 2
    rows, cum = [], 0.0
    for lv in book['bids']:
        if lv['precio'] < mid - width:
            break
        cum += lv['usd']
        rows.append({'precio': round(lv['precio'], 3), 'compra': round(cum, 2)})
    rows = rows[::-1]
    cum = 0.0
    for lv in book['asks']:
        if lv['precio'] > mid + width:
            break
        cum += lv['usd']
        rows.append({'precio': round(lv['precio'], 3), 'venta': round(cum, 2)})
    return rows


def polymarket_market_dashboard(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = frame.attrs['polymarket']
    timeout, ttl = float(p.get('timeout', 10)), float(p.get('cache_seconds', 30))
    n_trades = int(p.get('trades', 200))
    if not (1 <= timeout <= 30 and 0 <= ttl <= 3600 and 20 <= n_trades <= 1000):
        raise ValueError('Timeout 1–30 s, caché 0–3600 s, operaciones 20–1000.')
    started = time.perf_counter()
    notes: list[str] = []
    now = pd.Timestamp.now(tz='UTC')
    outcomes, tokens = meta.get('outcomes') or [], meta.get('tokens') or []
    # 1) order book of every outcome
    books: dict[str, Any] = {}
    for name, tok in zip(outcomes, tokens):
        try:
            if progress: progress({'detail': f'Libro «{name}»'})
            books[name] = _book_levels(tok, timeout, min(ttl, 10))
        except ValueError as exc:
            notes.append(f'Libro de «{name}» no disponible ({exc}).')
    # 2) recent trades (Data API v2; wallets are never kept)
    trades: list[dict[str, Any]] = []
    if meta.get('condition_id'):
        try:
            if progress: progress({'detail': 'Operaciones recientes'})
            raw = _get(f'{DATA_API}/trades', {'condition': meta['condition_id'], 'limit': n_trades}, ttl=min(ttl, 15), timeout=timeout)
            for t in (raw.get('data', []) if isinstance(raw, dict) else []):
                price, size = _f(t.get('price')), _f(t.get('size'))
                if price is None or size is None:
                    continue
                trades.append({'hora': pd.Timestamp(int(t['timestamp']), unit='s', tz='UTC').isoformat(), 'ts': int(t['timestamp']), 'lado': t.get('side'),
                               'outcome': t.get('outcome'), 'precio': price, 'acciones': size, 'usd': price * size})
        except (ValueError, KeyError, TypeError) as exc:
            notes.append(f'Operaciones no disponibles ({exc}).')
    trades.sort(key=lambda t: -t['ts'])
    # 3) holder concentration (aggregates only: no names, no wallets)
    holders: list[dict[str, Any]] = []
    if meta.get('condition_id'):
        try:
            raw = _get(f'{DATA_API}/holders', {'condition': meta['condition_id'], 'limit': 20}, ttl=max(ttl, 60), timeout=timeout)
            for block in (raw.get('data', []) if isinstance(raw, dict) else []):
                amounts = sorted((float(h.get('amount') or 0) for h in block.get('holders', [])), reverse=True)
                idx = next((h.get('outcome_index') for h in block.get('holders', []) if h.get('outcome_index') is not None), None)
                if amounts:
                    holders.append({'outcome': outcomes[idx] if isinstance(idx, int) and idx < len(outcomes) else block.get('token_id', '')[:8],
                                    'tenedores_listados': len(amounts), 'acciones_top': round(sum(amounts), 2),
                                    'mayor_tenedor_pct_top': round(amounts[0] / sum(amounts), 4) if sum(amounts) else None,
                                    'top5_pct_top': round(sum(amounts[:5]) / sum(amounts), 4) if sum(amounts) else None})
        except (ValueError, KeyError, TypeError) as exc:
            notes.append(f'Tenedores no disponibles ({exc}).')
    # 4) the other markets of the same event (multi-outcome events)
    siblings: list[dict[str, Any]] = []
    if meta.get('event_slug'):
        try:
            evs = _get(f'{GAMMA}/events', {'slug': meta['event_slug']}, ttl=ttl, timeout=timeout)
            for mk in (evs[0].get('markets', []) if evs else []):
                prices = [_f(x) for x in _jsonlist(mk.get('outcomePrices'))]
                if not prices or prices[0] is None:
                    continue
                siblings.append({'mercado': mk.get('groupItemTitle') or mk.get('question'), 'slug': mk.get('slug'), 'prob_primero': prices[0],
                                 'cambio_24h': _f(mk.get('oneDayPriceChange')), 'volumen_24h': _f(mk.get('volume24hr')) or 0.0,
                                 'liquidez': _f(mk.get('liquidityNum')) or 0.0, 'cerrado': bool(mk.get('closed')), 'actual': mk.get('slug') == meta.get('slug')})
            siblings.sort(key=lambda r: (r['cerrado'], -r['prob_primero']))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            notes.append(f'Mercados del evento no disponibles ({exc}).')
    # 5) price history statistics (first / UP outcome)
    s = frame['P_up'].astype(float)
    t_last = s.index[-1]
    def ago(hours: float) -> float | None:
        past = s[s.index <= t_last - pd.Timedelta(hours=hours)]
        return None if past.empty else float(s.iloc[-1] - past.iloc[-1])
    book_up = books.get(meta.get('up_outcome')) or {}
    bb = book_up.get('bids', [{}])[0].get('precio') if book_up.get('bids') else None
    ba = book_up.get('asks', [{}])[0].get('precio') if book_up.get('asks') else None
    mid = (bb + ba) / 2 if bb is not None and ba is not None and ba - bb <= .1 else float(s.iloc[-1])
    near = lambda side, ref: sum(lv['usd'] for lv in side if abs(lv['precio'] - ref) <= .05)
    bid_usd, ask_usd = (near(book_up.get('bids', []), bb), near(book_up.get('asks', []), ba)) if bb is not None and ba is not None else (0.0, 0.0)
    # taker side alone is misleading in a binary market: buying the second outcome is betting against the first.
    up_name = meta.get('up_outcome')
    for t in trades:
        t['hacia'] = 'UP' if (t['outcome'] == up_name) == (t['lado'] == 'BUY') else 'DOWN'
    buys = sum(t['usd'] for t in trades if t['hacia'] == 'UP')
    sells = sum(t['usd'] for t in trades if t['hacia'] == 'DOWN')
    vwap = {}
    for name in outcomes:
        ts = [t for t in trades if t['outcome'] == name]
        if ts and sum(t['acciones'] for t in ts):
            vwap[name] = sum(t['precio'] * t['acciones'] for t in ts) / sum(t['acciones'] for t in ts)
    span_min = (trades[0]['ts'] - trades[-1]['ts']) / 60 if len(trades) > 1 else None
    end = pd.Timestamp(meta['end']) if meta.get('end') else None
    secs = None if end is None else (end.tz_convert('UTC') - now).total_seconds()
    kpis = {
        'p_up': mid, 'spread': None if bb is None or ba is None else ba - bb, 'cambio_1h': ago(1), 'cambio_24h': ago(24) if ago(24) is not None else meta.get('change_24h'),
        'máximo_historial': float(max(s.max(), mid)), 'mínimo_historial': float(min(s.min(), mid)), 'volatilidad_puntos': float(s.diff().std() * 100) if len(s) > 2 else None,
        'volumen': meta.get('volume'), 'volumen_24h': meta.get('volume24h'), 'liquidez': meta.get('liquidity'),
        'profundidad_bid_usd': bid_usd, 'profundidad_ask_usd': ask_usd, 'desequilibrio': (bid_usd - ask_usd) / (bid_usd + ask_usd) if bid_usd + ask_usd else None,
        'operaciones': len(trades), 'presión_up_usd': buys, 'presión_down_usd': sells, 'presión_up_pct': buys / (buys + sells) if buys + sells else None,
        'volumen_reciente_usd': buys + sells, 'minutos_cubiertos': span_min,
        'operación_media_usd': (buys + sells) / len(trades) if trades else None, 'mayor_operación_usd': max((t['usd'] for t in trades), default=None),
        'operaciones_por_min': len(trades) / span_min if span_min else None, 'segundos_al_cierre': secs,
    }
    dashboard = {
        'mode': 'market', 'fetched_at': now.isoformat(), 'seconds': round(time.perf_counter() - started, 1),
        'market': {k: meta.get(k) for k in ('slug', 'question', 'event_title', 'event_slug', 'outcomes', 'outcome_prices', 'up_outcome', 'down_outcome', 'window',
                                             'start', 'end', 'closed', 'accepting_orders', 'resolution_source', 'description', 'condition_id')}
                  | {'url': f'https://polymarket.com/event/{meta.get("event_slug") or meta.get("slug")}', 'underlying': meta.get('underlying')},
        'kpis': kpis,
        'history': [{'hora': d.strftime('%Y-%m-%d %H:%M'), 'P': round(float(v), 4)} for d, v in s.items()][-600:],
        'depth': _depth_curve(book_up) if book_up else [],
        'book': {name: {'bids': b['bids'][:12], 'asks': b['asks'][:12]} for name, b in books.items()},
        'trades': [{k: t[k] for k in ('hora', 'lado', 'outcome', 'hacia', 'precio', 'acciones', 'usd')} for t in trades[:30]],
        'trade_flow': _trade_flow(trades),
        'vwap': vwap, 'holders': holders, 'siblings': siblings[:40],
        'decision': frame.attrs.get('updown'),
        'notes': notes,
    }
    out = frame.copy()
    out.attrs.update(frame.attrs)
    out.attrs['polymarket_dashboard'] = dashboard
    money = lambda v: '—' if v is None else (f'${v / 1e6:.2f} M' if v >= 1e6 else f'${v / 1e3:.1f} k' if v >= 1e3 else f'${v:.0f}')
    report = {
        'topic': 'polymarket_dashboard', 'selected': meta.get('question') or meta.get('slug'), 'features': ['P_up', 'libro', 'operaciones', 'tenedores', 'evento'],
        'splits': {'train': 0, 'validation': 0, 'test': len(trades)},
        'notes': [
            f'Dashboard del mercado «{meta.get("question")}» ({meta.get("slug")}), conectado a Polymarket Market.',
            f'Fuentes: libro CLOB /book de cada outcome, Data API v2 /trades (últimas {n_trades}) y /holders (sólo agregados, sin nombres ni wallets), Gamma /events para los mercados hermanos. {dashboard["seconds"]}s.',
            f'P({meta.get("up_outcome")}) {mid:.3f}; {len(trades)} operaciones recientes por {money(buys + sells)}, el {(kpis["presión_up_pct"] or 0) * 100:.0f} % empujando hacia «{up_name}».',
            'Presión hacia un outcome = comprar ese outcome o vender el contrario (el lado del taker por sí solo engaña en mercados binarios).',
            'El volumen de Gamma y el historial CLOB pueden ir con retraso respecto al libro y a las operaciones; los KPIs usan el libro actual.',
            *notes,
        ],
        'table': dashboard['trades'][:15],
        'metrics': {k: v for k, v in kpis.items() if isinstance(v, (int, float)) or v is None},
        'charts': [{'title': f'P({meta.get("up_outcome")}) histórica', 'rows': dashboard['history'], 'series': ['P'], 'x_key': 'hora', 'chart_type': 'line'}],
        'dashboard': dashboard,
    }
    return out, report


def _trade_flow(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """$ pushing toward each outcome per minute over the recent trades (oldest first)."""
    if not trades:
        return []
    df = pd.DataFrame(trades)
    df['minuto'] = pd.to_datetime(df['ts'], unit='s').dt.floor('min')
    g = df.pivot_table(index='minuto', columns='hacia', values='usd', aggfunc='sum', fill_value=0.0)
    return [{'minuto': i.strftime('%H:%M'), 'hacia UP': round(float(r.get('UP', 0.0)), 2), 'hacia DOWN': round(float(r.get('DOWN', 0.0)), 2)} for i, r in g.iterrows()][-120:]
