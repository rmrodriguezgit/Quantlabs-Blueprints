"""Cerebro fixer: turn validation and run-time errors into concrete, checked graph edits.

Order: (1) deterministic rules for known error messages, (2) a catch-all that clamps every
parameter into its catalog range/options, (3) the local LLM (if available) for anything left.
Every proposed edit is validated against the catalog and the graph before it is returned, so
the model can never introduce unknown nodes, keys, values or incompatible connections.
"""
from __future__ import annotations

import json
import re
from typing import Any

from . import llm
from .planner import accepts, clean_params, pins

Fix = dict[str, Any]

SYSTEM = """Eres Cerebro, el asistente de Quantlabs Blueprints. Recibes un grafo de nodos (blueprint), los errores de validación
y/o el error de ejecución, y el catálogo de parámetros permitidos de los nodos implicados.
Propón el MÍNIMO de cambios que corrigen el error. Sólo puedes usar estas operaciones:
- {"type":"set_param","node":"<id>","key":"<parámetro del catálogo>","value":<valor válido>}
- {"type":"connect","source":"<id>","target":"<id>","pin":"<pin de entrada>"}
- {"type":"remove_edge","edge":"<id de arista>"}
No inventes nodos, parámetros ni valores fuera de las opciones/rangos. Si el error no se puede corregir editando el grafo
(p. ej. falta internet o un paquete), devuelve fixes vacíos y explícalo en "unfixable".
Responde SOLO JSON: {"fixes":[...], "explanation":["por qué cada cambio"], "unfixable":["..."]}."""
SCHEMA = {'type': 'object', 'required': ['fixes', 'explanation', 'unfixable'], 'properties': {
    'fixes': {'type': 'array', 'items': {'type': 'object'}}, 'explanation': {'type': 'array', 'items': {'type': 'string'}},
    'unfixable': {'type': 'array', 'items': {'type': 'string'}}}}

YF_PERIOD_FOR = {'1m': '5d', '2m': '1mo', '5m': '1mo', '15m': '1mo', '30m': '1mo', '90m': '1mo', '60m': '1y', '1h': '1y'}


# ---------------- helpers ----------------

def _nodes(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n['id']: n for n in graph.get('nodes', [])}


def _param(node: dict[str, Any], key: str, catalog: dict[str, Any]) -> Any:
    params = node['data'].get('params', {})
    if key in params:
        return params[key]
    return next((f.get('value') for f in catalog.get(node['data']['kind'], {}).get('fields', []) if f['key'] == key), None)


def _label(node: dict[str, Any], catalog: dict[str, Any]) -> str:
    return node['data'].get('label') or catalog.get(node['data']['kind'], {}).get('label') or node['data']['kind']


def node_for_error(graph: dict[str, Any], error: str, catalog: dict[str, Any]) -> str | None:
    """The engine prefixes run-time errors with the failing node's label («Label: message»)."""
    nodes = sorted(_nodes(graph).values(), key=lambda n: -len(_label(n, catalog)))
    for n in nodes:
        if error.startswith(_label(n, catalog) + ':') or error.startswith(_label(n, catalog) + ' ('):
            return n['id']
    return None


def ancestors(graph: dict[str, Any], node_id: str) -> list[str]:
    seen, stack, order = set(), [node_id], []
    while stack:
        cur = stack.pop()
        for e in graph.get('edges', []):
            if e['target'] == cur and e['source'] not in seen:
                seen.add(e['source'])
                order.append(e['source'])
                stack.append(e['source'])
    return order


def _set(node: str, key: str, value: Any, why: str) -> tuple[Fix, str]:
    return {'type': 'set_param', 'node': node, 'key': key, 'value': value}, why


# ---------------- (1) rules for known messages ----------------

def _ticker_suggestion(query: str) -> str | None:
    try:
        import yfinance as yf
        quotes = yf.Search(query, max_results=5).quotes
    except Exception:
        return None
    for q in quotes:
        sym = str(q.get('symbol') or '')
        if sym and sym.upper() != query.upper() and '.' not in sym and q.get('quoteType') in ('EQUITY', 'ETF', 'CRYPTOCURRENCY', 'CURRENCY', 'INDEX', 'MUTUALFUND'):
            return sym
    return None


def _market_suggestion(query: str) -> str | None:
    try:
        from ..engine.nodes.polymarket_nodes import search_markets
    except ImportError:
        from engine.nodes.polymarket_nodes import search_markets
    try:
        hits = [h for h in search_markets(query, 5) if not h['closed'] and h.get('slug')]
    except Exception:
        return None
    return hits[0]['slug'] if hits else None


def rule_fixes(graph: dict[str, Any], error: str, node_id: str | None, catalog: dict[str, Any]) -> tuple[list[tuple[Fix, str]], list[str], bool]:
    """(fixes with reasons, unfixable notes, retry?) for one error message."""
    nodes = _nodes(graph)
    out: list[tuple[Fix, str]] = []
    unfixable: list[str] = []
    node = nodes.get(node_id) if node_id else None
    kind = node['data']['kind'] if node else None
    up = ancestors(graph, node_id) if node_id else []
    up_of = lambda k: next((i for i in up if nodes[i]['data']['kind'] == k), None)
    e = error

    if 'Engine busy' in e or 'Engine busy' in e.lower():
        return [], [], True
    # --- yfinance Data ---
    m = re.search(r'Intervalo (\w+): Yahoo permite como máximo', e) or re.search(r'Con intervalo (\w+) no se puede pedir period=max', e) \
        or re.search(r'Intervalo (\w+): Yahoo sólo conserva', e) or re.search(r'Yahoo sólo da \d+ días de datos (\w+)', e)
    if m and node and kind == 'yfinance_data':
        out.append(_set(node_id, 'range_mode', 'period', 'Usar un periodo relativo en vez de fechas antiguas.'))
        out.append(_set(node_id, 'period', YF_PERIOD_FOR.get(m.group(1), '1mo'), f'Yahoo sólo guarda un historial corto para el intervalo {m.group(1)}.'))
    if 'back_adjust sólo aplica' in e and node:
        out.append(_set(node_id, 'back_adjust', False, 'back_adjust requiere auto_adjust desactivado; se quita.'))
    if 'Pre/post mercado sólo existe' in e and node:
        out.append(_set(node_id, 'prepost', False, 'Pre/post mercado sólo existe en intradía.'))
    m = re.search(r'Yahoo no devolvió datos para (\S+)', e)
    if m and node and kind in ('yfinance_data', 'source'):
        sym = _ticker_suggestion(m.group(1))
        if sym:
            out.append(_set(node_id, 'ticker', sym, f'«{m.group(1)}» no existe en Yahoo; el símbolo más parecido es {sym}.'))
        else:
            unfixable.append(f'Yahoo no reconoce «{m.group(1)}» y no encontré un símbolo equivalente: revisa el ticker (p. ej. BTC-USD, EURUSD=X, ^GSPC).')
    if ('Fechas en formato' in e or 'fecha de fin' in e or 'fecha de inicio' in e or 'indica la fecha de inicio' in e) and node:
        out.append(_set(node_id, 'range_mode', 'period', 'Las fechas no eran válidas; se usa un periodo relativo.'))
    # --- Polymarket ---
    m = re.search(r"Mercado no encontrado en Polymarket: '([^']+)'", e) or re.search(r'No hay mercado activo para (\S+?);', e)
    if m and node and kind == 'polymarket_source':
        query = m.group(1).replace('-', ' ')
        slug = _market_suggestion(query)
        if slug:
            out.append(_set(node_id, 'market', slug, f'No existe «{m.group(1)}»; el mercado abierto más parecido es «{slug}».'))
        else:
            unfixable.append(f'No encontré ningún mercado abierto parecido a «{m.group(1)}»: búscalo en el Inspector.')
    if 'requiere una serie recurrente' in e and node and kind == 'laya_signal':
        out.append(_set(node_id, 'mode', 'daily', 'Este mercado no es una serie recurrente Up/Down; la calibración por ventanas no aplica.'))
        unfixable.append('Laya Signal en modo diario necesita OHLC diario: con un mercado no recurrente usa LAYA UP/DOWN directamente.')
    if 'Sólo' in e and 'ventanas resueltas' in e and node and kind == 'laya_signal':
        out.append(_set(node_id, 'train_windows', 120, 'Pocas ventanas resueltas: se piden 120.'))
    # --- Laya ---
    if 'Muy pocas ventanas para TR/VA/TE' in e:
        sig = up_of('laya_signal')
        if sig:
            cur = int(_param(nodes[sig], 'train_windows', catalog) or 120)
            out.append(_set(sig, 'train_windows', min(500, max(cur * 2, 120)), 'Más ventanas resueltas para que TR, VA y TE tengan muestra.'))
    if 'Muy pocas filas para TR/VA/TE' in e:
        sig = up_of('laya_signal')
        if sig:
            cur = int(_param(nodes[sig], 'max_rows', catalog) or 500)
            out.append(_set(sig, 'max_rows', min(3000, cur * 2), 'Más días evaluados para que TR, VA y TE tengan muestra.'))
    if ('weight_source=calibrated requiere' in e or 'requiere Laya Calibrate aguas arriba' in e) and node:
        out.append(_set(node_id, 'weight_source', 'manual', 'No hay Laya Calibrate antes: se usan los pesos del Inspector.'))
    if 'laya-mlx no está instalado' in e:
        if node and kind == 'laya_updown':
            out.append(_set(node_id, 'use_laya', False, 'laya-mlx no está instalado: LAYA UP/DOWN decide sin la señal de Laya.'))
        else:
            unfixable.append('laya-mlx no está instalado (requiere Apple Silicon): instálalo con .venv/bin/pip install laya-mlx.')
    if 'Laya Feedback simula una estrategia diaria' in e and node:
        unfixable.append('Laya Feedback no aplica a Polymarket: elimina el nodo y usa LAYA UP/DOWN con pesos «online».')
    # --- network ---
    if re.search(r'no responde|URLError|timed out|Temporary failure|Name or service', e):
        target = node_id if node and any(f['key'] == 'timeout' for f in catalog.get(kind, {}).get('fields', [])) else None
        if target:
            cur = float(_param(nodes[target], 'timeout', catalog) or 8)
            out.append(_set(target, 'timeout', min(30.0, cur * 2), 'La fuente tardó en responder: se duplica el timeout.'))
        unfixable.append('Parece un problema de red: comprueba la conexión a internet.')
    return out, unfixable, False


# ---------------- (2) catch-all: every parameter into its catalog range ----------------

def clamp_fixes(graph: dict[str, Any], catalog: dict[str, Any]) -> list[tuple[Fix, str]]:
    out = []
    for n in graph.get('nodes', []):
        kind, params = n['data']['kind'], n['data'].get('params', {})
        if kind not in catalog:
            continue
        notes: list[str] = []
        cleaned = clean_params(kind, {k: v for k, v in params.items() if any(f['key'] == k for f in catalog[kind].get('fields', []))}, catalog, notes)
        for key, value in params.items():
            if key not in cleaned or str(cleaned[key]) == str(value):
                continue   # same value, maybe another type (option 4 vs "4"): not an error
            if not (isinstance(value, (int, float)) and isinstance(cleaned[key], (int, float)) and float(value) == float(cleaned[key])):
                out.append(_set(n['id'], key, cleaned[key], f'{_label(n, catalog)}.{key}: «{value}» fuera de rango/opciones → {cleaned[key]}.'))
    return out


# ---------------- (3) LLM for the rest, sanitised ----------------

def sanitise(graph: dict[str, Any], fixes: list[Any], catalog: dict[str, Any]) -> tuple[list[Fix], list[str]]:
    nodes = _nodes(graph)
    edges = {e.get('id') for e in graph.get('edges', [])}
    ok, dropped = [], []
    for f in fixes if isinstance(fixes, list) else []:
        if not isinstance(f, dict):
            continue
        t = f.get('type')
        if t == 'set_param' and f.get('node') in nodes:
            kind = nodes[f['node']]['data']['kind']
            if not any(fd['key'] == f.get('key') for fd in catalog.get(kind, {}).get('fields', [])):
                dropped.append(f'parámetro inexistente {kind}.{f.get("key")}')
                continue
            notes: list[str] = []
            value = clean_params(kind, {f['key']: f.get('value')}, catalog, notes)[f['key']]
            ok.append({'type': 'set_param', 'node': f['node'], 'key': f['key'], 'value': value})
        elif t == 'connect' and f.get('source') in nodes and f.get('target') in nodes:
            src, dst, pin = nodes[f['source']]['data']['kind'], nodes[f['target']]['data']['kind'], str(f.get('pin', ''))
            if pins(dst).get(pin) and accepts(src, dst, pin):
                ok.append({'type': 'connect', 'source': f['source'], 'target': f['target'], 'pin': pin})
            else:
                dropped.append(f'conexión incompatible {src} → {dst}.{pin}')
        elif t == 'remove_edge' and f.get('edge') in edges:
            ok.append({'type': 'remove_edge', 'edge': f['edge']})
        else:
            dropped.append(f'operación no válida {json.dumps(f, ensure_ascii=False)[:80]}')
    return ok, dropped


def llm_fixes(graph: dict[str, Any], errors: list[str], catalog: dict[str, Any]) -> tuple[list[Fix], list[str], list[str], str | None]:
    kinds = {n['data']['kind'] for n in graph.get('nodes', [])}
    payload = {
        'grafo': {'nodes': [{'id': n['id'], 'kind': n['data']['kind'], 'label': n['data'].get('label'), 'params': n['data'].get('params', {})} for n in graph.get('nodes', [])],
                  'edges': [{'id': e.get('id'), 'source': e['source'], 'target': e['target'], 'pin': e.get('targetHandle')} for e in graph.get('edges', [])]},
        'errores': errors,
        'catalogo': {k: {'label': catalog[k].get('label'), 'input': catalog[k].get('input'), 'output': catalog[k].get('output'),
                         'fields': [{x: f.get(x) for x in ('key', 'label', 'value', 'options', 'min', 'max') if f.get(x) is not None} for f in catalog[k].get('fields', [])]}
                     for k in kinds if k in catalog},
    }
    value, model = llm.chat_json(SYSTEM, json.dumps(payload, ensure_ascii=False, default=str)[:24000], SCHEMA, timeout=120)
    fixes, dropped = sanitise(graph, value.get('fixes', []), catalog)
    explanation = [str(x)[:300] for x in value.get('explanation', []) if isinstance(x, str)][:8]
    unfixable = [str(x)[:300] for x in value.get('unfixable', []) if isinstance(x, str)][:5]
    if dropped:
        explanation.append('Descartado por no cumplir el catálogo: ' + '; '.join(dropped[:4]) + '.')
    return fixes, explanation, unfixable, model


# ---------------- entry point ----------------

def fix(graph: dict[str, Any], error: str | None, issues: list[dict[str, Any]], catalog: dict[str, Any], engine: str = 'auto') -> dict[str, Any]:
    errors = [i for i in issues if i.get('severity', 'error') == 'error']
    messages = [str(i.get('message', '')) for i in errors] + ([error] if error else [])
    fixes: list[Fix] = []
    reasons: list[str] = []
    unfixable: list[str] = []
    retry = False
    seen: set[str] = set()

    def add(items: list[tuple[Fix, str]]) -> None:
        for f, why in items:
            key = json.dumps(f, sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                fixes.append(f)
                reasons.append(why)

    for issue in errors:
        found, cant, again = rule_fixes(graph, str(issue.get('message', '')), issue.get('node'), catalog)
        add(found); unfixable += cant; retry = retry or again
    if error:
        found, cant, again = rule_fixes(graph, error, node_for_error(graph, error, catalog), catalog)
        add(found); unfixable += cant; retry = retry or again
    add(clamp_fixes(graph, catalog))
    used, model, fallback = 'rules', None, None
    if not fixes and not retry and messages and engine != 'rules':
        try:
            more, why, cant, model = llm_fixes(graph, messages, catalog)
            for f in more:
                key = json.dumps(f, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key); fixes.append(f)
            reasons += why
            unfixable += cant
            used = 'ollama'
        except llm.LLMError as exc:
            fallback = str(exc)
    if not fixes and not retry and not unfixable and messages:
        unfixable.append('No encontré una corrección segura para: ' + ' | '.join(m[:160] for m in messages[:3]))
    return {'fixes': fixes, 'explanation': reasons, 'unfixable': list(dict.fromkeys(unfixable)), 'retry': retry,
            'engine': used, 'model': model, 'fallback_reason': fallback}
