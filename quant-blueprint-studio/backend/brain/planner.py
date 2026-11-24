"""Prompt → plan (Qwen or rules) → compiled, engine-checked blueprint graph."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from . import llm, store
from .knowledge import fold, index, library, tool_card

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = json.loads((ROOT / 'src/renderer/data/node-contracts.json').read_text())
CONFIG_PIN = {'features', 'prepare', 'model', 'ml_model', 'backtest', 'curve_fit'}
NETWORK_PIN = {'model', 'curve_fit'}
SOURCES = {'source', 'portfolio_data'}
NAMES = {'bitcoin': 'BTC-USD', 'ethereum': 'ETH-USD', 'oro': 'GLD', 'gold': 'GLD', 'apple': 'AAPL', 'microsoft': 'MSFT', 'nvidia': 'NVDA',
         'tesla': 'TSLA', 'amazon': 'AMZN', 'google': 'GOOGL', 'meta': 'META', 's&p': 'SPY', 'sp500': 'SPY', 'nasdaq': 'QQQ',
         'euro': 'EURUSD=X', 'eurusd': 'EURUSD=X', 'peso': 'MXN=X', 'walmex': 'WALMEX.MX', 'cemex': 'CEMEXCPO.MX'}
NOT_TICKERS = {'RSI', 'MACD', 'EMA', 'SMA', 'VWAP', 'DQN', 'LSTM', 'GRU', 'CNN', 'GAN', 'VAE', 'ML', 'DL', 'RL', 'OLS', 'PCA', 'AE',
               'DAE', 'VA', 'TR', 'TE', 'MSE', 'IA', 'AI', 'ETF', 'USD', 'VAR', 'CVAR', 'ARIMA', 'GARCH', 'KNN', 'SVR', 'API', 'JSON', 'LAYA', 'MLX'}

SYSTEM = """Eres el Cerebro de Quantlabs Blueprints, un estudio visual de investigación cuantitativa por nodos.
Convierte el enunciado en un plan de nodos usando EXCLUSIVAMENTE las herramientas listadas (campo tool = el identificador antes de «·»).
Reglas:
1. El primer paso es una fuente: source (un activo: synthetic o yfinance con ticker) o portfolio_data (varios activos).
2. Cada paso recibe en su entrada principal la salida del paso anterior compatible: frame→frame, tensor→tensor, network→network, portfolio→portfolio.
   Usa "inputs" {pin: id_paso} solo si hay ambigüedad. Red: network_input → layer… → network_output → model (pin network) con prepare (pin tensor).
3. Orden: datos → limpieza/indicadores → preparación → modelo → evaluación (backtest/risk) → plot.
4. "parameters" solo con claves de la ficha de esa herramienta, valores dentro de su rango u opciones.
5. Si algo no lo cubre ninguna herramienta, NO lo simules: descríbelo en "missing".
6. "why": una frase en español sobre para qué sirve el paso. Máximo 12 pasos.
Responde SOLO JSON: {"title": "...", "objective": "...", "steps": [{"id": "datos", "tool": "source", "parameters": {"source": "yfinance", "ticker": "SPY"}, "inputs": {}, "why": "..."}], "missing": [{"name": "...", "purpose": "..."}]}"""

PLAN_SCHEMA = {'type': 'object', 'required': ['title', 'objective', 'steps', 'missing'], 'properties': {
    'title': {'type': 'string'}, 'objective': {'type': 'string'},
    'steps': {'type': 'array', 'items': {'type': 'object', 'required': ['id', 'tool', 'why'], 'properties': {
        'id': {'type': 'string'}, 'tool': {'type': 'string'}, 'parameters': {'type': 'object'}, 'inputs': {'type': 'object'}, 'why': {'type': 'string'}}}},
    'missing': {'type': 'array', 'items': {'type': 'object', 'required': ['name', 'purpose'], 'properties': {'name': {'type': 'string'}, 'purpose': {'type': 'string'}}}}}}


def detect_source(prompt: str) -> dict[str, Any]:
    """Tickers, horizon and data source stated in the prompt (deterministic)."""
    found: dict[str, Any] = {}
    folded = fold(prompt)
    tickers = [NAMES[w] for w in NAMES if re.search(rf'(?<![a-z]){re.escape(w)}(?![a-z])', folded)]
    tickers += [t for t in re.findall(r'(?<![\w^])(\^?[A-Z]{1,6}(?:[.-][A-Z]{1,4})?(?:=X)?)(?![\w])', prompt) if t.upper() not in NOT_TICKERS and len(t) >= 2]
    tickers = list(dict.fromkeys(tickers))
    if tickers:
        found['tickers'] = tickers
    years = re.search(r'(\d{1,2})\s*(?:anos|ano|years|year|yrs|y\b)', folded)
    if years:
        found['years'] = max(1, min(20, int(years.group(1))))
    if re.search(r'sintetic|synthetic|sin internet|offline', folded):
        found['source'] = 'synthetic'
    elif tickers or 'yfinance' in folded or 'yahoo' in folded:
        found['source'] = 'yfinance'
    return found


def clean_params(kind: str, raw: Any, catalog: dict[str, Any], notes: list[str]) -> dict[str, Any]:
    fields = {f['key']: f for f in catalog[kind].get('fields', [])}
    params: dict[str, Any] = {}
    for key, value in (raw or {}).items() if isinstance(raw, dict) else []:
        f = fields.get(key)
        if not f:
            notes.append(f'{kind}: parámetro desconocido «{key}» descartado.')
            continue
        default = f.get('value')
        if f.get('options'):
            if str(value) in map(str, f['options']):
                params[key] = str(value)
            else:
                notes.append(f'{kind}.{key}: «{value}» no es opción válida; se usa {default}.')
        elif isinstance(default, bool):
            params[key] = value if isinstance(value, bool) else str(value).lower() in ('true', '1', 'yes', 'si', 'sí')
        elif isinstance(default, (int, float)):
            try:
                number = float(value)
            except (TypeError, ValueError):
                notes.append(f'{kind}.{key}: «{value}» no es número; se usa {default}.')
                continue
            lo, hi = f.get('min'), f.get('max')
            clamped = min(hi if hi is not None else number, max(lo if lo is not None else number, number))
            if clamped != number:
                notes.append(f'{kind}.{key}: {number} fuera de rango; ajustado a {clamped}.')
            params[key] = int(clamped) if isinstance(default, int) and not isinstance(default, bool) and float(clamped).is_integer() else clamped
        elif isinstance(value, (str, int, float)):
            params[key] = str(value)[:500]
    return {**{k: f['value'] for k, f in fields.items()}, **params}


def pins(kind: str) -> dict[str, str]:
    entry = CONTRACTS['nodes'][kind]
    result = {entry['input']: entry['input']} if entry['input'] else {}
    result.update(entry['extraInputs'])
    if kind in CONFIG_PIN:
        result['config'] = 'config'
    if kind in NETWORK_PIN:
        result['network'] = 'network'
    return result


def accepts(src: str, dst: str, pin: str) -> bool:
    return pin != 'tensor' or src in CONTRACTS['tensorSources'].get(dst, [])


def compile_plan(plan: dict[str, Any], catalog: dict[str, Any], hint: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    steps_out, dropped, notes = [], [], []
    by_step: dict[str, str] = {}
    steps = [s for s in plan.get('steps', []) if isinstance(s, dict)][:12]
    if steps and steps[0].get('tool') not in SOURCES:
        steps.insert(0, {'id': 'datos', 'tool': 'source', 'parameters': {}, 'why': 'Fuente de datos añadida: el plan no tenía una.'})
    for step in steps:
        kind = str(step.get('tool', ''))
        sid = re.sub(r'[^\w-]', '_', str(step.get('id') or kind))[:40] or kind
        if kind not in catalog or kind not in CONTRACTS['nodes']:
            dropped.append({'step': sid, 'tool': kind, 'reason': 'La herramienta no existe en el catálogo.'})
            continue
        params = clean_params(kind, step.get('parameters'), catalog, notes)
        if kind == 'source' and hint:
            if hint.get('source'):
                params['source'] = hint['source']
            if hint.get('tickers') and params.get('source') == 'yfinance':
                params['ticker'] = hint['tickers'][0]
            if hint.get('years'):
                params['years'] = hint['years']
        if kind == 'portfolio_data' and hint.get('tickers') and len(hint['tickers']) > 1 and 'tickers' in params:
            params['tickers'] = ','.join(hint['tickers'])
        if kind == 'laya_signal' and hint.get('tickers'):
            params['ticker'] = hint['tickers'][0]
        node_id = sid if sid not in {n['id'] for n in nodes} else f'{sid}_{len(nodes)}'
        wanted = step.get('inputs') if isinstance(step.get('inputs'), dict) else {}
        new_edges = []
        primary = CONTRACTS['nodes'][kind]['input']
        for pin, ptype in pins(kind).items():
            source_id = by_step.get(str(wanted.get(pin, '')))
            candidates = [n for n in reversed(nodes) if catalog[n['data']['kind']].get('output') == ptype and accepts(n['data']['kind'], kind, pin)]
            if source_id and not any(n['id'] == source_id for n in candidates):
                notes.append(f'{node_id}.{pin}: la conexión pedida a «{wanted.get(pin)}» es incompatible; se busca otra.')
                source_id = None
            if not source_id and pin == primary and candidates:
                used = {e['source'] for e in edges if e['targetHandle'] == pin}
                source_id = next((n['id'] for n in candidates if n['id'] not in used), candidates[0]['id'])
            if source_id:
                new_edges.append({'id': f'{source_id}-{node_id}-{pin}', 'source': source_id, 'target': node_id,
                                  'sourceHandle': ptype, 'targetHandle': pin, 'animated': True})
        if primary and not any(e['targetHandle'] == primary for e in new_edges):
            dropped.append({'step': sid, 'tool': kind, 'reason': f'Ningún paso anterior produce «{primary}».'})
            continue
        if kind == 'plot' and len(new_edges) > 1:
            new_edges = new_edges[:1]
        by_step[str(step.get('id') or sid)] = node_id
        nodes.append({'id': node_id, 'type': 'blueprint', 'position': {'x': 0, 'y': 0},
                      'data': {'kind': kind, 'label': str(step.get('label') or catalog[kind].get('label', kind))[:60], 'params': params}})
        edges += new_edges
        steps_out.append({'id': node_id, 'tool': kind, 'label': catalog[kind].get('label', kind), 'why': str(step.get('why', ''))[:300], 'params': params})
    # Layout: column = longest path from a source, rows stacked per column.
    depth = {n['id']: 0 for n in nodes}
    for _ in nodes:
        for e in edges:
            depth[e['target']] = max(depth[e['target']], depth[e['source']] + 1)
    rows: dict[int, int] = {}
    for n in nodes:
        col = depth[n['id']]
        n['position'] = {'x': col * 330, 'y': 80 + rows.get(col, 0) * 190}
        rows[col] = rows.get(col, 0) + 1
    graph = {'nodes': nodes, 'edges': edges}
    check = structural_check(graph)
    return {'title': str(plan.get('title') or 'Blueprint del Cerebro')[:120], 'objective': str(plan.get('objective', ''))[:400],
            'graph': graph, 'steps': steps_out, 'dropped': dropped, 'notes': notes, 'check': check,
            'missing': [m for m in plan.get('missing', []) if isinstance(m, dict) and m.get('name')][:6]}


def structural_check(graph: dict[str, Any]) -> dict[str, Any]:
    """The engine's own pin, cycle and required-input checks, without executing."""
    try:
        from ..engine.dag_runner import DAGRunner, PINS
    except ImportError:  # server started as a script: backend/ is the import root
        from engine.dag_runner import DAGRunner, PINS
    if not graph['nodes']:
        return {'ok': False, 'errors': ['El plan no produjo nodos ejecutables.']}
    try:
        runner = DAGRunner(copy.deepcopy(graph))
        runner.topological_sort()
    except ValueError as exc:
        return {'ok': False, 'errors': [str(exc)]}
    missing = [f"{n['data']['label']}: falta la entrada {PINS[n['data']['kind']][0]}" for n in graph['nodes']
               if PINS[n['data']['kind']][0] and not any(e['target'] == n['id'] and e['targetHandle'] == PINS[n['data']['kind']][0] for e in graph['edges'])]
    return {'ok': not missing, 'errors': missing}


INTENTS = {  # prompt words (folded) → node kinds a matching template must contain
    'model': r'red(es)? neuronal|neural|deep|lstm|gru|cnn|transformer|capas|dense',
    'backtest': r'backtest|estrategia|strategy',
    'rl_agent': r'refuerzo|reinforcement|dqn|q-learning|agente',
    'portfolio': r'portafolio|cartera|portfolio|markowitz',
    'risk': r'riesgo|risk|var\b|cvar|drawdown|sharpe',
    'ml_model': r'ridge|random forest|bosque|boosting|svr|knn|ml clasic|machine learning',
    'laya_signal': r'laya|decisiones tipadas',
    'data_quality': r'calidad|limpieza|faltantes|duplicad',
    'curve_fit': r'curva|curve|aproximacion',
}


def rank_templates(prompt: str, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Templates covering more requested intents first; BM25 breaks ties."""
    wanted = {k for k, rx in INTENTS.items() if re.search(rx, fold(prompt))}
    scores = {r['id']: r['score'] for r in retrieved if r['type'] == 'blueprint'}
    def key(e: dict[str, Any]) -> tuple[int, int, float]:
        kinds = {n['data']['kind'] for n in e['graph']['nodes']}
        return (len(wanted & kinds), -len(kinds - wanted - {'source', 'features', 'plot', 'prepare', 'config', 'network_input', 'layer', 'network_output'}), scores.get(e['id'], 0.0))
    return sorted(library(), key=key, reverse=True)


def rules_plan(prompt: str, catalog: dict[str, Any], retrieved: list[dict[str, Any]], hint: dict[str, Any]) -> dict[str, Any]:
    """Deterministic fallback: adapt the best-matching library blueprint to the prompt's data."""
    if prompt:
        entry = rank_templates(prompt, retrieved)[0]
    else:
        best = next((r for r in retrieved if r['type'] == 'blueprint'), None)
        entry = next((e for e in library() if best and e['id'] == best['id']), None) or next(e for e in library() if e['id'] == 'area-returns')
    return graph_plan(entry['graph'], entry['title'], f"Adaptado de la plantilla «{entry['title']}»: {entry['description']}", catalog)


def graph_plan(source: dict[str, Any], title: str, objective: str, catalog: dict[str, Any]) -> dict[str, Any]:
    """An existing graph (template or remembered blueprint) as an ordered plan the compiler re-checks."""
    graph = copy.deepcopy(source)
    order = {n['id']: i for i, n in enumerate(graph['nodes'])}
    steps = []
    for node in sorted(graph['nodes'], key=lambda n: (n['position']['x'], n['position']['y'], order[n['id']])):
        kind = node['data']['kind']
        inputs = {e['targetHandle']: e['source'] for e in graph['edges'] if e['target'] == node['id']}
        steps.append({'id': node['id'], 'tool': kind, 'label': node['data'].get('label'), 'parameters': node['data'].get('params', {}),
                      'inputs': inputs, 'why': catalog.get(kind, {}).get('subtitle', '')})
    return {'title': title, 'objective': objective, 'steps': steps, 'missing': []}


def remembered(prompt: str) -> dict[str, Any] | None:
    """A 👍-rated blueprint whose prompt covers most of this prompt's words."""
    from .knowledge import tokens
    wanted = set(tokens(prompt))
    best, score = None, 0.0
    for item in store.liked():
        words = set(tokens(f"{item['prompt']} {item.get('title') or ''}"))
        coverage = len(wanted & words) / max(1, len(wanted))
        if coverage > score:
            best, score = item, coverage
    return best if score >= .6 else None


def plan(prompt: str, catalog: dict[str, Any], engine: str = 'auto') -> dict[str, Any]:
    prompt = prompt.strip()[:2000]
    if len(prompt) < 4:
        raise ValueError('Describe el análisis con al menos unas palabras.')
    if engine not in ('auto', 'ollama', 'rules'):
        raise ValueError('Motor: auto, ollama o rules.')
    catalog = {k: v for k, v in catalog.items() if k in CONTRACTS['nodes'] and isinstance(v, dict)}
    bm25 = index(catalog)
    retrieved = bm25.search(prompt, 8)
    tools = bm25.search(prompt, 14, 'tool')
    hint = detect_source(prompt)
    used, model, reason, raw = 'rules', None, None, None
    if engine in ('auto', 'ollama'):
        cards = {t['id'] for t in tools} | {'source', 'features', 'plot', 'prepare'}
        memo = remembered(prompt)
        examples = ([graph_plan(memo['graph'], memo['title'] or 'Valorado', '', catalog)] if memo else []) + [rules_plan('', catalog, [r], hint) for r in retrieved if r['type'] == 'blueprint'][:2]
        user = ('Herramientas disponibles:\n' + '\n'.join(tool_card(k, catalog[k]) for k in sorted(cards) if k in catalog)
                + '\n\nEjemplos de planes válidos:\n' + '\n'.join(json.dumps({'title': e['title'], 'steps': [{k: s[k] for k in ('id', 'tool', 'inputs')} for s in e['steps']]}, ensure_ascii=False) for e in examples)
                + (f'\n\nDatos detectados en el enunciado: {json.dumps(hint, ensure_ascii=False)}' if hint else '')
                + f'\n\nEnunciado: {prompt}')
        try:
            raw, model = llm.chat_json(SYSTEM, user, PLAN_SCHEMA)
            used = 'ollama'
        except llm.LLMError as exc:
            if engine == 'ollama':
                raise ValueError(f'Qwen no disponible: {exc}') from exc
            reason = str(exc)
    if raw is not None:
        result = compile_plan(raw, catalog, hint)
        if engine == 'auto' and (not result['check']['ok'] or len(result['steps']) < 2):
            reason = 'El plan de Qwen no pasó la validación: ' + '; '.join(result['check']['errors'] or ['menos de dos pasos'])
            used, raw = 'rules', None
    memo = None
    if raw is None:
        memo = remembered(prompt)
        base = (graph_plan(memo['graph'], memo['title'] or 'Blueprint valorado', f"Reutiliza un blueprint que valoraste 👍 para «{memo['prompt'][:120]}».", catalog)
                if memo else rules_plan(prompt, catalog, retrieved, hint))
        result = compile_plan(base, catalog, hint)
    result.update({'engine': used, 'model': model, 'fallback_reason': reason, 'detected': hint,
                   'retrieved': ([{'type': 'memory', 'id': memo['id'], 'title': memo['title'], 'score': 1.0}] if memo else [])
                   + [{'type': r['type'], 'id': r['id'], 'title': r['title'], 'score': r['score']} for r in retrieved[:6]]})
    try:
        result['plan_id'] = store.remember(result, prompt)
    except OSError:
        result['plan_id'] = None   # memory is optional; the plan is still valid
    return result
