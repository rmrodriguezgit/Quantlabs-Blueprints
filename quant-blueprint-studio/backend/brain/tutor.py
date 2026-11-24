"""Cerebro Tutor: answers questions about tools, concepts and examples from local cards.

Retrieval is BM25 plus word coverage, question-type boost and exact-name match. With Ollama the
model sees only the question and the retrieved cards, and must cite them by id; citations that
were not provided are dropped. Without a model the answer is composed from the cards.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from . import llm
from .knowledge import BM25, fold, library, tokens

ROOT = Path(__file__).resolve().parents[2]
QUESTION_WORDS = set('que cual cuales como cuando donde por para sirve sirven significa hace hacen uso usar usa uso puedo debo es son el la los las un una de del en y o se me mi este esta esto nodo herramienta explica explicame dime what how why is are the does do which'.split())
INTENT = [(re.compile(r'^\s*(que|what)\s+(es|son|significa|is|are|mide)\b|\bsignifica\b|\bdefinicion\b|\bconcepto\b'), 'concept'),
          (re.compile(r'para que sirve|herramienta|nodo|\bcomo (uso|configuro|conecto)\b|which tool'), 'tool'),
          (re.compile(r'\bejemplo|\breceta|\bplantilla|\bblueprint|\brecorrido'), 'example')]
THIS = re.compile(r'\beste (nodo|paso)\b|\besta herramienta\b|\bthis node\b')
SYSTEM = """Eres el Cerebro Tutor de Quantlabs Blueprints, un estudio visual de investigación cuantitativa.
Responde en español, claro y breve (máx. 180 palabras), usando SOLO las FICHAS recibidas. Si no bastan, dilo.
Cita cada ficha que uses con su id entre corchetes, p. ej. [tool:features] o [concept:sharpe]. Usa **negritas** para términos clave.
No des asesoría financiera ni prometas rentabilidad.
Responde SOLO JSON: {"answer": "...", "sources": ["tool:..."], "follow_up": ["pregunta corta", "..."]}"""
SCHEMA = {'type': 'object', 'required': ['answer', 'sources', 'follow_up'], 'properties': {
    'answer': {'type': 'string'}, 'sources': {'type': 'array', 'items': {'type': 'string'}},
    'follow_up': {'type': 'array', 'items': {'type': 'string'}}}}


@lru_cache(maxsize=1)
def glossary() -> list[dict[str, Any]]:
    return json.loads((ROOT / 'src/renderer/data/glossary.json').read_text())['terms']


def corpus(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    docs = []
    for kind, d in catalog.items():
        params = [f"{f.get('label', f['key'])}" for f in d.get('fields', [])][:10]
        docs.append({'id': f'tool:{kind}', 'kind': 'tool', 'key': kind, 'title': d.get('label', kind), 'category': d.get('category', ''),
                     'body': d.get('subtitle', ''), 'input': d.get('input'), 'output': d.get('output'), 'parameters': params,
                     'aliases': [kind.replace('_', ' ')],
                     'text': ' '.join([kind.replace('_', ' '), d.get('label', ''), d.get('category', ''), d.get('subtitle', ''), ' '.join(params), d.get('hint', '')])})
    for term in glossary():
        docs.append({'id': f"concept:{term['id']}", 'kind': 'concept', 'key': term['id'], 'title': term['title'], 'body': term['body'],
                     'how': term.get('how', ''), 'tools': term.get('tools', []), 'aliases': term.get('aliases', []),
                     'text': ' '.join([term['title'], ' '.join(term.get('aliases', [])), term['body'], term.get('how', '')])})
    for entry in library():
        kinds = list(dict.fromkeys(n['data']['kind'] for n in entry['graph']['nodes']))
        docs.append({'id': f"example:{entry['id']}", 'kind': 'example', 'key': entry['id'], 'title': entry['title'], 'body': entry['description'],
                     'area': entry['area'], 'tools': kinds, 'aliases': [],
                     'text': ' '.join([entry['title'], entry['area'], entry['description'], ' '.join(entry.get('instructions', [])), ' '.join(kinds)])})
    return docs


def topic(question: str) -> str:
    words = re.findall(r'[a-z0-9_]+', fold(question))
    return ' '.join(w for w in words if w not in QUESTION_WORDS) or question


def retrieve(question: str, catalog: dict[str, Any], node_type: str | None = None) -> list[dict[str, Any]]:
    docs = corpus(catalog)
    bm25 = BM25(docs)
    folded, subject = fold(question), topic(question)
    padded = f" {subject.replace('-', ' ')} "
    wanted = {w for w in subject.split() if len(w) > 1}
    boost = {kind: 2.0 for pattern, kind in INTENT if pattern.search(folded)}
    ranked: list[dict[str, Any]] = []
    for kind, limit in (('tool', 4), ('concept', 4), ('example', 3)):
        found = bm25.search(subject, limit + 6, kind)
        for hit in found:
            words = set(tokens(hit['text']))
            coverage = sum(1 for w in wanted if w in words or w[:5] in words) / max(1, len(wanted))
            names = [hit['key'], hit['title'], *hit.get('aliases', [])]
            exact = any(f" {fold(str(n)).replace('_', ' ').replace('-', ' ').strip()} " in padded for n in names if len(str(n)) > 1)
            hit['score'] = round((coverage * 25 + hit['score']) * boost.get(kind, 1.0) + (12.0 if exact else 0.0), 3)
        ranked += sorted(found, key=lambda h: -h['score'])[:limit]
    if node_type and node_type in catalog and (THIS.search(folded) or not ranked):
        selected = next(d for d in docs if d['id'] == f'tool:{node_type}')
        ranked = [{**selected, 'score': 99.0}] + [h for h in ranked if h['id'] != selected['id']]
    return sorted(ranked, key=lambda h: -h['score'])


def rules_answer(ranked: list[dict[str, Any]], catalog: dict[str, Any]) -> tuple[str, list[str]]:
    if not ranked:
        return ('No encontré herramientas, conceptos ni ejemplos relacionados. Prueba con otras palabras '
                '(p. ej. «Sharpe», «fuga de información», «red neuronal», «Laya») o abre la biblioteca.'), []
    top, parts = ranked[0], []
    used = [top['id']]
    if top['kind'] == 'concept':
        parts.append(f"**{top['title']}**: {top['body']}")
        if top.get('how'):
            parts.append(f"Cómo leerlo: {top['how']}")
        tools = [catalog[k]['label'] for k in top.get('tools', []) if k in catalog][:5]
        if tools:
            parts.append('Herramientas relacionadas: ' + ', '.join(tools) + '.')
            used += [f'tool:{k}' for k in top.get('tools', []) if k in catalog][:2]
    elif top['kind'] == 'tool':
        parts.append(f"**{top['title']}** ({top['category']}): {top['body']}.")
        parts.append(f"Entrada: {top.get('input') or 'ninguna'} · Salida: {top.get('output') or 'ninguna'}.")
        if top['parameters']:
            parts.append('Parámetros principales: ' + ', '.join(top['parameters'][:8]) + '.')
        examples = [e for e in library() if top['key'] in {n['data']['kind'] for n in e['graph']['nodes']}][:3]
        if examples:
            parts.append('La usan: ' + '; '.join(e['title'] for e in examples) + '.')
            used += [f"example:{e['id']}" for e in examples]
    else:
        parts.append(f"**{top['title']}** ({top.get('area', '')}): {top['body']}")
        tools = [catalog[k]['label'] for k in top.get('tools', []) if k in catalog][:7]
        if tools:
            parts.append('Herramientas del flujo: ' + ' → '.join(tools) + '.')
    related = [h for h in ranked[1:] if h['kind'] in ('tool', 'concept') and h['id'] not in used][:3]
    if related:
        parts.append('Relacionado: ' + ', '.join(h['title'] for h in related) + '.')
        used += [h['id'] for h in related]
    return '\n\n'.join(parts), used


def suggestions(ranked: list[dict[str, Any]]) -> list[str]:
    out = []
    for doc in ranked[:4]:
        if doc['kind'] == 'tool':
            out.append(f"¿Cómo interpreto el resultado de {doc['title']}?")
        elif doc['kind'] == 'concept':
            out.append(f"¿Qué herramienta usa {doc['title'].split(' (')[0].lower()}?")
        else:
            out.append(f"¿Qué aprendo con «{doc['title']}»?")
    return list(dict.fromkeys(out))[:3]


def actions(ids: list[str], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    titles = {e['id']: e['title'] for e in library()}
    for identifier in ids:
        kind, _, key = identifier.partition(':')
        if kind == 'concept':
            term = next((t for t in glossary() if t['id'] == key), None)
            for tool in (term or {}).get('tools', [])[:2]:
                if tool in catalog:
                    out.setdefault(('add_tool', tool), {'type': 'add_tool', 'kind': tool, 'label': catalog[tool].get('label', tool)})
        elif kind == 'tool' and key in catalog:
            out[('add_tool', key)] = {'type': 'add_tool', 'kind': key, 'label': catalog[key].get('label', key)}
        elif kind == 'example' and key in titles:
            out[('open_example', key)] = {'type': 'open_example', 'id': key, 'label': titles[key]}
    return list(out.values())[:4]


def ask(question: str, catalog: dict[str, Any], engine: str = 'auto', node_type: str | None = None) -> dict[str, Any]:
    question = (question or '').strip()
    if not 2 <= len(question) <= 1000:
        raise ValueError('Escribe una pregunta de 2 a 1.000 caracteres.')
    if engine not in ('auto', 'ollama', 'rules'):
        raise ValueError('Motor: auto, ollama o rules.')
    ranked = retrieve(question, catalog, node_type)
    answer, used = rules_answer(ranked, catalog)
    out: dict[str, Any] = {'engine': 'rules', 'model': None, 'fallback_reason': None, 'follow_up': suggestions(ranked)}
    if engine in ('auto', 'ollama') and ranked:
        cards = [{k: v for k, v in d.items() if k not in ('text', 'score', 'key', 'aliases')} for d in ranked[:9]]
        try:
            value, model = llm.chat_json(SYSTEM, 'FICHAS:\n' + json.dumps(cards, ensure_ascii=False)[:9000] + '\n\nPREGUNTA:\n' + question, SCHEMA, timeout=120)
            text = str(value.get('answer', '')).strip()
            if not text:
                raise llm.LLMError('Respuesta vacía.')
            known = {d['id'] for d in ranked}
            cited = [s for s in value.get('sources', []) if isinstance(s, str) and s in known]
            cited += [s for s in re.findall(r'\[([a-z]+:[^\]\s]+)\]', text) if s in known and s not in cited]
            follow = [str(q).strip()[:120] for q in value.get('follow_up', []) if isinstance(q, str) and q.strip()][:3]
            answer, used = text[:1800], cited or used
            out.update(engine='ollama', model=model, follow_up=follow or out['follow_up'])
        except llm.LLMError as exc:
            if engine == 'ollama':
                raise ValueError(f'Qwen no disponible: {exc}') from exc
            out['fallback_reason'] = str(exc)
    by_id = {d['id']: d for d in ranked}
    sources = [{'id': i, 'kind': by_id[i]['kind'], 'title': by_id[i]['title']} for i in dict.fromkeys(used) if i in by_id]
    if not sources:
        sources = [{'id': d['id'], 'kind': d['kind'], 'title': d['title']} for d in ranked[:3]]
    return {**out, 'question': question, 'answer': answer, 'sources': sources, 'actions': actions(list(dict.fromkeys([s['id'] for s in sources] + used)), catalog)}
