"""BM25 retrieval over catalog cards and library blueprints, with accent folding."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STOP = set('de la el los las y o a en un una para con por del al que se su sus es como the of and to in for with on from mi me quiero hacer haz analiza analizar usa usar'.split())
# Spanish ↔ English bridges so either language retrieves the same cards.
SYNONYMS = {
    'precio': 'price close', 'precios': 'price close', 'rendimiento': 'return returns', 'rendimientos': 'return returns',
    'riesgo': 'risk var drawdown', 'volatilidad': 'volatility', 'cartera': 'portfolio', 'portafolio': 'portfolio markowitz',
    'red': 'network neural dense', 'neuronal': 'network neural', 'capas': 'layer dense', 'pronostico': 'forecast model predict',
    'prediccion': 'forecast predict model', 'indicadores': 'features rsi macd ema sma bollinger', 'grafico': 'plot chart',
    'graficar': 'plot chart', 'refuerzo': 'rl agent dqn q-learning', 'agente': 'agent rl', 'estrategia': 'backtest strategy',
    'limpieza': 'quality clean missing duplicates', 'calidad': 'quality clean', 'escalar': 'scale scaling', 'laya': 'laya decision typed',
    'decisiones': 'decision laya', 'pesos': 'weights calibrate', 'calibrar': 'calibrate weights', 'retroalimentacion': 'feedback online',
    'acciones': 'stock ticker yfinance', 'accion': 'stock ticker yfinance', 'mensual': 'monthly resample', 'histograma': 'histogram distribution',
    'comparar': 'compare', 'modelos': 'model models compare', 'clasicos': 'classical ml ridge forest', 'transformer': 'transformer attention',
    'lstm': 'lstm recurrent', 'recurrente': 'recurrent lstm gru', 'convolucional': 'cnn convolutional', 'generativo': 'gan vae generative',
}


def fold(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', text.lower()) if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    words = [w for w in re.findall(r'[a-z0-9_]+', fold(text)) if w not in STOP and len(w) > 1]
    return words + [s for w in words for s in SYNONYMS.get(w, '').split()]


class BM25:
    def __init__(self, docs: list[dict[str, Any]], k1: float = 1.4, b: float = .75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.toks = [tokens(d['text']) for d in docs]
        self.avg = sum(map(len, self.toks)) / max(len(self.toks), 1)
        df: dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + .5) / (c + .5)) for w, c in df.items()}

    def search(self, query: str, k: int = 6, kind: str | None = None) -> list[dict[str, Any]]:
        q = tokens(query)
        scored = []
        for doc, t in zip(self.docs, self.toks):
            if kind and doc.get('type', doc.get('kind')) != kind:
                continue
            tf: dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg)) for w in q if w in tf)
            if s > 0:
                scored.append((s, doc))
        return [{**d, 'score': round(s, 3)} for s, d in sorted(scored, key=lambda x: -x[0])[:k]]


@lru_cache(maxsize=1)
def library() -> list[dict[str, Any]]:
    return json.loads((ROOT / 'public/blueprints/library.json').read_text())['entries']


def tool_card(kind: str, d: dict[str, Any]) -> str:
    fields = '; '.join(f"{f['key']}={f.get('value')}" + (f" ∈ {{{', '.join(map(str, f['options']))}}}" if f.get('options') else '')
                       + (f" [{f.get('min')}–{f.get('max')}]" if f.get('min') is not None else '') for f in d.get('fields', []))
    return f"{kind} · {d.get('label')} — {d.get('subtitle')} ({d.get('category')}). in: {d.get('input') or '—'} → out: {d.get('output') or '—'}. params: {fields}"


def index(catalog: dict[str, Any]) -> BM25:
    docs = [{'type': 'tool', 'id': k, 'title': v.get('label', k), 'text': tool_card(k, v)} for k, v in catalog.items()]
    for e in library():
        kinds = ' '.join(n['data']['kind'] for n in e['graph']['nodes'])
        docs.append({'type': 'blueprint', 'id': e['id'], 'title': e['title'],
                     'text': ' '.join([e['title'], e['area'], e['description'], ' '.join(e.get('instructions', [])), kinds])})
    return BM25(docs)
