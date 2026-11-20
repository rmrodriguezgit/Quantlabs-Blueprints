"""Explain one result: deterministic reading always, Qwen reading on demand with number checking.

The model receives only the aggregated report (metrics, a trimmed table, notes), never data rows.
Numbers in its text that do not match the report are returned as ``unverified``.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from . import llm

SYSTEM = """Eres el tutor de Quantlabs Blueprints. Explica en español, para un estudiante, el resultado de UN nodo.
Usa SOLO las cifras del resumen que recibes; no inventes cifras. Sé prudente: un backtest o una métrica fuera de muestra no prometen rentabilidad.
Responde SOLO JSON: {"summary": "2-3 frases", "findings": ["..."], "cautions": ["..."], "next_steps": ["..."]} (máx. 4 elementos por lista)."""
SCHEMA = {'type': 'object', 'required': ['summary', 'findings', 'cautions', 'next_steps'], 'properties': {
    'summary': {'type': 'string'}, 'findings': {'type': 'array', 'items': {'type': 'string'}},
    'cautions': {'type': 'array', 'items': {'type': 'string'}}, 'next_steps': {'type': 'array', 'items': {'type': 'string'}}}}


def _num(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _pct(x: float) -> str:
    return f'{x * 100:.1f} %'


def compact(report: dict[str, Any]) -> dict[str, Any]:
    table = report.get('table') or []
    return {'title': report.get('title'), 'topic': report.get('topic'), 'selected': report.get('selected'),
            'splits': report.get('splits'), 'metrics': {k: v for k, v in (report.get('metrics') or {}).items() if _num(v) is not None or isinstance(v, str)},
            'table': [{k: v for k, v in list(row.items())[:8]} for row in table[:12]], 'notes': (report.get('notes') or [])[:6]}


def rules(report: dict[str, Any]) -> dict[str, list[str] | str]:
    m = {k: _num(v) for k, v in (report.get('metrics') or {}).items()}
    topic, findings, cautions, steps = str(report.get('topic', '')), [], [], []
    summary = f"{report.get('title', 'Resultado')}: selección «{report.get('selected', '—')}»."

    def get(*keys: str) -> float | None:
        return next((m[k] for k in keys if m.get(k) is not None), None)

    if topic == 'laya_signal':
        acc = get('acierto_crudo_up_down')
        summary = f"Laya respondió {int(get('dias') or 0)} días; su elección up/down coincidió con el mercado el {_pct(acc) if acc is not None else '—'} de las veces."
        if acc is not None and abs(acc - .5) < .03:
            findings.append('El acierto está a menos de 3 puntos del 50 %: equivale a lanzar una moneda en la dirección diaria.')
        conf = get('confianza_media')
        if conf is not None and conf < .6:
            findings.append(f'Confianza media {conf:.2f}: Laya duda entre opciones; respuestas poco concentradas.')
        steps.append('Conecta Laya Calibrate para medir y corregir la calibración en datos fuera de muestra.')
    elif topic == 'laya_calibrate':
        for metric, better in (('logloss', 'menor'), ('brier', 'menor'), ('ece', 'menor'), ('accuracy', 'mayor')):
            raw, cal = get(f'te_{metric}_cruda'), get(f'te_{metric}_calibrada')
            if raw is not None and cal is not None:
                improved = cal < raw if better == 'menor' else cal > raw
                findings.append(f"TE {metric}: {raw:.3f} → {cal:.3f} ({'mejora' if improved else 'empeora'}; {better} es mejor).")
        base, acc = get('tasa_base_tr'), get('te_accuracy_calibrada')
        if base is not None and acc is not None and acc <= max(base, 1 - base):
            cautions.append(f'Acierto calibrado {_pct(acc)} no supera la tasa base {_pct(max(base, 1 - base))}: probabilidades más honestas no implican señal.')
        summary = 'Ajuste de pesos sobre las respuestas de Laya, evaluado una sola vez en TE.'
        steps.append('Prueba otra redacción de las preguntas en Laya Signal y compara el logloss TE.')
    elif topic == 'laya_feedback':
        s, b = get('retorno_estrategia'), get('retorno_buy_hold')
        if s is not None and b is not None:
            summary = f"Estrategia con retroalimentación {_pct(s)} frente a {_pct(b)} de comprar y mantener."
            findings.append('La estrategia ' + ('superó' if s > b else 'no superó') + ' a comprar y mantener en el periodo evaluado.')
        l1, l2 = get('logloss_1a_mitad'), get('logloss_2a_mitad')
        if l1 is not None and l2 is not None:
            findings.append(f"Logloss {l1:.3f} → {l2:.3f}: la retroalimentación {'aprende' if l2 < l1 else 'no mejora'} con el tiempo.")
        ab = get('abstencion')
        if ab is not None and ab > .5:
            cautions.append(f'Se abstuvo el {_pct(ab)} de los días: baja la confianza mínima o la métrica es poco representativa.')
    else:
        for key in ('total_return', 'sharpe', 'max_drawdown', 'rmse', 'mse', 'mae', 'r2', 'accuracy', 'directional_accuracy'):
            for k, v in m.items():
                if v is not None and key in k.lower() and len(findings) < 5:
                    findings.append(f'{k} = {v:.4g}.')
    sharpe, dd = get('sharpe', 'Sharpe'), get('max_drawdown')
    if sharpe is not None and sharpe < 0:
        cautions.append(f'Sharpe negativo ({sharpe:.2f}): el rendimiento no compensa el riesgo en esta muestra.')
    if dd is not None and dd < -.2:
        cautions.append(f'Caída máxima {_pct(dd)}: riesgo de pérdida alto.')
    splits = report.get('splits') or {}
    if isinstance(splits, dict) and 0 < (splits.get('test') or 0) < 60:
        cautions.append(f"Solo {splits['test']} observaciones de prueba: las métricas tienen mucha varianza.")
    cautions.append('Resultado educativo sobre datos históricos; no es asesoría financiera.')
    if not steps:
        steps.append('Cambia un parámetro a la vez y vuelve a ejecutar para ver su efecto en las métricas.')
    return {'summary': summary, 'findings': findings or ['Sin hallazgos automáticos para este tipo de nodo; revisa la tabla.'], 'cautions': cautions, 'next_steps': steps}


def _numbers(text: str) -> list[float]:
    return [float(x.replace(',', '.')) for x in re.findall(r'-?\d+(?:[.,]\d+)?', text)]


def verify(texts: list[str], report: dict[str, Any]) -> list[str]:
    known: list[float] = []
    def walk(v: Any) -> None:
        if _num(v) is not None:
            known.extend([v, v * 100])
        elif isinstance(v, dict):
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
        elif isinstance(v, str):
            known.extend(_numbers(v))
    walk(compact(report))
    bad = []
    for text in texts:
        for n in _numbers(text):
            if abs(n) <= 12 and float(n).is_integer():
                continue   # counts, list ordinals, small integers
            if not any(abs(n - k) <= max(.006, abs(k) * .006) or round(k, 1) == n or round(k, 2) == n for k in known):
                bad.append(f'{n:g}')
    return list(dict.fromkeys(bad))


def explain(report: dict[str, Any], engine: str = 'auto') -> dict[str, Any]:
    if not isinstance(report, dict) or not report.get('metrics') and not report.get('table'):
        raise ValueError('El resultado no tiene métricas ni tablas que explicar.')
    reading = rules(report)
    out = {'rules': reading, 'engine': 'rules', 'model': None, 'unverified': [], 'fallback_reason': None, **reading}
    if engine == 'rules':
        return out
    try:
        value, model = llm.chat_json(SYSTEM, json.dumps({'resultado': compact(report), 'lectura_por_reglas': reading}, ensure_ascii=False, default=str), SCHEMA, timeout=120)
        answer = {'summary': str(value.get('summary', ''))[:800]}
        for key in ('findings', 'cautions', 'next_steps'):
            answer[key] = [str(x)[:400] for x in value.get(key, []) if isinstance(x, (str, int, float))][:4]
        if not answer['summary']:
            raise llm.LLMError('Respuesta vacía.')
        texts = [answer['summary'], *answer['findings'], *answer['cautions'], *answer['next_steps']]
        out.update(answer, engine='ollama', model=model, unverified=verify(texts, report))
    except llm.LLMError as exc:
        if engine == 'ollama':
            raise ValueError(f'Qwen no disponible: {exc}') from exc
        out['fallback_reason'] = str(exc)
    return out
