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

    if topic in POLYMARKET_RULES or (topic in ('laya_signal', 'laya_calibrate') and _is_polymarket(report)):
        return POLYMARKET_RULES[topic if topic in POLYMARKET_RULES else f'{topic}_polymarket'](report, m)
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


# ---------------- Polymarket readings ----------------

def _is_polymarket(report: dict[str, Any]) -> bool:
    return 'ventanas' in (report.get('metrics') or {}) or 'tramos' in (report.get('metrics') or {})


def _finish(summary: str, findings: list[str], cautions: list[str], steps: list[str]) -> dict[str, list[str] | str]:
    cautions.append('Señal de investigación sobre mercados de predicción; no es asesoría financiera.')
    return {'summary': summary, 'findings': findings or ['Sin hallazgos automáticos; revisa la tabla.'], 'cautions': cautions,
            'next_steps': steps or ['Cambia un parámetro a la vez y vuelve a ejecutar para ver su efecto.']}


def _pm_source(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    mk = report.get('market') or {}
    u = mk.get('underlying') or {}
    p_up = m.get('p_up')
    hist = m.get('p_up_historial')
    summary = f"Mercado «{mk.get('question') or report.get('selected')}» (ventana {mk.get('window', '—')}): P(UP) implícita {_pct(p_up) if p_up is not None else '—'} (precio medio del libro)."
    findings, cautions, steps = [], [], []
    spread = m.get('spread')
    if spread is not None:
        findings.append(f'Spread del libro {spread:.3f}: ' + ('estrecho, el precio es fiable.' if spread <= .02 else 'amplio; el precio implícito es impreciso.'))
        if spread > .05:
            cautions.append(f'Spread {spread:.3f} > 0.05: liquidez baja, P(UP) puede moverse mucho con poco volumen.')
    if hist is not None and p_up is not None and abs(hist - p_up) >= .03:
        findings.append(f'El último punto del historial CLOB ({_pct(hist)}) difiere del libro actual: el historial va con retraso; las decisiones usan el libro.')
    imb = m.get('imbalance')
    if imb is not None and abs(imb) >= .3:
        findings.append(f"Desequilibrio del libro {imb:+.2f}: más profundidad del lado {'comprador (UP)' if imb > 0 else 'vendedor (DOWN)'} cerca del precio.")
    if u.get('price_to_beat') is not None and u.get('spot') is not None:
        findings.append(f"{u.get('asset')} {u['spot']:.2f} frente a precio a vencer {u['price_to_beat']:.2f} ({u['diff']:+.2f}; {u['diff_pct']:+.3f} %): si cerrara ahora ganaría {u.get('leading')}.")
        sig = u.get('signal') or {}
        if sig.get('z') is not None:
            findings.append(f"Distancia normalizada z = {sig['z']:.2f} con {sig.get('minutes_left', 0):.1f} min restantes: " +
                            ('aún reversible.' if abs(sig['z']) < 1 else 'difícil de revertir si la volatilidad se mantiene.'))
    elif mk.get('window') and mk.get('slug', '').count('-updown-'):
        cautions.append('No se pudo obtener el precio del subyacente o el precio a vencer en esta ejecución.')
    if mk.get('closed'):
        cautions.append('El mercado está cerrado: el precio ya refleja la resolución.')
    steps.append('Conecta LAYA UP/DOWN (o Laya Signal → Laya Calibrate) para convertir estos datos en una decisión.')
    return _finish(summary, findings, cautions, steps)


def _pm_updown(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    d = report.get('decision') or {}
    p_up, conf = _num(d.get('probability_up')), _num(d.get('confidence'))
    bucket, total = d.get('bucket'), d.get('buckets_total')
    when = f' (minuto {bucket + 1} de {total})' if isinstance(bucket, int) and total else ''
    summary = (f"Decisión {d.get('decision', '—')} para la ventana {d.get('horizon', '—')}{when}: P(UP) {_pct(p_up) if p_up is not None else '—'}, "
               f"confianza {_pct(conf) if conf is not None else '—'}" + (' (accionable).' if d.get('actionable') else ' (por debajo del umbral: sin señal).'))
    findings, cautions, steps = [], [], []
    allc = {k: v for k, v in (d.get('contributions') or {}).items() if _num(v) is not None}
    bias = allc.pop('bias', 0.0)
    contrib = allc
    if contrib:
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
        k, v = top[0]
        findings.append(f"La señal que más pesó fue «{k}» ({v:+.2f} en log-odds, hacia {'UP' if v > 0 else 'DOWN'}).")
        if abs(bias) > abs(v):
            findings.append(f"El sesgo aprendido ({bias:+.2f}) pesa más que cualquier señal: la decisión refleja sobre todo la tasa base de UP de la calibración, no información de esta ventana.")
        against = [kk for kk, vv in top if vv * (1 if (p_up or .5) >= .5 else -1) < -0.2]
        if against:
            findings.append('Empujaron en contra de la decisión: ' + ', '.join(against) + '.')
    pm_ = _num(d.get('p_market'))
    if pm_ is not None and p_up is not None:
        if (pm_ >= .5) != (p_up >= .5):
            cautions.append(f'La decisión contradice al mercado (P mercado {_pct(pm_)}): sólo vale la pena si la calibración muestra ventaja frente al mercado.')
        else:
            findings.append(f'Coincide con el mercado (P mercado {_pct(pm_)}); la diferencia de probabilidad es {abs(p_up - pm_) * 100:.1f} puntos.')
    lay = d.get('laya') or {}
    if lay.get('used') and _num(lay.get('P_up')) is not None and lay['P_up'] > .75:
        cautions.append(f"Laya da P(UP) {_pct(lay['P_up'])}; en las pruebas Laya tiende a responder UP casi siempre, por eso su peso calibrado es bajo.")
    stats = d.get('bucket_stats') or []
    row = next((r for r in stats if r.get('minuto') == (bucket + 1 if isinstance(bucket, int) else None)), None)
    if row and _num(row.get('ventaja vs mercado')) is not None:
        adv = row['ventaja vs mercado']
        findings.append(f"En TE, en este minuto, la calibrada tuvo log-loss {row.get('logloss calibrada')} frente a {row.get('logloss mercado')} del mercado: "
                        + ('ventaja a favor del modelo.' if adv > 0 else 'el mercado fue más preciso; no hay ventaja demostrada en este minuto.'))
        if (row.get('TE n') or 0) < 60:
            cautions.append(f"Sólo {row.get('TE n')} ventanas de prueba en este minuto: la ventaja puede ser azar.")
    learn = d.get('learning') or {}
    if learn:
        hr, mh = _num(learn.get('hit_rate')), _num(learn.get('market_hit_rate'))
        if hr is not None and mh is not None:
            findings.append(f"Aprendizaje en línea: acierto {_pct(hr)} vs mercado {_pct(mh)} en {min(learn.get('updates', 0), learn.get('window', 50))} decisiones resueltas.")
        else:
            findings.append(f"Aprendizaje en línea: {learn.get('updates', 0)} decisiones resueltas, {len(learn.get('pending') or [])} pendientes; aún sin métricas.")
    steps.append('Deja el ciclo en vivo varias ventanas y compara acierto y log-loss frente al mercado por minuto.')
    return _finish(summary, findings, cautions, steps)


def _pm_signal(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    rows = [r for r in (report.get('table') or []) if isinstance(r.get('minuto'), int)]
    base = m.get('tasa_base_up')
    summary = f"Laya respondió {int(m.get('ventanas') or 0)} ventanas resueltas en {int(m.get('tramos') or 1)} momento(s) de decisión; tasa base UP {_pct(base) if base is not None else '—'}."
    findings, cautions = [], []
    for r in rows:
        vals = {k: r.get(f'acierto {k}', 0) for k in ('mercado', 'Laya', 'distancia')}
        top = max(vals.values())
        best = ' = '.join(k for k, v in vals.items() if abs(v - top) < 1e-9)
        findings.append(f"Minuto {r['minuto']}: mercado {_pct(r['acierto mercado'])}, Laya {_pct(r['acierto Laya'])}, distancia {_pct(r['acierto distancia'])} "
                        f"({'empate: ' if ' = ' in best else 'mejor: '}{best}).")
    if base is not None:
        laya_rows = [r['acierto Laya'] for r in rows]
        if laya_rows and max(abs(a - max(base, 1 - base)) for a in laya_rows) < .03:
            cautions.append('El acierto de Laya es casi igual a la tasa base: probablemente responde siempre lo mismo.')
    return _finish(summary, findings[:6], cautions, ['Conecta Laya Calibrate para ajustar los pesos por minuto con estos resultados reales.'])


def _pm_calibrate(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    rows = [r for r in (report.get('table') or []) if isinstance(r.get('minuto'), int)]
    findings, cautions = [], []
    mk, cal = m.get('te_logloss_mercado'), m.get('te_logloss_calibrada')
    summary = (f"Calibración con resultados reales en {int(m.get('tramos') or 1)} minuto(s): log-loss TE total {cal:.3f} calibrada vs {mk:.3f} mercado "
               f"({'mejor que el mercado' if cal < mk else 'sin ventaja frente al mercado'})." if mk is not None and cal is not None else 'Calibración con resultados reales.')
    good = [r for r in rows if (r.get('ventaja vs mercado') or 0) > 0]
    if rows:
        best = max(rows, key=lambda r: r.get('ventaja vs mercado') or -9)
        findings.append(f"Mayor ventaja en el minuto {best['minuto']}: log-loss {best['logloss calibrada']} vs {best['logloss mercado']} del mercado.")
        findings.append(f'La calibrada tiene menor log-loss que el mercado en {len(good)} de {len(rows)} minutos.')
        worse_acc = [r['minuto'] for r in good if (r.get('acierto calibrada') or 0) < (r.get('acierto mercado') or 0)]
        if worse_acc:
            cautions.append('En los minutos ' + ', '.join(map(str, worse_acc)) + ' la calibrada tiene mejor log-loss pero peor acierto que el mercado: '
                            'sus probabilidades son más prudentes, no más certeras.')
        stuck = [r['minuto'] for r in rows if r.get('convergió') == 'no']
        if stuck:
            cautions.append('En los minutos ' + ', '.join(map(str, stuck)) + ' la pérdida de validación aún bajaba al final: el ajuste no convergió; sube épocas o lr.')
        late = rows[-1]
        findings.append(f"En el último minuto el acierto del mercado es {_pct(late.get('acierto mercado', 0))}: cerca del cierre el precio ya incorpora casi todo.")
    acc_m, acc_c = m.get('te_accuracy_mercado'), m.get('te_accuracy_calibrada')
    if acc_m is not None and acc_c is not None:
        findings.append(f'Acierto TE total {_pct(acc_c)} calibrada vs {_pct(acc_m)} mercado.')
    splits = report.get('splits') or {}
    if 0 < (splits.get('test') or 0) < 60:
        cautions.append(f"Sólo {splits['test']} ventanas de prueba: diferencias de pocos puntos pueden ser azar.")
    return _finish(summary, findings, cautions, ['Usa LAYA UP/DOWN con «calibrated» u «online» y vigila la ventaja por minuto en vivo.'])


def _pm_market_board(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    d = report.get('dashboard') or {}
    mk, k = d.get('market') or {}, d.get('kpis') or {}
    up, down = mk.get('up_outcome', 'UP'), mk.get('down_outcome', 'DOWN')
    p = _num(k.get('p_up'))
    money = lambda v: '—' if v is None else (f'${v / 1e6:.2f} M' if v >= 1e6 else f'${v / 1e3:.1f} k' if v >= 1e3 else f'${v:.2f}')
    summary = f"«{mk.get('question')}»: el mercado da {_pct(p) if p is not None else '—'} a «{up}» según el libro actual."
    findings, cautions = [], []
    sp = _num(k.get('spread'))
    if sp is not None:
        findings.append(f'Spread {sp:.3f}: ' + ('precio fiable.' if sp <= .02 else 'amplio, la probabilidad es aproximada.'))
    pr, vol = _num(k.get('presión_up_pct')), _num(k.get('volumen_reciente_usd'))
    if pr is not None and vol:
        side = up if pr >= .5 else down
        findings.append(f"En las {k.get('operaciones')} operaciones recientes ({money(vol)}) el {max(pr, 1 - pr) * 100:.0f} % del dinero empujó hacia «{side}».")
    c1, c24 = _num(k.get('cambio_1h')), _num(k.get('cambio_24h'))
    if c24 is not None:
        findings.append(f"Cambio de «{up}»: {c1 * 100 if c1 is not None else 0:+.1f} pts en 1 h y {c24 * 100:+.1f} pts en 24 h (historial CLOB).")
    imb = _num(k.get('desequilibrio'))
    if imb is not None and abs(imb) >= .3:
        findings.append(f"El libro a ±5¢ tiene más profundidad del lado {'comprador' if imb > 0 else 'vendedor'} de «{up}» ({imb:+.2f}).")
    u = mk.get('underlying') or {}
    if u.get('diff') is not None:
        findings.append(f"{u.get('asset')} {u['diff']:+.2f} $ respecto al precio a vencer: si cerrara ahora ganaría {u.get('leading')}.")
    sib = [s for s in d.get('siblings') or [] if not s.get('cerrado')]
    if len(sib) > 1:
        top = sib[0]
        findings.append(f"El evento tiene {len(sib)} mercados abiertos; el favorito es «{top['mercado']}» ({top['prob_primero'] * 100:.1f} %).")
    for h in d.get('holders') or []:
        if (h.get('mayor_tenedor_pct_top') or 0) > .5:
            cautions.append(f"En «{h['outcome']}» un solo tenedor tiene más de la mitad de lo que suman los 20 mayores: posición muy concentrada.")
    if (k.get('operaciones') or 0) < 20:
        cautions.append('Pocas operaciones recientes: el flujo y el VWAP son poco representativos.')
    cautions.append('Gamma y el historial CLOB pueden ir con retraso respecto al libro; la probabilidad mostrada es la del libro.')
    return _finish(summary, findings, cautions, ['Conecta LAYA UP/DOWN después de Polymarket Market para convertir estos datos en una decisión.'])


def _pm_dashboard(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    d = report.get('dashboard') or {}
    if d.get('mode') == 'market':
        return _pm_market_board(report, m)
    k = d.get('kpis') or {}
    money = lambda v: '—' if v is None else (f'${v / 1e6:.2f} M' if v >= 1e6 else f'${v / 1e3:.1f} k' if v >= 1e3 else f'${v:.0f}')
    summary = (f"{k.get('eventos', 0)} eventos y {k.get('mercados', 0)} mercados abiertos ({(d.get('filters') or {}).get('category', 'todas')}): "
               f"volumen 24 h {money(k.get('volumen_24h'))}, interés abierto {money(k.get('interés_abierto'))}.")
    findings, cautions = [], []
    cats = d.get('categories') or []
    if cats and k.get('volumen_24h'):
        c0 = cats[0]
        findings.append(f"«{c0['categoría']}» concentra {money(c0['volumen_24h'])} ({c0['volumen_24h'] / k['volumen_24h'] * 100:.0f} % del volumen 24 h).")
    conc = k.get('concentración_top10')
    if conc is not None:
        findings.append(f'Los 10 eventos más activos suman el {conc * 100:.0f} % del volumen: ' + ('actividad muy concentrada.' if conc > .6 else 'actividad repartida.'))
    for key, text in (('movers', 'Mayor movimiento 24 h'), ('contested', 'Más disputado con volumen'), ('top_markets', 'Mercado más negociado')):
        rows = d.get(key) or []
        if rows:
            r = rows[0]
            extra = f" ({r['cambio_24h'] * 100:+.1f} pts)" if key == 'movers' and r.get('cambio_24h') is not None else f" (líder «{r.get('outcome_líder')}» {r.get('prob_líder', 0) * 100:.0f} %)"
            findings.append(f"{text}: «{r['mercado']}»{extra}.")
    mx = k.get('mercados_por_evento_max')
    if mx and k.get('mercados') and mx > .2 * k['mercados']:
        cautions.append(f'Un solo evento tiene {mx} mercados: los conteos de mercados (p. ej. «cierran en 24 h») están dominados por pocos eventos con muchas apuestas derivadas.')
    sp = k.get('spread_mediano')
    if sp is not None and sp > .05:
        cautions.append(f'Spread mediano {sp:.2f}: muchos mercados tienen poca liquidez; sus probabilidades son imprecisas.')
    cautions.append('Instantánea del momento de la consulta; cubre sólo las páginas descargadas, no todo Polymarket.')
    return _finish(summary, findings, cautions, ['Elige una categoría o sube «Páginas» para ampliar la cobertura; usa ● En vivo para seguirlo.'])


def _yf_data(report: dict[str, Any], m: dict[str, float | None]) -> dict[str, list[str] | str]:
    tr, vol, dd = m.get('retorno_total'), m.get('volatilidad_anual'), m.get('max_drawdown')
    summary = f"{report.get('selected')}: {int(m.get('barras') or 0)} barras descargadas de Yahoo Finance."
    findings, cautions = [], []
    if tr is not None:
        findings.append(f'Retorno total en el rango: {_pct(tr)} (de la primera a la última barra).')
    if vol is not None:
        findings.append(f'Volatilidad anualizada {_pct(vol)}' + (' (alta).' if vol > .4 else '.'))
    if dd is not None:
        findings.append(f'Caída máxima {_pct(dd)} desde un máximo previo.')
        if dd < -.2:
            cautions.append(f'Caída máxima {_pct(dd)}: periodo de pérdidas importantes en la muestra.')
    if (m.get('barras') or 0) < 60:
        cautions.append('Menos de 60 barras: pocas para entrenar o validar un modelo temporal.')
    notes = ' '.join(report.get('notes') or [])
    if 'volumen 0' in notes:
        cautions.append('Hay muchas barras sin volumen: no uses VWAP ni indicadores de volumen con esta serie.')
    cautions.append('Datos de Yahoo Finance sin garantía de exactitud; no es asesoría financiera.')
    return {'summary': summary, 'findings': findings, 'cautions': cautions,
            'next_steps': ['Conecta Technical Features o Data Quality para preparar la serie antes de modelar.']}


POLYMARKET_RULES = {'yfinance_data': _yf_data, 'polymarket_dashboard': _pm_dashboard, 'polymarket_source': _pm_source, 'laya_updown': _pm_updown,
                    'laya_signal_polymarket': _pm_signal, 'laya_calibrate_polymarket': _pm_calibrate}


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
