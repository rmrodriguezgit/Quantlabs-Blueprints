"""ML taxonomy tools: the families of the "Machine Learning Algorithms" map not covered elsewhere.

Unsupervised (frame in → frame out, fitted on the first ``fit_fraction`` rows only, applied to all):
  ml_cluster  · K-Means, DBSCAN, Hierarchical (agglomerative)
  ml_reduce   · PCA, ICA, t-SNE, UMAP
  ml_assoc    · Apriori, FP-Growth over discretised market events (up/down, high volume…)
  ml_anomaly  · Isolation Forest, Z-Score, One-Class SVM
Semi-supervised (needs «Dividir TR/VA/TE» with a classification target):
  ml_semi     · Self-Training, Co-Training, Label Propagation; only a fraction of TR keeps its label
Self-supervised (needs «Dividir TR/VA/TE»):
  ml_selfsup  · Contrastive (SimCLR-style), Masked modelling, Autoencoder; pretext on TR features,
                then a linear probe on the embedding vs the raw features (selection VA, TE once)
Reinforcement learning (frame with Close):
  ml_rl       · model-free Q-Learning, DQN, PPO; model-based Dyna-Q, MCTS on a model fitted on TR

Supervised classification/regression (Naive Bayes, Decision Tree, XGBoost…) live in ml_pipeline_nodes.
All fitted statistics use the training rows only; reports follow the common study format.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from .ml_pipeline_nodes import SPLITS, _attrs, _counts, _meta, _names, _report, _score, _x

Progress = Callable[[dict[str, Any]], None] | None


# ---------------------------------------------------------------- helpers
def _numeric(frame: pd.DataFrame, p: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    cols = _names(p.get('features'))
    if not cols:
        skip = {'Split', 'Target', 'Prediction', 'Proba', 'Cluster', 'Anomaly', 'AnomalyScore'}
        cols = [c for c in frame.select_dtypes('number').columns if c not in skip and not str(c).startswith(('Comp_', 'Emb_'))]
    lost = [c for c in cols if c not in frame.columns]
    if lost:
        raise ValueError(f'Columnas inexistentes: {", ".join(lost)}.')
    if not cols:
        raise ValueError('No hay columnas numéricas: conecta datos o indica «features».')
    x = frame[cols].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('Hay faltantes o infinitos: conecta «Limpieza de datos» antes.')
    return cols, x


def _fit_rows(frame: pd.DataFrame, p: dict[str, Any]) -> np.ndarray:
    """Boolean mask of rows used to fit: TR if a split exists, else the first fit_fraction rows."""
    if 'Split' in frame:
        mask = (frame['Split'] == 'TR').to_numpy()
        if mask.sum() >= 10:
            return mask
    frac = float(p.get('fit_fraction', 0.7))
    if not 0.2 <= frac <= 1:
        raise ValueError('Fracción de ajuste entre 0.2 y 1.')
    mask = np.zeros(len(frame), bool)
    mask[:max(10, int(len(frame) * frac))] = True
    return mask


def _standardize(x: np.ndarray, fit: np.ndarray) -> np.ndarray:
    mu, sd = x[fit].mean(0), x[fit].std(0)
    return (x - mu) / np.where(sd > 0, sd, 1.0)


def _sample(rows: list[dict[str, Any]], n: int = 1500) -> list[dict[str, Any]]:
    step = max(1, len(rows) // n)
    return rows[::step]


def _done(frame: pd.DataFrame, out: pd.DataFrame, topic: str, **body: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    out.attrs = _attrs(frame, **body.pop('ml', {}))
    fit_n = int(body.pop('fit_n', 0))
    splits = _counts(frame) if 'Split' in frame else {'train': fit_n, 'validation': 0, 'test': 0}
    return out, _report(topic, splits=splits, **body)


# ---------------------------------------------------------------- clustering
CLUSTERING = ('K-Means', 'DBSCAN', 'Hierarchical')


def cluster(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.neighbors import KNeighborsClassifier
    algorithm = str(p.get('algorithm', 'K-Means'))
    if algorithm not in (*CLUSTERING, 'Compare'):
        raise ValueError(f'Algoritmo de clustering: Compare, {", ".join(CLUSTERING)}.')
    k, eps, minpts, seed = int(p.get('clusters', 3)), float(p.get('eps', 1.0)), int(p.get('min_samples', 10)), int(p.get('seed', 42))
    if not 2 <= k <= 20:
        raise ValueError('Número de clusters entre 2 y 20.')
    cols, raw = _numeric(frame, p)
    fit = _fit_rows(frame, p)
    x = _standardize(raw, fit)
    if fit.sum() > 20000:
        raise ValueError('Máximo 20 000 filas de ajuste para clustering.')
    rows, best, labels_all, chosen = [], -np.inf, None, ''
    for name in (CLUSTERING if algorithm == 'Compare' else (algorithm,)):
        if name == 'K-Means':
            model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(x[fit])
            lab = model.predict(x)
        else:
            model = (DBSCAN(eps=eps, min_samples=minpts) if name == 'DBSCAN' else AgglomerativeClustering(n_clusters=k, linkage=str(p.get('linkage', 'ward'))))
            fit_lab = model.fit_predict(x[fit])
            core = fit_lab >= 0
            if core.sum() < 2 or len(set(fit_lab[core])) < 1:
                lab = np.full(len(x), -1)
            else:  # out-of-sample rows take the label of their nearest fitted neighbour (fit rows keep theirs)
                lab = KNeighborsClassifier(n_neighbors=1).fit(x[fit][core], fit_lab[core]).predict(x)
                lab[np.where(fit)[0]] = fit_lab
        fl = lab[fit]
        valid = fl >= 0
        n_cl = len(set(fl[valid]))
        sc = {'clusters': n_cl, 'ruido %': float(100 * (~valid).mean())}
        if n_cl >= 2:
            sc |= {'silhouette': float(silhouette_score(x[fit][valid], fl[valid], sample_size=min(3000, int(valid.sum())), random_state=seed)),
                   'Davies-Bouldin': float(davies_bouldin_score(x[fit][valid], fl[valid])),
                   'Calinski-Harabasz': float(calinski_harabasz_score(x[fit][valid], fl[valid]))}
        rows.append({'algoritmo': name, **sc})
        if sc.get('silhouette', -np.inf) > best or labels_all is None:
            best, labels_all, chosen = sc.get('silhouette', -np.inf), lab, name
        if progress:
            progress({'detail': f'{name}: {n_cl} clusters'})
    out = frame.copy()
    out['Cluster'] = labels_all.astype(int)
    profile = out.loc[fit].groupby('Cluster')[cols].mean().reset_index()
    profile.insert(1, 'filas', out.loc[fit].groupby('Cluster').size().to_numpy())
    prof_rows = [{k: (float(v) if isinstance(v, (float, np.floating)) else int(v) if isinstance(v, (int, np.integer)) else v) for k, v in r.items()} for r in profile.to_dict('records')]
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=seed).fit(x[fit]).transform(x) if x.shape[1] >= 2 else np.column_stack([x[:, 0], np.zeros(len(x))])
    pts = _sample([{'PC1': float(a), **{f'cluster {c}': float(b) if c == l else None for c in sorted(set(labels_all))}} for a, b, l in zip(xy[:, 0], xy[:, 1], labels_all)])
    charts = [{'title': f'{chosen} · clusters en el plano PCA (ajuste)', 'x_key': 'PC1', 'series': [f'cluster {c}' for c in sorted(set(labels_all))], 'rows': pts, 'chart_type': 'scatter'}]
    if 'Close' in out:
        charts.append({'title': 'Precio coloreado por cluster', 'x_key': 'x', 'series': [f'cluster {c}' for c in sorted(set(labels_all))],
                       'rows': _sample([{'x': _x(i), **{f'cluster {c}': float(v) if c == l else None for c in sorted(set(labels_all))}} for i, v, l in zip(out.index, out['Close'], labels_all)]), 'chart_type': 'scatter'})
    if len(rows) > 1:
        charts.append({'title': 'Comparación · silhouette (más alto es mejor)', 'x_key': 'algoritmo', 'series': ['silhouette'], 'rows': rows, 'chart_type': 'bar'})
    notes = [f'Rasgos estandarizados con {int(fit.sum())} filas de ajuste; el resto recibe la etiqueta del vecino ajustado más cercano (K-Means: centroide).',
             'Sin etiquetas: los clusters son agrupaciones descriptivas (p. ej. regímenes), no clases verdaderas.']
    if chosen == 'DBSCAN':
        notes.append('DBSCAN marca ruido con −1; ajusta eps / min_samples si todo cae en un cluster o en ruido.')
    return _done(frame, out, 'ml_cluster', selected=chosen, scope='no supervisado · ajuste en TR', features=cols, table=prof_rows,
                 metrics={k: v for k, v in next(r for r in rows if r['algoritmo'] == chosen).items() if k != 'algoritmo'},
                 charts=charts, notes=notes, comparison=rows, fit_n=int(fit.sum()), ml={'clustering': chosen})


# ---------------------------------------------------------------- dimensionality reduction
REDUCERS = ('PCA', 'ICA', 't-SNE', 'UMAP')


def reduce(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    algorithm, k, seed = str(p.get('algorithm', 'PCA')), int(p.get('components', 2)), int(p.get('seed', 42))
    if algorithm not in REDUCERS:
        raise ValueError(f'Reducción: {", ".join(REDUCERS)}.')
    cols, raw = _numeric(frame, p)
    if not 1 <= k <= min(10, len(cols)):
        raise ValueError(f'Componentes entre 1 y {min(10, len(cols))} (nº de rasgos).')
    fit = _fit_rows(frame, p)
    x = _standardize(raw, fit)
    metrics: dict[str, float] = {}
    table: list[dict[str, Any]] = []
    notes: list[str] = []
    if algorithm == 'PCA':
        from sklearn.decomposition import PCA
        m = PCA(n_components=k, random_state=seed).fit(x[fit])
        z = m.transform(x)
        rec = m.inverse_transform(z)
        metrics = {'varianza explicada': float(m.explained_variance_ratio_.sum()), 'MSE reconstrucción': float(np.mean((rec - x) ** 2))}
        table = [{'componente': f'Comp_{i + 1}', 'varianza %': float(100 * v), **{c: float(w) for c, w in zip(cols, m.components_[i])}} for i, v in enumerate(m.explained_variance_ratio_)]
    elif algorithm == 'ICA':
        from sklearn.decomposition import FastICA
        m = FastICA(n_components=k, random_state=seed, max_iter=1000, whiten='unit-variance').fit(x[fit])
        z = m.transform(x)
        metrics = {'MSE reconstrucción': float(np.mean((m.inverse_transform(z) - x) ** 2)),
                   'curtosis media |exceso|': float(np.mean(np.abs(pd.DataFrame(z[fit]).kurt().to_numpy())))}
        table = [{'componente': f'Comp_{i + 1}', **{c: float(w) for c, w in zip(cols, m.mixing_[:, i])}} for i in range(k)]
        notes.append('ICA busca fuentes estadísticamente independientes (no gaussianas); la tabla muestra la matriz de mezcla.')
    else:
        if len(x) > 6000:
            raise ValueError(f'{algorithm}: máximo 6 000 filas (coste cuadrático).')
        if algorithm == 't-SNE':
            from sklearn.manifold import TSNE
            z = TSNE(n_components=k if k <= 3 else 2, perplexity=float(p.get('perplexity', 30)), random_state=seed, init='pca').fit_transform(x)
            notes.append('t-SNE no tiene transform(): se ajusta sobre todas las filas y sólo sirve para visualizar, nunca como rasgo de un modelo predictivo.')
        else:
            try:
                import umap
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ValueError('UMAP requiere el paquete umap-learn (pip install umap-learn).') from exc
            m = umap.UMAP(n_components=k, n_neighbors=int(p.get('neighbors', 15)), random_state=seed).fit(x[fit])
            z = m.transform(x)
            notes.append('UMAP se ajusta en TR y transforma el resto; preserva estructura local y parte de la global.')
        from sklearn.manifold import trustworthiness
        idx = np.where(fit)[0][:2000]
        metrics = {'trustworthiness (k=10)': float(trustworthiness(x[idx], z[idx], n_neighbors=10))}
    out = frame.copy()
    for i in range(z.shape[1]):
        out[f'Comp_{i + 1}'] = z[:, i]
    rows = [{'Comp_1': float(a), 'Comp_2': float(b) if z.shape[1] > 1 else 0.0} for a, b in zip(z[:, 0], z[:, 1] if z.shape[1] > 1 else z[:, 0])]
    charts = [{'title': f'{algorithm} · proyección 2D', 'x_key': 'Comp_1', 'series': ['Comp_2'], 'rows': _sample(rows), 'chart_type': 'scatter'}]
    if algorithm == 'PCA':
        charts.append({'title': 'Varianza explicada por componente', 'x_key': 'componente', 'series': ['varianza %'], 'rows': table, 'chart_type': 'bar'})
    notes.insert(0, f'{len(cols)} rasgos → {z.shape[1]} componentes; estandarización con {int(fit.sum())} filas de ajuste.')
    return _done(frame, out, 'ml_reduce', selected=algorithm, scope='no supervisado · reducción', features=cols, table=table,
                 metrics=metrics, charts=charts, notes=notes, fit_n=int(fit.sum()), ml={'reduction': algorithm})


# ---------------------------------------------------------------- association rules
def _events(frame: pd.DataFrame, p: dict[str, Any], fit: np.ndarray) -> list[frozenset[str]]:
    cols, raw = _numeric(frame, p)
    q = float(p.get('quantile', 0.33))
    lo, hi = np.quantile(raw[fit], q, axis=0), np.quantile(raw[fit], 1 - q, axis=0)
    baskets = []
    for t in range(len(raw)):
        items = {f'{c}↑' for c, v, h in zip(cols, raw[t], hi) if v > h} | {f'{c}↓' for c, v, l in zip(cols, raw[t], lo) if v < l}
        if p.get('next_day', True) and 'LogReturn' in frame and t + 1 < len(raw):
            items.add('MAÑANA sube' if frame['LogReturn'].iloc[t + 1] > 0 else 'MAÑANA baja')
        baskets.append(frozenset(items))
    return baskets


def _apriori(baskets: list[frozenset[str]], support: float, max_len: int) -> dict[frozenset[str], int]:
    n, need = len(baskets), support * len(baskets)
    counts = Counter(i for b in baskets for i in b)
    level = {frozenset([i]): c for i, c in counts.items() if c >= need}
    found = dict(level)
    k = 2
    while level and k <= max_len:
        keys = list(level)
        cands = {a | b for a, b in itertools.combinations(keys, 2) if len(a | b) == k}
        cands = {c for c in cands if all(frozenset(s) in level for s in itertools.combinations(c, k - 1))}  # Apriori pruning
        cnt = Counter()
        for b in baskets:
            for c in cands:
                if c <= b:
                    cnt[c] += 1
        level = {c: v for c, v in cnt.items() if v >= need}
        found |= level
        k += 1
    del n
    return found


def _fpgrowth(baskets: list[frozenset[str]], support: float, max_len: int) -> dict[frozenset[str], int]:
    need = support * len(baskets)
    freq = Counter(i for b in baskets for i in b)
    order = {i: r for r, (i, c) in enumerate(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))) if c >= need}

    def mine(trans: list[tuple[list[str], int]], suffix: frozenset[str], out: dict[frozenset[str], int]) -> None:
        # conditional pattern bases: count items, then recurse on each frequent item (FP-tree flattened as weighted paths)
        cnt: Counter = Counter()
        for path, w in trans:
            for i in path:
                cnt[i] += w
        for item, c in cnt.items():
            if c < need:
                continue
            pattern = suffix | {item}
            out[pattern] = c
            if len(pattern) >= max_len:
                continue
            cond = []
            for path, w in trans:
                if item in path:
                    prefix = path[:path.index(item)]
                    if prefix:
                        cond.append((prefix, w))
            if cond:
                mine(cond, pattern, out)

    paths = Counter(tuple(sorted((i for i in b if i in order), key=order.get)) for b in baskets)
    result: dict[frozenset[str], int] = {}
    mine([(list(k), v) for k, v in paths.items() if k], frozenset(), result)
    return result


def association(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    algorithm = str(p.get('algorithm', 'Apriori'))
    if algorithm not in ('Apriori', 'FP-Growth'):
        raise ValueError('Reglas de asociación: Apriori o FP-Growth.')
    support, conf, max_len = float(p.get('min_support', 0.05)), float(p.get('min_confidence', 0.55)), int(p.get('max_len', 3))
    if not 0.005 <= support <= 0.9 or not 0.1 <= conf <= 1 or not 2 <= max_len <= 4:
        raise ValueError('Soporte 0.005–0.9, confianza 0.1–1 y longitud 2–4.')
    fit = _fit_rows(frame, p)
    baskets = _events(frame, p, fit)
    tr, te = [b for b, f in zip(baskets, fit) if f], [b for b, f in zip(baskets, fit) if not f]
    sets = (_apriori if algorithm == 'Apriori' else _fpgrowth)(tr, support, max_len)
    n = len(tr)
    supp = {k: v / n for k, v in sets.items()}
    rules = []
    for itemset, s in supp.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for ante in map(frozenset, itertools.combinations(itemset, r)):
                cons = itemset - ante
                if ante not in supp or cons not in supp:
                    continue
                c = s / supp[ante]
                if c < conf:
                    continue
                hold = [b for b in te if ante <= b]
                rules.append({'si': ' + '.join(sorted(ante)), 'entonces': ' + '.join(sorted(cons)), 'soporte': s, 'confianza': c,
                              'lift': c / supp[cons], 'confianza fuera de muestra': (sum(cons <= b for b in hold) / len(hold)) if hold else None,
                              'casos fuera de muestra': len(hold)})
    rules.sort(key=lambda r: (-r['lift'], -r['confianza']))
    rules = rules[:int(p.get('top', 30))]
    out = frame.copy()
    out['Items'] = [', '.join(sorted(b)) for b in baskets]
    notes = [f'{algorithm}: {len(sets)} conjuntos frecuentes en {n} sesiones de ajuste; {len(rules)} reglas con confianza ≥ {conf}.',
             'Cada sesión es una «cesta» de eventos: rasgo↑ / rasgo↓ si supera los cuantiles de ajuste' + (' y el resultado de la sesión siguiente.' if p.get('next_day', True) else '.'),
             'Lift > 1 indica co-ocurrencia mayor que la independencia; la confianza fuera de muestra comprueba si se mantiene después.']
    charts = [{'title': 'Reglas · lift (top 15)', 'x_key': 'regla', 'series': ['lift'], 'rows': [{'regla': f"{r['si']} → {r['entonces']}", 'lift': r['lift']} for r in rules[:15]], 'chart_type': 'bar'}] if rules else []
    return _done(frame, out, 'ml_assoc', selected=algorithm, scope='no supervisado · reglas', features=[], table=rules, charts=charts, notes=notes,
                 metrics={'conjuntos frecuentes': len(sets), 'reglas': len(rules), 'lift máximo': rules[0]['lift'] if rules else 0.0}, fit_n=n, ml={'association': algorithm})


# ---------------------------------------------------------------- anomaly detection
ANOMALY = ('Isolation Forest', 'Z-Score', 'One-Class SVM')


def anomaly(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    algorithm, cont, seed = str(p.get('algorithm', 'Isolation Forest')), float(p.get('contamination', 0.02)), int(p.get('seed', 42))
    if algorithm not in (*ANOMALY, 'Compare'):
        raise ValueError(f'Anomalías: Compare, {", ".join(ANOMALY)}.')
    if not 0.001 <= cont <= 0.3:
        raise ValueError('Contaminación entre 0.001 y 0.3.')
    cols, raw = _numeric(frame, p)
    fit = _fit_rows(frame, p)
    x = _standardize(raw, fit)
    scores: dict[str, np.ndarray] = {}
    for name in (ANOMALY if algorithm == 'Compare' else (algorithm,)):
        if name == 'Z-Score':
            s = np.abs(x).max(1)
        elif name == 'Isolation Forest':
            s = -IsolationForest(n_estimators=200, contamination=cont, random_state=seed).fit(x[fit]).score_samples(x)
        else:
            if fit.sum() > 8000:
                raise ValueError('One-Class SVM: máximo 8 000 filas de ajuste.')
            s = -OneClassSVM(nu=cont, gamma='scale').fit(x[fit]).score_samples(x)
        scores[name] = s
    chosen = algorithm if algorithm != 'Compare' else 'Isolation Forest'
    s = scores[chosen]
    thr = float(p.get('z_threshold', 3.0)) if chosen == 'Z-Score' else float(np.quantile(s[fit], 1 - cont))
    flag = s > thr
    out = frame.copy()
    out['AnomalyScore'], out['Anomaly'] = s, flag.astype(int)
    rows = [{'algoritmo': k, 'anómalas ajuste %': float(100 * (v[fit] > (float(p.get('z_threshold', 3.0)) if k == 'Z-Score' else np.quantile(v[fit], 1 - cont))).mean()),
             'anómalas fuera %': float(100 * (v[~fit] > (float(p.get('z_threshold', 3.0)) if k == 'Z-Score' else np.quantile(v[fit], 1 - cont))).mean()) if (~fit).any() else None} for k, v in scores.items()]
    if len(scores) > 1:
        names = list(scores)
        for a, b in itertools.combinations(names, 2):
            rows.append({'algoritmo': f'rango {a} vs {b}', 'anómalas ajuste %': float(pd.Series(scores[a]).corr(pd.Series(scores[b]), method='spearman')), 'anómalas fuera %': None})
    y = out['Close'] if 'Close' in out else out[cols[0]]
    charts = [{'title': f'{chosen} · serie con anomalías', 'x_key': 'x', 'series': ['valor', 'anomalía'],
               'rows': _sample([{'x': _x(i), 'valor': float(v), 'anomalía': float(v) if f else None} for i, v, f in zip(out.index, y, flag)], 2500), 'chart_type': 'scatter'},
              {'title': 'Puntuación de anomalía', 'x_key': 'puntuación', 'series': ['puntuación'], 'rows': [{'puntuación': float(v)} for v in s], 'chart_type': 'histogram'}]
    top = out.loc[flag].assign(_s=s[flag]).sort_values('_s', ascending=False).head(20)
    table = [{'fecha': _x(i), 'puntuación': float(r['_s']), **{c: float(r[c]) for c in cols[:6]}} for i, r in top.iterrows()]
    notes = [f'Umbral {"|z| > " + str(thr) if chosen == "Z-Score" else f"cuantil {1 - cont:.3f} de la puntuación en ajuste"}; {int(flag.sum())} filas marcadas.',
             'Ajuste y estandarización sólo con las filas de ajuste; fuera de muestra se aplica el mismo umbral.']
    if len(scores) > 1:
        notes.append('Compare muestra el % marcado por cada método y la correlación de rangos entre puntuaciones; la columna Anomaly usa Isolation Forest.')
    return _done(frame, out, 'ml_anomaly', selected=chosen, scope='no supervisado · anomalías', features=cols, table=table or rows, charts=charts, notes=notes,
                 metrics={'anómalas': int(flag.sum()), 'umbral': thr}, comparison=rows, fit_n=int(fit.sum()), ml={'anomaly': chosen})


# ---------------------------------------------------------------- semi-supervised
SEMI = ('Self-Training', 'Co-Training', 'Label Propagation')


def _co_training(xa: np.ndarray, xb: np.ndarray, y: np.ndarray, labeled: np.ndarray, rounds: int, per_round: int, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    y = y.copy()
    lab = labeled.copy()
    for _ in range(rounds):
        ma, mb = LogisticRegression(max_iter=2000).fit(xa[lab], y[lab]), GaussianNB().fit(xb[lab], y[lab])
        pool = np.where(~lab)[0]
        if not len(pool):
            break
        for m, view in ((ma, xa), (mb, xb)):  # each view labels its most confident examples for the other
            pool = np.where(~lab)[0]
            if not len(pool):
                break
            pr = m.predict_proba(view[pool])
            pick = pool[np.argsort(-pr.max(1))[:per_round]]
            y[pick] = m.classes_[pr[np.argsort(-pr.max(1))[:per_round]].argmax(1)]
            lab[pick] = True

    class Both:
        classes_ = ma.classes_

        def predict_proba(self, x: np.ndarray) -> np.ndarray:
            h = xa.shape[1]
            return (ma.predict_proba(x[:, :h]) + mb.predict_proba(x[:, h:])) / 2

        def predict(self, x: np.ndarray) -> np.ndarray:
            return self.classes_[self.predict_proba(x).argmax(1)]
    del seed
    return Both()


def semi(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier
    ml = _meta(frame)
    if ml['task'] != 'classification':
        raise ValueError('Semi-supervisado requiere objetivo de clasificación: en «Dividir» usa target_mode = direction.')
    algorithm, frac, seed = str(p.get('algorithm', 'Compare')), float(p.get('labeled_fraction', 0.1)), int(p.get('seed', 42))
    if algorithm not in (*SEMI, 'Compare'):
        raise ValueError(f'Semi-supervisado: Compare, {", ".join(SEMI)}.')
    if not 0.01 <= frac <= 0.9:
        raise ValueError('Fracción etiquetada entre 0.01 y 0.9.')
    feats = ml['features']
    s = frame['Split']
    X = {k: frame.loc[s == k, feats].to_numpy(float) for k in SPLITS}
    Y = {k: frame.loc[s == k, 'Target'].to_numpy(int) for k in SPLITS}
    rng = np.random.default_rng(seed)
    labeled = rng.random(len(Y['TR'])) < frac
    if len(set(Y['TR'][labeled])) < 2:
        raise ValueError('Muy pocas etiquetas: sube la fracción etiquetada.')
    half = max(1, len(feats) // 2)
    models: dict[str, Any] = {'Sólo etiquetadas (referencia)': LogisticRegression(max_iter=2000).fit(X['TR'][labeled], Y['TR'][labeled]),
                              'Todas las etiquetas (techo)': LogisticRegression(max_iter=2000).fit(X['TR'], Y['TR'])}
    yu = np.where(labeled, Y['TR'], -1)
    for name in (SEMI if algorithm == 'Compare' else (algorithm,)):
        if name == 'Self-Training':
            models[name] = SelfTrainingClassifier(LogisticRegression(max_iter=2000), threshold=float(p.get('threshold', 0.6))).fit(X['TR'], yu)
        elif name == 'Label Propagation':
            if len(X['TR']) > 10000:
                raise ValueError('Label Propagation: máximo 10 000 filas TR.')
            models[name] = LabelSpreading(kernel='knn', n_neighbors=int(p.get('neighbors', 10)), alpha=0.2, max_iter=100).fit(X['TR'], yu)
        else:
            if len(feats) < 2:
                raise ValueError('Co-Training necesita ≥ 2 rasgos para formar dos vistas.')
            models[name] = _co_training(X['TR'][:, :half], X['TR'][:, half:], Y['TR'], labeled, int(p.get('rounds', 10)), max(1, int(len(Y['TR']) * 0.02)), seed)
        if progress:
            progress({'detail': name})
    rows, best, chosen = [], -np.inf, ''
    for name, m in models.items():
        pr = m.predict_proba(X['VA'])[:, 1]
        sc = _score('classification', Y['VA'], (pr >= 0.5).astype(float), pr)
        rows.append({'método': name, **{f'VA {k}': v for k, v in sc.items()}})
        if name in SEMI and sc['Accuracy'] > best:
            best, chosen = sc['Accuracy'], name
    win = models[chosen]
    pr_te = win.predict_proba(X['TE'])[:, 1] if len(X['TE']) else np.array([])
    te = _score('classification', Y['TE'], (pr_te >= 0.5).astype(float), pr_te) if len(X['TE']) > 1 else {}
    out = frame.copy()
    out['Proba'] = win.predict_proba(out[feats].to_numpy(float))[:, 1]
    out['Prediction'] = (out['Proba'] >= 0.5).astype(float)
    notes = [f'Se conservan {int(labeled.sum())} de {len(labeled)} etiquetas TR ({100 * frac:.0f} %); el resto se trata como no etiquetado.',
             'Referencia = sólo etiquetadas; techo = todas las etiquetas. VA elige el método semi-supervisado y TE se mide una vez.',
             'Co-Training divide los rasgos en dos vistas (primera/segunda mitad): Logística y Naive Bayes se etiquetan mutuamente.']
    return _done(frame, out, 'ml_semi', selected=chosen, scope='semi-supervisado · selección VA · TE', features=feats, table=rows,
                 charts=[{'title': 'VA Accuracy por método', 'x_key': 'método', 'series': ['VA Accuracy'], 'rows': rows, 'chart_type': 'bar'}],
                 notes=notes, metrics={f'TE {k}': v for k, v in te.items()}, ml={'algorithm': chosen})


# ---------------------------------------------------------------- self-supervised
SELFSUP = ('Contrastive', 'Masked Modeling', 'Autoencoder')


def _windows(x: np.ndarray, w: int) -> np.ndarray:
    pad = np.vstack([np.repeat(x[:1], w - 1, 0), x])
    return np.stack([pad[i:i + w] for i in range(len(x))])  # causal window ending at each row


def selfsup(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    import torch
    from torch import nn
    from sklearn.linear_model import LogisticRegression, Ridge
    ml = _meta(frame)
    algorithm, w, dim = str(p.get('algorithm', 'Contrastive')), int(p.get('window', 10)), int(p.get('embedding', 8))
    epochs, seed = int(p.get('epochs', 30)), int(p.get('seed', 42))
    if algorithm not in SELFSUP:
        raise ValueError(f'Auto-supervisado: {", ".join(SELFSUP)}.')
    if not 2 <= w <= 60 or not 2 <= dim <= 64 or not 1 <= epochs <= 300:
        raise ValueError('Ventana 2–60, embedding 2–64 y épocas 1–300.')
    torch.manual_seed(seed)
    feats, task = ml['features'], ml['task']
    s = frame['Split'].to_numpy()
    xw = torch.tensor(_windows(frame[feats].to_numpy(float), w), dtype=torch.float32)
    tr = torch.tensor(s == 'TR')
    d_in = xw.shape[1] * xw.shape[2]
    enc = nn.Sequential(nn.Flatten(), nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, dim))
    head = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, d_in)) if algorithm != 'Contrastive' else nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=float(p.get('learning_rate', 1e-3)))
    data, mask_p, tau = xw[tr], float(p.get('mask_ratio', 0.3)), 0.2
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(data))
        total = 0.0
        for i in range(0, len(data), 128):
            b = data[perm[i:i + 128]]
            if algorithm == 'Contrastive':  # SimCLR: two noisy, masked views of the same window are positives
                v1 = b * (torch.rand_like(b) > 0.1) + 0.1 * torch.randn_like(b)
                v2 = b * (torch.rand_like(b) > 0.1) + 0.1 * torch.randn_like(b)
                z1, z2 = nn.functional.normalize(head(enc(v1)), dim=1), nn.functional.normalize(head(enc(v2)), dim=1)
                z = torch.cat([z1, z2])
                sim = z @ z.T / tau
                sim.fill_diagonal_(-1e9)
                n = len(b)
                target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)])
                loss = nn.functional.cross_entropy(sim, target)
            elif algorithm == 'Masked Modeling':  # BERT-style: hide entries, predict them
                m = torch.rand_like(b) < mask_p
                rec = head(enc(b.masked_fill(m, 0.0))).view_as(b)
                loss = ((rec - b) ** 2)[m].mean() if m.any() else ((rec - b) ** 2).mean()
            else:
                loss = ((head(enc(b)).view_as(b) - b) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        hist.append({'época': ep + 1, 'pérdida pretexto': total / len(data)})
        if progress:
            progress({'epoch': ep + 1, 'epochs': epochs, 'loss': hist[-1]['pérdida pretexto']})
    with torch.no_grad():
        emb = enc(xw).numpy()
    y = frame['Target'].to_numpy(float)
    rows, probes = [], {}
    for name, feat in (('rasgos brutos (ventana)', xw.flatten(1).numpy()), (f'embedding {algorithm}', emb)):
        mdl = LogisticRegression(max_iter=3000) if task == 'classification' else Ridge(alpha=1.0)
        mdl.fit(feat[s == 'TR'], y[s == 'TR'].astype(int) if task == 'classification' else y[s == 'TR'])
        probes[name] = (mdl, feat)
        for k in ('VA', 'TE'):
            idx = s == k
            if idx.sum() < 2:
                continue
            if task == 'classification':
                pr = mdl.predict_proba(feat[idx])[:, 1]
                sc = _score(task, y[idx], (pr >= 0.5).astype(float), pr)
            else:
                sc = _score(task, y[idx], mdl.predict(feat[idx]))
            rows.append({'sonda lineal': name, 'conjunto': k, **sc})
    key = 'Accuracy' if task == 'classification' else 'RMSE'
    va = {r['sonda lineal']: r[key] for r in rows if r['conjunto'] == 'VA'}
    chosen = (max if task == 'classification' else min)(va, key=va.get)
    mdl, feat = probes[chosen]
    out = frame.copy()
    for i in range(emb.shape[1]):
        out[f'Emb_{i + 1}'] = emb[:, i]
    out['Prediction'] = mdl.predict(feat).astype(float)
    notes = [f'Pretexto {algorithm} sin etiquetas, sólo con ventanas TR de {w} filas × {len(feats)} rasgos → embedding {dim}.',
             'Evaluación estándar: sonda lineal congelada sobre el embedding frente a la misma sonda sobre rasgos brutos; VA elige, TE se informa.']
    return _done(frame, out, 'ml_selfsup', selected=chosen, scope='auto-supervisado · sonda lineal', features=feats, table=rows,
                 charts=[{'title': 'Pérdida de la tarea pretexto (TR)', 'x_key': 'época', 'series': ['pérdida pretexto'], 'rows': hist, 'chart_type': 'line'},
                         {'title': f'Sonda lineal · {key}', 'x_key': 'sonda lineal', 'series': [key], 'rows': [r for r in rows if r['conjunto'] == 'VA'], 'chart_type': 'bar'}],
                 notes=notes, metrics={f'TE {k}': v for r in rows if r['conjunto'] == 'TE' and r['sonda lineal'] == chosen for k, v in r.items() if isinstance(v, float)},
                 ml={'algorithm': chosen})


# ---------------------------------------------------------------- reinforcement learning
RL_FREE, RL_BASED = ('Q-Learning', 'DQN', 'PPO'), ('Dyna-Q', 'MCTS')
ACTIONS = np.array([-1.0, 0.0, 1.0])  # short / cash / long


class _Env:
    """Daily trading: state at t from causal features, action = position held over t→t+1, reward net of costs."""

    def __init__(self, feats: np.ndarray, ret: np.ndarray, cost: float, bins: int) -> None:
        self.x, self.r, self.cost = feats, ret, cost
        self.edges = [np.quantile(feats[:, j], np.linspace(0, 1, bins + 1)[1:-1]) for j in range(feats.shape[1])]
        self.bins = bins

    def disc(self, t: int, pos: int) -> tuple[int, ...]:
        return (*(int(np.searchsorted(e, self.x[t, j])) for j, e in enumerate(self.edges)), pos)

    def reward(self, t: int, prev: int, a: int) -> float:
        return float(100 * (ACTIONS[a] * self.r[t] - self.cost * abs(ACTIONS[a] - ACTIONS[prev])))


def _run_policy(env: _Env, idx: np.ndarray, policy: Callable[[int, int], int]) -> tuple[np.ndarray, list[int]]:
    pos, rews, acts = 1, [], []
    for t in idx[:-1]:
        a = policy(int(t), pos)
        rews.append(env.reward(int(t), pos, a) / 100)
        acts.append(a)
        pos = a
    return np.array(rews), acts


def _tabular(env: _Env, idx: np.ndarray, episodes: int, rng: np.random.Generator, gamma: float, plan_steps: int = 0) -> Callable[[int, int], int]:
    Q: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
    model: dict[tuple, tuple[float, tuple]] = {}
    for ep in range(episodes):
        eps, pos = max(0.05, 1 - ep / max(1, episodes * 0.7)), 1
        for t in idx[:-1]:
            s = env.disc(int(t), pos)
            a = int(rng.integers(3)) if rng.random() < eps else int(Q[s].argmax())
            r, s2 = env.reward(int(t), pos, a), env.disc(int(t) + 1, a)
            Q[s][a] += 0.1 * (r + gamma * Q[s2].max() - Q[s][a])
            if plan_steps:  # Dyna-Q: learn a (deterministic, last-seen) model and replay simulated transitions
                model[(s, a)] = (r, s2)
                keys = list(model)
                for _ in range(plan_steps):
                    ps, pa = keys[rng.integers(len(keys))]
                    pr, ps2 = model[(ps, pa)]
                    Q[ps][pa] += 0.1 * (pr + gamma * Q[ps2].max() - Q[ps][pa])
            pos = a
    return lambda t, pos: int(Q[env.disc(t, pos)].argmax()) if env.disc(t, pos) in Q else 1


def _mcts(env: _Env, tr_idx: np.ndarray, rng: np.random.Generator, sims: int, depth: int, gamma: float) -> Callable[[int, int], int]:
    """Model-based planning: the model is the empirical TR distribution of next return and next state given the discrete
    market state. UCT search over `depth` steps from each decision point; no look-ahead into real future prices."""
    trans: dict[tuple, list[tuple[float, tuple]]] = defaultdict(list)
    for t in tr_idx[:-1]:
        m, m2 = env.disc(int(t), 0)[:-1], env.disc(int(t) + 1, 0)[:-1]
        trans[m].append((float(env.r[t]), m2))
    every = [v for vs in trans.values() for v in vs]

    def step(m: tuple, pos: int, a: int) -> tuple[float, tuple]:
        opts = trans.get(m) or every
        r, m2 = opts[rng.integers(len(opts))]
        return float(100 * (ACTIONS[a] * r - env.cost * abs(ACTIONS[a] - ACTIONS[pos]))), m2

    def plan(t: int, pos: int) -> int:
        N: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
        W: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(3))
        root = (env.disc(t, 0)[:-1], pos)
        for _ in range(sims):
            node, path, g, disc = root, [], 0.0, 1.0
            for _d in range(depth):
                n = N[node]
                ucb = np.where(n == 0, np.inf, W[node] / np.maximum(n, 1) + 1.4 * np.sqrt(np.log(n.sum() + 1) / np.maximum(n, 1)))
                a = int(np.argmax(ucb + 1e-9 * rng.random(3)))
                r, m2 = step(node[0], node[1], a)
                path.append((node, a, g, disc))
                g += disc * r
                disc *= gamma
                node = (m2, a)
            for nd, a, g0, d0 in path:  # back up the return from that node onwards
                N[nd][a] += 1
                W[nd][a] += (g - g0) / d0
        return int(np.argmax(N[root]))
    return plan


def _deep(env: _Env, idx: np.ndarray, algorithm: str, episodes: int, gamma: float, seed: int, progress: Progress) -> Callable[[int, int], int]:
    import torch
    from torch import nn
    torch.manual_seed(seed)
    d = env.x.shape[1] + 3

    def state(t: int, pos: int) -> torch.Tensor:
        return torch.tensor(np.concatenate([env.x[t], np.eye(3)[pos]]), dtype=torch.float32)

    if algorithm == 'DQN':
        net, tgt = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3)), nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 3))
        tgt.load_state_dict(net.state_dict())
        opt, buf, rng, step = torch.optim.Adam(net.parameters(), 1e-3), [], np.random.default_rng(seed), 0
        for ep in range(episodes):
            eps, pos = max(0.05, 1 - ep / max(1, episodes * 0.7)), 1
            for t in idx[:-1]:
                s = state(int(t), pos)
                a = int(rng.integers(3)) if rng.random() < eps else int(net(s).argmax())
                buf.append((s, a, env.reward(int(t), pos, a), state(int(t) + 1, a)))
                buf = buf[-10000:]
                pos, step = a, step + 1
                if len(buf) >= 128 and step % 4 == 0:
                    batch = [buf[i] for i in rng.integers(len(buf), size=64)]
                    S, A = torch.stack([b[0] for b in batch]), torch.tensor([b[1] for b in batch])
                    R, S2 = torch.tensor([b[2] for b in batch], dtype=torch.float32), torch.stack([b[3] for b in batch])
                    with torch.no_grad():  # Double DQN target
                        y = R + gamma * tgt(S2).gather(1, net(S2).argmax(1, keepdim=True)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(net(S).gather(1, A[:, None]).squeeze(1), y)
                    opt.zero_grad(); loss.backward(); opt.step()
                if step % 200 == 0:
                    tgt.load_state_dict(net.state_dict())
            if progress:
                progress({'epoch': ep + 1, 'epochs': episodes})
        return lambda t, pos: int(net(state(t, pos)).argmax())
    # PPO (clipped surrogate) with a shared-body actor-critic, one rollout per episode over TR
    body = nn.Sequential(nn.Linear(d, 64), nn.Tanh())
    pi, v = nn.Linear(64, 3), nn.Linear(64, 1)
    opt = torch.optim.Adam([*body.parameters(), *pi.parameters(), *v.parameters()], 3e-4)
    for ep in range(episodes):
        S, A, R, LP, pos = [], [], [], [], 1
        with torch.no_grad():
            for t in idx[:-1]:
                s = state(int(t), pos)
                dist = torch.distributions.Categorical(logits=pi(body(s)))
                a = int(dist.sample())
                S.append(s); A.append(a); LP.append(float(dist.log_prob(torch.tensor(a)))); R.append(env.reward(int(t), pos, a))
                pos = a
            S_, A_, LP_ = torch.stack(S), torch.tensor(A), torch.tensor(LP)
            vals = v(body(S_)).squeeze(1)
            adv, gae, nxt = torch.zeros(len(R)), 0.0, 0.0
            for i in reversed(range(len(R))):  # GAE(λ=0.95)
                delta = R[i] + gamma * nxt - float(vals[i])
                gae = delta + gamma * 0.95 * gae
                adv[i], nxt = gae, float(vals[i])
            ret = adv + vals
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(4):
            for i in range(0, len(R), 256):
                sl = slice(i, i + 256)
                h = body(S_[sl])
                dist = torch.distributions.Categorical(logits=pi(h))
                ratio = torch.exp(dist.log_prob(A_[sl]) - LP_[sl])
                loss = -torch.min(ratio * adv[sl], ratio.clamp(0.8, 1.2) * adv[sl]).mean() + 0.5 * ((v(h).squeeze(1) - ret[sl]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); opt.step()
        if progress:
            progress({'epoch': ep + 1, 'epochs': episodes})
    return lambda t, pos: int(pi(body(state(t, pos))).argmax())


def _stats(r: np.ndarray, acts: list[int]) -> dict[str, float]:
    eq = np.cumprod(1 + r)
    sharpe = float(np.sqrt(252) * r.mean() / r.std()) if r.std() > 0 else 0.0
    return {'ROI %': float(100 * (eq[-1] - 1)) if len(eq) else 0.0, 'Sharpe': sharpe,
            'Max drawdown %': float(100 * (eq / np.maximum.accumulate(eq) - 1).min()) if len(eq) else 0.0,
            'largo %': float(100 * np.mean(np.array(acts) == 2)) if acts else 0.0, 'corto %': float(100 * np.mean(np.array(acts) == 0)) if acts else 0.0}


def rl(frame: pd.DataFrame, p: dict[str, Any], progress: Progress = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    algorithm = str(p.get('algorithm', 'Compare'))
    family = str(p.get('family', 'model-free'))
    allowed = RL_FREE + RL_BASED
    if algorithm not in (*allowed, 'Compare'):
        raise ValueError(f'RL: Compare, {", ".join(allowed)}.')
    if 'Close' not in frame:
        raise ValueError('RL necesita la columna Close (conecta datos de mercado).')
    episodes, gamma, seed, cost = int(p.get('episodes', 20)), float(p.get('gamma', 0.95)), int(p.get('seed', 42)), float(p.get('fee_bps', 5)) / 1e4
    if not 1 <= episodes <= 300 or not 0 < gamma < 1:
        raise ValueError('Episodios 1–300 y gamma en (0, 1).')
    ret = np.log(frame['Close']).diff().shift(-1).fillna(0).to_numpy()  # return earned holding from t to t+1
    lr = np.log(frame['Close']).diff().fillna(0)
    feats_df = pd.DataFrame({'mom5': lr.rolling(5).sum(), 'mom20': lr.rolling(20).sum(), 'vol20': lr.rolling(20).std()}, index=frame.index)
    extra = [c for c in _names(p.get('features')) if c in frame]
    feats_df = pd.concat([feats_df, frame[extra]], axis=1).bfill().fillna(0)
    n = len(frame)
    tr_n, va_n = int(n * float(p.get('train_ratio', 0.7))), int(n * float(p.get('val_ratio', 0.15)))
    x = feats_df.to_numpy(float)
    x = (x - x[:tr_n].mean(0)) / np.where(x[:tr_n].std(0) > 0, x[:tr_n].std(0), 1)
    env = _Env(x, ret, cost, int(p.get('bins', 3)))
    env.edges = [np.quantile(x[:tr_n, j], np.linspace(0, 1, env.bins + 1)[1:-1]) for j in range(x.shape[1])]
    idx = {'TR': np.arange(0, tr_n), 'VA': np.arange(tr_n, tr_n + va_n), 'TE': np.arange(tr_n + va_n, n)}
    if min(len(v) for v in idx.values()) < 20:
        raise ValueError('Muy pocas filas: cada conjunto necesita ≥ 20 sesiones.')
    rng = np.random.default_rng(seed)
    if algorithm == 'Compare':
        cands = RL_FREE if family == 'model-free' else RL_BASED if family == 'model-based' else allowed
    else:
        cands = (algorithm,)
    policies: dict[str, Callable[[int, int], int]] = {'Comprar y mantener': lambda t, pos: 2, 'Efectivo': lambda t, pos: 1}
    for name in cands:
        if progress:
            progress({'detail': f'entrenando {name}'})
        if name == 'Q-Learning':
            policies[name] = _tabular(env, idx['TR'], episodes, rng, gamma)
        elif name == 'Dyna-Q':
            policies[name] = _tabular(env, idx['TR'], episodes, rng, gamma, plan_steps=int(p.get('planning_steps', 10)))
        elif name == 'MCTS':
            policies[name] = _mcts(env, idx['TR'], rng, int(p.get('simulations', 60)), int(p.get('depth', 5)), gamma)
        else:
            policies[name] = _deep(env, idx['TR'], name, episodes, gamma, seed, progress)
    rows, curves = [], {}
    for name, pol in policies.items():
        for k in ('VA', 'TE'):
            r, acts = _run_policy(env, idx[k], pol)
            rows.append({'política': name, 'tipo': 'model-based' if name in RL_BASED else 'model-free' if name in RL_FREE else 'referencia', 'conjunto': k, **_stats(r, acts)})
            if k == 'TE':
                curves[name] = np.cumprod(1 + r)
    va = {r['política']: r['Sharpe'] for r in rows if r['conjunto'] == 'VA' and r['política'] in cands}
    chosen = max(va, key=va.get)
    dates = frame.index[idx['TE'][:-1]]
    eq_rows = [{'x': _x(d), **{k: float(v[i]) for k, v in curves.items()}} for i, d in enumerate(dates)]
    out = frame.copy()
    pos, positions = 1, []
    for t in range(n):
        pos = policies[chosen](t, pos) if t < n - 1 else pos
        positions.append(float(ACTIONS[pos]))
    out['Position'] = positions
    out['Split'] = np.where(np.arange(n) < tr_n, 'TR', np.where(np.arange(n) < tr_n + va_n, 'VA', 'TE')) if 'Split' not in out else out['Split']
    notes = [f'Entorno diario: estado = momentum 5/20, volatilidad 20 (escalados con TR) + posición; acciones corto/efectivo/largo; coste {cost * 1e4:.1f} bps por cambio.',
             'Model-free (Q-Learning, DQN doble, PPO con GAE) aprende la política de la experiencia; model-based (Dyna-Q planifica con un modelo aprendido, MCTS busca con UCT sobre un modelo empírico de TR).',
             f'Entrenamiento sólo en TR; VA elige «{chosen}» por Sharpe y TE se evalúa una vez sin aprender. Es didáctico: no es una estrategia validada.']
    te_row = next(r for r in rows if r['política'] == chosen and r['conjunto'] == 'TE')
    return _done(frame, out, 'ml_rl', selected=chosen, scope='refuerzo · TR aprende · VA elige · TE', features=list(feats_df.columns), table=rows,
                 charts=[{'title': 'TE · capital acumulado por política', 'x_key': 'x', 'series': list(curves), 'rows': eq_rows, 'chart_type': 'line'},
                         {'title': 'VA · Sharpe por política', 'x_key': 'política', 'series': ['Sharpe'], 'rows': [r for r in rows if r['conjunto'] == 'VA'], 'chart_type': 'bar'}],
                 notes=notes, metrics={f'TE {k}': v for k, v in te_row.items() if isinstance(v, float)},
                 fit_n=tr_n, ml={'rl': chosen})


STEPS = {'ml_cluster': cluster, 'ml_reduce': reduce, 'ml_assoc': association, 'ml_anomaly': anomaly,
         'ml_semi': semi, 'ml_selfsup': selfsup, 'ml_rl': rl}

