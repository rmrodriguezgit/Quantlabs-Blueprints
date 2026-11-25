"""Builds the «Taxonomía ML» library entries (one per family of the ML algorithms map + full overview)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AREA = 'Taxonomía ML · panorámica'
FEATS = 'LogReturn,Volatility,RSI,EMA,SMA'


def node(i, kind, label, params, x, y):
    return {'id': i, 'type': 'blueprint', 'position': {'x': x, 'y': y}, 'data': {'kind': kind, 'label': label, 'params': params}}


def edge(a, b):
    return {'id': f'{a}-{b}', 'source': a, 'target': b, 'sourceHandle': 'frame', 'targetHandle': 'frame', 'animated': True}


def base(y=100):
    return [node('Datos', 'source', 'Datos', {'source': 'synthetic', 'ticker': 'SPY', 'years': 4}, 0, y),
            node('Indicadores', 'features', 'Indicadores', {}, 300, y),
            node('Limpiar', 'ml_clean', 'Limpieza', {'missing': 'drop'}, 600, y)], [edge('Datos', 'Indicadores'), edge('Indicadores', 'Limpiar')]


def split(task):
    return node('Dividir', 'ml_split', 'Dividir TR / VA / TE', {'target': 'LogReturn', 'target_mode': 'direction' if task == 'cls' else 'next', 'features': FEATS, 'gap': 1}, 900, 100)


def entry(i, title, desc, instr, nodes, edges):
    return {'id': i, 'title': title, 'area': AREA, 'description': desc, 'instructions': instr, 'graph': {'nodes': nodes, 'edges': edges}}


COMMON = 'Fuente synthetic para ejecutar sin internet; cambia a yfinance para activos reales. Todo lo ajustado usa sólo TR.'
E = []
for task, i, title, algos in (('cls', 'taxonomy-classification', 'Supervisado · Clasificación', 'Logística, Naive Bayes, Árbol, Random Forest, XGBoost, KNN, SVM…'),
                              ('reg', 'taxonomy-regression', 'Supervisado · Regresión', 'Lineal múltiple, Lasso, Ridge, Árbol, Random Forest, XGBoost, KNN, SVR…')):
    n, e = base()
    key = 'algorithm_cls' if task == 'cls' else 'algorithm'
    n += [split(task), node('Entrenar', 'ml_train', 'Entrenar · Compare', {key: 'Compare', 'estimators': 100, 'max_depth': 4}, 0, 380),
          node('Evaluar', 'ml_evaluate', 'Evaluar TE', {}, 300, 380), node('Ver', 'ml_visualize', 'Visualizar', {}, 600, 380)]
    e += [edge('Limpiar', 'Dividir'), edge('Dividir', 'Entrenar'), edge('Entrenar', 'Evaluar'), edge('Evaluar', 'Ver')]
    E.append(entry(i, title, f'{algos} Compare entrena todos en TR, elige en VA y mide TE una vez.',
                   [COMMON, 'Elige un algoritmo concreto en Entrenar para estudiarlo aislado.', 'XGBoost usa la librería xgboost si carga (macOS: brew install libomp); si no, HistGradientBoosting.'], n, e))

for i, kind, title, params, desc in (
        ('taxonomy-clustering', 'ml_cluster', 'No supervisado · Clustering', {'algorithm': 'Compare', 'features': 'LogReturn,Volatility,RSI'}, 'K-Means, DBSCAN y jerárquico agrupan sesiones similares (regímenes); silhouette compara.'),
        ('taxonomy-dimred', 'ml_reduce', 'No supervisado · Reducción de dimensionalidad', {'algorithm': 'PCA', 'features': FEATS}, 'PCA, ICA, t-SNE y UMAP reducen rasgos conservando la información clave.'),
        ('taxonomy-association', 'ml_assoc', 'No supervisado · Reglas de asociación', {'algorithm': 'Apriori', 'features': 'LogReturn,Volatility,RSI'}, 'Apriori y FP-Growth descubren eventos que co-ocurren (soporte, confianza, lift, validación fuera de muestra).'),
        ('taxonomy-anomaly', 'ml_anomaly', 'No supervisado · Detección de anomalías', {'algorithm': 'Compare', 'features': 'LogReturn,Volatility'}, 'Isolation Forest, Z-Score y One-Class SVM marcan sesiones raras.')):
    n, e = base()
    n.append(node('Tool', kind, title.split(' · ')[1], params, 900, 100))
    e.append(edge('Limpiar', 'Tool'))
    E.append(entry(i, title, desc, [COMMON, 'Cambia «Algoritmo» en el Inspector para recorrer cada método.'], n, e))

for i, kind, title, params, desc in (
        ('taxonomy-semi', 'ml_semi', 'Semi-supervisado', {'algorithm': 'Compare', 'labeled_fraction': 0.1}, 'Self-Training, Co-Training y Label Propagation con sólo 10 % de etiquetas en TR, frente a referencia y techo.'),
        ('taxonomy-selfsup', 'ml_selfsup', 'Auto-supervisado', {'algorithm': 'Contrastive', 'epochs': 30}, 'Aprendizaje contrastivo, masked modeling y autoencoder sin etiquetas; sonda lineal frente a rasgos brutos.')):
    n, e = base()
    n += [split('cls'), node('Tool', kind, title, params, 1200, 100)]
    e += [edge('Limpiar', 'Dividir'), edge('Dividir', 'Tool')]
    E.append(entry(i, title, desc, [COMMON, 'Cambia el enfoque/técnica en el Inspector.'], n, e))

for fam, title, algos in (('model-free', 'Refuerzo · Model-Free', 'Q-Learning, DQN y PPO aprenden la política directamente de la experiencia.'),
                          ('model-based', 'Refuerzo · Model-Based', 'Dyna-Q y MCTS planifican con un modelo del entorno aprendido en TR.')):
    n = [node('Datos', 'source', 'Datos', {'source': 'synthetic', 'ticker': 'SPY', 'years': 4}, 0, 100),
         node('RL', 'ml_rl', title, {'family': fam, 'algorithm': 'Compare', 'episodes': 20}, 300, 100)]
    E.append(entry(f'taxonomy-rl-{fam}', title, algos + ' Comparadas con comprar y mantener y efectivo.', [COMMON, 'VA elige por Sharpe; TE sin aprender. Didáctico, no es una estrategia validada.'], n, [edge('Datos', 'RL')]))

# overview: every family on one canvas
n, e = base(400)
n += [split('cls'),
      node('Clas', 'ml_train', 'Supervisado · Compare', {'algorithm_cls': 'Compare', 'estimators': 60, 'max_depth': 4}, 1200, 0),
      node('Clu', 'ml_cluster', 'Clustering', {'algorithm': 'K-Means', 'features': 'LogReturn,Volatility,RSI'}, 900, 250),
      node('Red', 'ml_reduce', 'Reducción · PCA', {'algorithm': 'PCA', 'features': FEATS}, 900, 400),
      node('Aso', 'ml_assoc', 'Asociación · FP-Growth', {'algorithm': 'FP-Growth', 'features': 'LogReturn,Volatility,RSI'}, 900, 550),
      node('Ano', 'ml_anomaly', 'Anomalías · Isolation Forest', {'algorithm': 'Isolation Forest', 'features': 'LogReturn,Volatility'}, 900, 700),
      node('Semi', 'ml_semi', 'Semi-supervisado', {'algorithm': 'Compare'}, 1200, 150),
      node('Self', 'ml_selfsup', 'Auto-supervisado', {'algorithm': 'Contrastive', 'epochs': 15}, 1200, 300),
      node('RL', 'ml_rl', 'Refuerzo · todos', {'family': 'all', 'algorithm': 'Compare', 'episodes': 10}, 300, 850)]
n[3]['position'] = {'x': 900, 'y': 100}
e += [edge('Limpiar', x) for x in ('Dividir', 'Clu', 'Red', 'Aso', 'Ano')] + [edge('Dividir', x) for x in ('Clas', 'Semi', 'Self')] + [edge('Datos', 'RL')]
E.insert(0, entry('taxonomy-overview', 'Machine Learning · mapa completo', 'Supervisado, no supervisado (clustering, reducción, asociación, anomalías), semi-supervisado, auto-supervisado y refuerzo en un solo lienzo.',
                  [COMMON, 'Cada rama produce su propio informe; abre las fichas de cada familia para estudiarlas con detalle.'], n, e))

for path in (ROOT / 'public/blueprints/library.json', ROOT / 'src/renderer/data/library.json'):
    lib = json.loads(path.read_text())
    ids = {x['id'] for x in E}
    lib['entries'] = [x for x in lib['entries'] if x['id'] not in ids] + E
    path.write_text(json.dumps(lib, indent=2, ensure_ascii=False) + '\n')
for x in E:
    (ROOT / f'public/blueprints/{x["id"]}.json').write_text(json.dumps({'format': 'quantlabs-blueprint-library', 'schemaVersion': 1, 'entries': [x]}, indent=2, ensure_ascii=False) + '\n')
print(len(E), 'entries')
