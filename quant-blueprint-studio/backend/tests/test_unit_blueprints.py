"""Cobertura U2–U10: cada infografía aporta sus herramientas y cada ejemplo tiene un Blueprint que se ejecuta."""
import ast
import json
import unittest
from pathlib import Path

import torch

from backend.engine.dag_runner import DAGRunner

ROOT = Path(__file__).resolve().parents[2]
NODES = ROOT / 'backend/engine/nodes'

# Unidad → ejemplo(s) → segmentos de la infografía → herramientas que los implementan.
COVERAGE: dict[str, dict[str, dict[str, list[str]]]] = {
    'U2': {'unit-2': {'01 descargar': ['source'], '02 indicadores': ['features'], '03 objetivo': ['sup_target'], '04 tiempo': ['sup_split'],
                      '05 escalar': ['sup_scale'], '06 MLP': ['sup_mlp'], '07 pérdida': ['sup_loss'], '08 pesos': ['sup_train'],
                      '09 checkpoint': ['sup_train'], '10 evaluar': ['sup_evaluate']}},
    'U3': {'unit-3': {'01 preparar': ['sup_target', 'sup_split', 'sup_scale'], '02 inicialización': ['sup_init'], '03 optimizadores': ['sup_optimizer'],
                      '04–07 bucle': ['sup_train'], '05 pérdida': ['sup_loss'], '08 presupuesto': ['sup_train'], '09 seleccionar': ['sup_select'],
                      '10 evaluar': ['sup_evaluate']}},
    'U4': {'unit-4': {'01 etiquetas': ['sup_target'], '02 referencia': ['sup_init', 'sup_mlp'], '03–07 regularización': ['sup_regularizer'],
                      '08 pérdidas': ['sup_loss', 'sup_train'], '09 variante': ['sup_select'], '10 probabilidades': ['sup_evaluate']}},
    'U5': {'unit-5': {'01 desarrollo/test': ['sup_split'], '02–08 búsqueda': ['sup_hpo'], '09 modelo final': ['sup_scale', 'sup_train'],
                      '10 documentar': ['sup_evaluate']}},
    'U6': {'unit-6': {'01 indicadores': ['sup_scale'], '02/04 ventanas': ['sup_windows'], '03 fechas': ['sup_split'], '05–07 CNN': ['sup_cnn'],
                      '08 MLP referencia': ['sup_mlp'], '09 seleccionar': ['sup_train', 'sup_select'], '10 evaluar': ['sup_evaluate']}},
    'U7': {'unit-7': {'01 secuencias': ['sup_windows'], '02–07 recurrentes': ['sup_rnn'], '08 sin estados': ['sup_train'],
                      '09 seleccionar': ['sup_select'], '10 Ridge': ['sup_evaluate']}},
    'U8': {'unit-8-tools': {'01–10': ['ae_split', 'ae_standardize', 'ae_pca', 'ae_encoder', 'ae_decoder', 'ae_noise', 'ae_train', 'ae_select', 'ae_ridge', 'ae_test']}},
    'U9': {'unit-9-tools': {'01–10': ['gen_windows', 'gen_condition', 'gen_split', 'gen_scale', 'gen_cvae', 'gen_reparam', 'gen_vae_train', 'gen_gan_train', 'gen_select', 'gen_evaluate']},
           'unit-9-cdcgan': {'01 retornos': ['gen_windows'], '02 condiciones': ['gen_condition'], '03 separar/escalar': ['gen_split', 'gen_scale'],
                             '04–08 cDCGAN': ['gen_dcgan_train'], '09 seleccionar': ['gen_select', 'gen_regime_compare'], '10 evaluar': ['gen_evaluate']}},
    'U10': {'unit-10': {'01 datos': ['sup_target', 'sup_split', 'sup_scale'], '02–03 entorno': ['rl_env'], '04 redes': ['rl_qnet'],
                        '05–06 replay': ['rl_replay'], '07 objetivo': ['rl_target'], '08 actualizar': ['rl_train'], '09 sincronizar': ['rl_sync'],
                        '10 evaluar': ['rl_baselines', 'rl_evaluate']}},
}
FAST = {'epochs': 2, 'trials': 2, 'patience': 0}


def entries() -> dict[str, dict]:
    return {e['id']: e for e in json.loads((ROOT / 'public/blueprints/library.json').read_text())['entries']}


class UnitBlueprintTest(unittest.TestCase):
    def test_every_unit_2_to_10_is_covered(self) -> None:
        self.assertEqual(list(COVERAGE), [f'U{i}' for i in range(2, 11)])
        lib = entries()
        for unit, examples in COVERAGE.items():
            for eid, segments in examples.items():
                self.assertIn(eid, lib, f'{unit}: falta el Blueprint {eid}')
                kinds = {n['data']['kind'] for n in lib[eid]['graph']['nodes']}
                self.assertNotIn('study', kinds, f'{eid} usa todavía el monolito')
                for seg, tools in segments.items():
                    for tool in tools:
                        self.assertIn(tool, kinds, f'{unit} {eid} segmento {seg}: falta {tool}')
                self.assertTrue(lib[eid].get('infographic'), eid)

    def test_no_library_example_uses_the_monolith(self) -> None:
        for eid, e in entries().items():
            if eid.startswith('unit-'):
                self.assertFalse({'study', 'rl_agent'} & {n['data']['kind'] for n in e['graph']['nodes']}, eid)

    def test_tools_do_not_import_each_other(self) -> None:
        tool_modules = {'sup_pipeline_nodes', 'rl_pipeline_nodes', 'gen_pipeline_nodes', 'ae_pipeline_nodes', 'study_nodes', 'ml_pipeline_nodes'}
        for name in ('sup_pipeline_nodes', 'rl_pipeline_nodes', 'gen_pipeline_nodes'):
            tree = ast.parse((NODES / f'{name}.py').read_text())
            imported = {(n.module or '').lstrip('.') for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
                       {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
            self.assertFalse(imported & (tool_modules - {name}), f'{name} importa otra familia de herramientas')

    def test_each_blueprint_runs_end_to_end(self) -> None:
        torch.set_num_threads(1)
        lib = entries()
        for eid in [e for ex in COVERAGE.values() for e in ex]:
            with self.subTest(blueprint=eid):
                g = json.loads(json.dumps(lib[eid]['graph']))
                for n in g['nodes']:
                    p = n['data']['params']
                    for k, v in FAST.items():
                        if k in p or n['data']['kind'] in ('sup_train', 'sup_hpo', 'rl_train', 'rl_baselines', 'gen_vae_train', 'gen_gan_train', 'gen_dcgan_train', 'ae_train'):
                            if k == 'patience' and n['data']['kind'] not in ('sup_train',):
                                continue
                            p[k] = v
                out = DAGRunner(g).execute()
                json.dumps(out, allow_nan=False)
                tools = [n for n in g['nodes'] if n['data']['kind'] not in ('source', 'features', 'plot')]
                self.assertEqual(len(out['studies']), len(tools), eid)
                for s in out['studies']:
                    self.assertTrue(s['selected'], f'{eid}: {s["topic"]}')


if __name__ == '__main__':
    unittest.main()
