"""Brain: deterministic planning, compilation against the catalog, and explanation checks (no Ollama needed)."""
import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from backend.brain import explain, planner
from backend.engine.dag_runner import DAGRunner

ROOT = Path(__file__).resolve().parents[2]


def catalog() -> dict:
    """The real UI catalog, exported from src/renderer/types/blueprint.ts with esbuild."""
    script = ("import('esbuild').then(async ({build}) => {"
              "const r = await build({entryPoints:['src/renderer/types/blueprint.ts'],bundle:true,format:'esm',platform:'node',write:false,logLevel:'error'});"
              "const m = await import('data:text/javascript;base64,' + Buffer.from(r.outputFiles[0].text).toString('base64'));"
              "process.stdout.write(JSON.stringify(m.catalog)); })")
    try:
        out = subprocess.run(['node', '-e', script], cwd=ROOT, capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise unittest.SkipTest(f'Node/esbuild unavailable to export the catalog: {exc}')
    return json.loads(out)


class BrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cat = catalog()

    def test_detects_tickers_years_and_source(self) -> None:
        hint = planner.detect_source('Analiza Bitcoin y NVDA 5 años con RSI y MACD')
        self.assertEqual(hint['tickers'][:2], ['BTC-USD', 'NVDA'])
        self.assertEqual(hint['years'], 5)
        self.assertEqual(hint['source'], 'yfinance')
        self.assertNotIn('RSI', hint['tickers'])
        self.assertEqual(planner.detect_source('datos sintéticos sin internet')['source'], 'synthetic')

    def test_compiler_rejects_unknown_tools_and_cleans_parameters(self) -> None:
        plan = {'title': 't', 'objective': 'o', 'missing': [], 'steps': [
            {'id': 'd', 'tool': 'source', 'parameters': {'years': 99, 'source': 'bloomberg', 'hack': 1}, 'why': ''},
            {'id': 'x', 'tool': 'garch_magic', 'why': ''},
            {'id': 'f', 'tool': 'features', 'why': ''},
            {'id': 'p', 'tool': 'plot', 'inputs': {'frame': 'f'}, 'why': ''}]}
        result = planner.compile_plan(plan, self.cat, {})
        params = result['graph']['nodes'][0]['data']['params']
        self.assertEqual(params['years'], 20)
        self.assertEqual(params['source'], 'synthetic')
        self.assertNotIn('hack', params)
        self.assertEqual([d['tool'] for d in result['dropped']], ['garch_magic'])
        self.assertTrue(result['check']['ok'], result['check'])
        self.assertIn(('f', 'p'), {(e['source'], e['target']) for e in result['graph']['edges']})

    def test_compiler_adds_missing_source_and_drops_unreachable_steps(self) -> None:
        result = planner.compile_plan({'steps': [{'id': 'b', 'tool': 'backtest', 'why': ''}]}, self.cat, {})
        self.assertEqual(result['graph']['nodes'][0]['data']['kind'], 'source')
        self.assertIn('backtest', [d['tool'] for d in result['dropped']])

    def test_rules_plans_are_valid_and_execute(self) -> None:
        prompts = {'agente de refuerzo sobre Bitcoin': 'rl_agent', 'portafolio de AAPL MSFT con riesgo': 'portfolio',
                   'red neuronal y backtest en SPY': 'backtest', 'Laya sobre QQQ': 'laya_signal'}
        for prompt, kind in prompts.items():
            with self.subTest(prompt=prompt):
                result = planner.plan(prompt, self.cat, 'rules')
                self.assertEqual(result['engine'], 'rules')
                self.assertTrue(result['check']['ok'], result['check'])
                self.assertIn(kind, [s['tool'] for s in result['steps']])
        graph = deepcopy(planner.plan('agente de refuerzo con 2 años de datos sintéticos', self.cat, 'rules')['graph'])
        for node in graph['nodes']:
            if 'epochs' in node['data']['params']:
                node['data']['params']['epochs'] = 1
        self.assertEqual(len(DAGRunner(graph).execute()['nodes']), len(graph['nodes']))

    def test_explain_rules_and_number_verification(self) -> None:
        report = {'title': 'F', 'topic': 'laya_feedback', 'selected': 's', 'splits': {'test': 40}, 'table': [],
                  'metrics': {'retorno_estrategia': -0.045, 'retorno_buy_hold': -0.039, 'logloss_1a_mitad': 0.724, 'logloss_2a_mitad': 0.71, 'abstencion': 0.12}}
        out = explain.explain(report, 'rules')
        self.assertIn('-4.5 %', out['summary'])
        self.assertTrue(any('aprende' in f for f in out['findings']))
        self.assertTrue(any('40 observaciones' in c for c in out['cautions']))
        self.assertEqual(explain.verify(['El retorno fue -4.5 % y el logloss 0.724.'], report), [])
        self.assertEqual(explain.verify(['El Sharpe fue 1.87.'], report), ['1.87'])
        with self.assertRaises(ValueError):
            explain.explain({'metrics': {}, 'table': []}, 'rules')


if __name__ == '__main__':
    unittest.main()


class TutorAndMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cat = catalog()

    def test_tutor_routes_question_types_and_selected_node(self) -> None:
        from backend.brain import tutor
        cases = {'¿Qué es el ratio de Sharpe?': 'concept:sharpe', '¿Para qué sirve Technical Features?': 'tool:features',
                 '¿Cómo evito la fuga de información?': 'concept:fuga', 'ejemplo de red neuronal': 'example:area-network'}
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(tutor.ask(question, self.cat, 'rules')['sources'][0]['id'], expected)
        this = tutor.ask('¿Qué hace este nodo?', self.cat, 'rules', 'laya_calibrate')
        self.assertEqual(this['sources'][0]['id'], 'tool:laya_calibrate')
        leak = tutor.ask('¿Cómo evito la fuga de información?', self.cat, 'rules')
        self.assertIn('data_quality', [a.get('kind') for a in leak['actions']])
        with self.assertRaises(ValueError):
            tutor.ask('x', self.cat, 'rules')

    def test_tutor_drops_invented_citations(self) -> None:
        from unittest import mock
        from backend.brain import tutor
        fake = ({'answer': 'Ver [concept:sharpe] y [tool:inventado].', 'sources': ['concept:sharpe', 'tool:inventado'], 'follow_up': []}, 'qwen-test')
        with mock.patch('backend.brain.llm.chat_json', return_value=fake):
            out = tutor.ask('¿Qué es el ratio de Sharpe?', self.cat, 'auto')
        self.assertEqual(out['engine'], 'ollama')
        self.assertEqual([s['id'] for s in out['sources']], ['concept:sharpe'])

    def test_memory_rating_reuse_and_missing_tools(self) -> None:
        import os
        import tempfile
        from unittest import mock
        from backend.brain import store
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'QL_BRAIN_DIR': tmp}):
            first = planner.plan('Agente de refuerzo sobre Bitcoin', self.cat, 'rules')
            self.assertTrue(first['plan_id'])
            store.rate(first['plan_id'], 1)
            again = planner.plan('agente de refuerzo bitcoin 3 años', self.cat, 'rules')
            self.assertEqual(again['retrieved'][0]['type'], 'memory')
            store.remember({'title': 't', 'steps': [], 'missing': [{'name': 'GARCH', 'purpose': 'volatilidad condicional'}]}, 'garch por favor')
            store.remember({'title': 't', 'steps': [], 'missing': [{'name': 'garch', 'purpose': 'x'}]}, 'otra vez garch')
            requests = store.memory()['requests']
            self.assertEqual((requests[0]['name'], requests[0]['count']), ('GARCH', 2))
            store.set_request_status(requests[0]['id'], 'planned')
            self.assertEqual(store.memory()['requests'][0]['status'], 'planned')
            with self.assertRaises(ValueError):
                store.rate('nope', 1)
