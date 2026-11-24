"""Cerebro fixer: rules for known errors, range catch-all and LLM output sanitising."""
import unittest
from unittest import mock

from backend.brain import fixer, llm

CATALOG = {
    'yfinance_data': {'label': 'yfinance Data', 'input': None, 'output': 'frame', 'fields': [
        {'key': 'ticker', 'value': 'SPY'}, {'key': 'interval', 'value': '1d', 'options': ['1m', '5m', '1h', '1d']},
        {'key': 'range_mode', 'value': 'period', 'options': ['period', 'start_end']},
        {'key': 'period', 'value': '1y', 'options': ['5d', '1mo', '1y', 'max']},
        {'key': 'back_adjust', 'value': False}, {'key': 'timeout', 'value': 20, 'min': 3, 'max': 120}]},
    'polymarket_source': {'label': 'Polymarket Market', 'input': None, 'output': 'frame', 'fields': [{'key': 'market', 'value': 'btc-updown-15m'}, {'key': 'timeout', 'value': 8, 'min': 1, 'max': 30}]},
    'laya_signal': {'label': 'Laya Signal', 'input': 'frame', 'output': 'frame', 'fields': [
        {'key': 'train_windows', 'value': 120, 'min': 20, 'max': 500}, {'key': 'max_rows', 'value': 500, 'min': 30, 'max': 3000},
        {'key': 'mode', 'value': 'auto', 'options': ['auto', 'daily', 'polymarket']}]},
    'laya_calibrate': {'label': 'Laya Calibrate', 'input': 'frame', 'output': 'frame', 'fields': [{'key': 'epochs', 'value': 300, 'min': 10, 'max': 5000}]},
    'laya_updown': {'label': 'LAYA UP/DOWN', 'input': 'frame', 'output': 'frame', 'fields': [
        {'key': 'weight_source', 'value': 'manual', 'options': ['manual', 'calibrated', 'online']}, {'key': 'use_laya', 'value': True}]},
}


def node(i, kind, label=None, **params):
    return {'id': i, 'data': {'kind': kind, 'label': label or i, 'params': params}}


def edge(a, b):
    return {'id': f'{a}-{b}', 'source': a, 'target': b, 'sourceHandle': 'frame', 'targetHandle': 'frame'}


class FixerTests(unittest.TestCase):
    def test_yahoo_intraday_limit_from_runtime_error(self):
        g = {'nodes': [node('Y', 'yfinance_data', 'Datos QQQ', ticker='QQQ', interval='1m', period='1y')], 'edges': []}
        r = fixer.fix(g, 'Datos QQQ: Intervalo 1m: Yahoo permite como máximo 7 días por descarga y pediste ~366.', [], CATALOG, 'rules')
        self.assertIn({'type': 'set_param', 'node': 'Y', 'key': 'period', 'value': '5d'}, r['fixes'])

    def test_validation_issue_with_node(self):
        g = {'nodes': [node('Y', 'yfinance_data', back_adjust=True)], 'edges': []}
        r = fixer.fix(g, None, [{'message': 'Y: back_adjust sólo aplica con auto_adjust desactivado.', 'node': 'Y', 'severity': 'error'}], CATALOG, 'rules')
        self.assertEqual(r['fixes'], [{'type': 'set_param', 'node': 'Y', 'key': 'back_adjust', 'value': False}])

    def test_unknown_ticker_uses_search(self):
        g = {'nodes': [node('Y', 'yfinance_data', 'Datos', ticker='APPLE')], 'edges': []}
        with mock.patch.object(fixer, '_ticker_suggestion', return_value='AAPL'):
            r = fixer.fix(g, 'Datos: Yahoo no devolvió datos para APPLE (1d, 1y). Revisa el símbolo', [], CATALOG, 'rules')
        self.assertEqual(r['fixes'][0]['value'], 'AAPL')

    def test_missing_market_uses_polymarket_search(self):
        g = {'nodes': [node('P', 'polymarket_source', 'Mercado', market='bitcoin-up-or-dwn')], 'edges': []}
        with mock.patch.object(fixer, '_market_suggestion', return_value='btc-updown-15m-1'):
            r = fixer.fix(g, "Mercado: Mercado no encontrado en Polymarket: 'bitcoin-up-or-dwn'.", [], CATALOG, 'rules')
        self.assertEqual(r['fixes'][0], {'type': 'set_param', 'node': 'P', 'key': 'market', 'value': 'btc-updown-15m-1'})

    def test_small_sample_raises_upstream_laya_signal(self):
        g = {'nodes': [node('P', 'polymarket_source'), node('S', 'laya_signal', train_windows=30), node('C', 'laya_calibrate', 'Calibrar')],
             'edges': [edge('P', 'S'), edge('S', 'C')]}
        r = fixer.fix(g, 'Calibrar: Muy pocas ventanas para TR/VA/TE; sube «Ventanas de entrenamiento» en Laya Signal.', [], CATALOG, 'rules')
        self.assertEqual(r['fixes'][0], {'type': 'set_param', 'node': 'S', 'key': 'train_windows', 'value': 120})

    def test_calibrated_without_calibrate_and_missing_laya(self):
        g = {'nodes': [node('U', 'laya_updown', 'Decisión', weight_source='calibrated')], 'edges': []}
        r = fixer.fix(g, 'Decisión: weight_source=calibrated requiere Laya Calibrate aguas arriba.', [], CATALOG, 'rules')
        self.assertEqual(r['fixes'][0]['value'], 'manual')
        r = fixer.fix(g, 'Decisión: laya-mlx no está instalado. Ejecuta: pip install laya-mlx', [], CATALOG, 'rules')
        self.assertIn({'type': 'set_param', 'node': 'U', 'key': 'use_laya', 'value': False}, r['fixes'])

    def test_engine_busy_means_retry(self):
        r = fixer.fix({'nodes': [], 'edges': []}, 'Engine busy. Wait for the current run.', [], CATALOG, 'rules')
        self.assertTrue(r['retry'])
        self.assertEqual(r['fixes'], [])

    def test_catch_all_clamps_ranges_and_options(self):
        g = {'nodes': [node('S', 'laya_signal', train_windows=9000, mode='weird')], 'edges': []}
        r = fixer.fix(g, 'S: algo raro', [], CATALOG, 'rules')
        self.assertIn({'type': 'set_param', 'node': 'S', 'key': 'train_windows', 'value': 500}, r['fixes'])
        self.assertIn({'type': 'set_param', 'node': 'S', 'key': 'mode', 'value': 'auto'}, r['fixes'])

    def test_same_value_other_type_is_not_an_error(self):
        cat = {'model': {'label': 'm', 'fields': [{'key': 'heads', 'value': '2', 'options': ['1', '2', '4']}, {'key': 'lr', 'value': .01, 'min': 0, 'max': 1}]}}
        g = {'nodes': [node('M', 'model', heads=4, lr=1)], 'edges': []}
        self.assertEqual(fixer.clamp_fixes(g, cat), [])

    def test_llm_output_is_sanitised(self):
        g = {'nodes': [node('Y', 'yfinance_data', 'Datos'), node('S', 'laya_signal', 'Señal')], 'edges': []}
        proposal = {'fixes': [
            {'type': 'set_param', 'node': 'Y', 'key': 'timeout', 'value': 999},      # clamped to 120
            {'type': 'set_param', 'node': 'Y', 'key': 'hack', 'value': 1},          # unknown key → dropped
            {'type': 'set_param', 'node': 'ZZ', 'key': 'ticker', 'value': 'X'},     # unknown node → dropped
            {'type': 'connect', 'source': 'Y', 'target': 'S', 'pin': 'frame'},     # valid
            {'type': 'delete_everything'}],
            'explanation': ['subir timeout', 'conectar'], 'unfixable': []}
        with mock.patch.object(llm, 'chat_json', return_value=(proposal, 'qwen')):
            r = fixer.fix(g, 'Señal: error desconocido sin regla', [], CATALOG, 'auto')
        self.assertEqual(r['engine'], 'ollama')
        self.assertEqual(r['fixes'], [{'type': 'set_param', 'node': 'Y', 'key': 'timeout', 'value': 120},
                                      {'type': 'connect', 'source': 'Y', 'target': 'S', 'pin': 'frame'}])
        self.assertTrue(any('Descartado' in t for t in r['explanation']))

    def test_no_llm_reports_unfixable(self):
        g = {'nodes': [node('S', 'laya_signal', 'Señal')], 'edges': []}
        with mock.patch.object(llm, 'chat_json', side_effect=llm.LLMError('Ollama no responde')):
            r = fixer.fix(g, 'Señal: error desconocido sin regla', [], CATALOG, 'auto')
        self.assertEqual(r['fixes'], [])
        self.assertEqual(r['fallback_reason'], 'Ollama no responde')
        self.assertTrue(r['unfixable'])


if __name__ == '__main__':
    unittest.main()
