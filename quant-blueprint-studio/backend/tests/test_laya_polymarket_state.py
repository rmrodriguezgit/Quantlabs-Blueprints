"""Estado y preguntas que LAYA recibe en los ejemplos de Polymarket (valores dinámicos por ventana)."""
import unittest
import unittest.mock

import numpy as np
import pandas as pd

from backend.engine.nodes import polymarket_nodes as pm


def klines(start, n_before=30, n_in=4, base=94000.0, vol_before=10.0, vol_in=20.0):
    out, p = [], base
    for i in range(-n_before, n_in):
        o = p
        p = p * (1 + (.0005 if i % 3 else -.0003))
        out.append([start + 60 * i, o, p, vol_before if i < 0 else vol_in])
    return out


class State(unittest.TestCase):
    def test_indicators(self):
        self.assertIsNone(pm.rsi([1, 2, 3]))
        self.assertEqual(pm.rsi(list(range(1, 20))), 100.0)
        self.assertLess(pm.rsi(list(range(20, 1, -1))), 1)
        self.assertEqual(pm.volume_trend([10] * 15 + [20] * 5), 'increasing')
        self.assertEqual(pm.volume_trend([10] * 15 + [5] * 5), 'decreasing')
        self.assertEqual(pm.volume_trend([10] * 20), 'stable')
        self.assertEqual(pm.orderbook_bias(.3), 'bullish_depth')
        self.assertEqual(pm.orderbook_bias(-.3), 'bearish_depth')
        self.assertEqual(pm.orderbook_bias(0), 'balanced')
        self.assertIsNone(pm.orderbook_bias(float('nan')))

    def test_target_carries_prices_and_no_future(self):
        start = 1_000_020 - 1_000_020 % 60
        k = klines(start)
        t = pm.target_from_klines(k, start, start + 150, start + 300, True)
        self.assertEqual(t['p_open'], k[30][1])
        self.assertEqual(t['p_now'], k[31][2])           # only klines finished by t_dec
        self.assertIsNotNone(t['rsi_14'])
        self.assertEqual(t['volume_trend'], 'increasing')
        old = pm.target_from_klines([x[:3] for x in k], start, start + 150, start + 300, True)   # cached 3-col klines still work
        self.assertIsNone(old['volume_trend'])
        self.assertAlmostEqual(old['z'], t['z'])

    def test_state_and_question_use_live_values(self):
        start = 1_000_020 - 1_000_020 % 60
        t = pm.target_from_klines(klines(start), start, start + 150, start + 300, True)
        z = pm.market_signals(pd.Series(np.linspace(.5, .6, 10)), .6, .25, 5, t)
        s = pm.laya_state('Bitcoin Up or Down', '5m', .6, z, .01, 2.5, t, 'btc', has_book=True)
        self.assertEqual(s['symbol'], 'BTC/USD')
        self.assertEqual(s['timeframe'], '5m')
        self.assertEqual(s['reference_price'], round(t['p_open'], 2))
        self.assertEqual(s['current_price'], round(t['p_now'], 2))
        self.assertEqual(s['recent_orderbook_bias'], 'bullish_depth')
        self.assertIsNone(pm.laya_state('q', '5m', .6, z, None, 2.5, t, 'btc')['recent_orderbook_bias'])   # history: no book
        q = pm.laya_question('5m', risk=pm.RISK_Q, asset='btc', reference=s['reference_price'])
        ref = f"{s['reference_price']:,.2f} USD"
        self.assertIn(ref, q['direction']['instructions'])
        self.assertIn(ref, q['direction']['criteria']['down'])
        self.assertEqual(set(q['high_volatility_risk']['criteria']), {'true', 'false'})
        self.assertNotIn('USD', pm.laya_question('5m')['direction']['instructions'])   # no price known → generic


class AskLaya(unittest.TestCase):
    def test_per_state_questions_and_risk(self):
        calls = []

        class Agent:
            def predict_batch(self, states, q, batch_size=64):
                calls.append(len(states))
                ans = {'direction': {'probabilities': {'up': .7, 'down': .3}}}
                if 'high_volatility_risk' in q:
                    ans['high_volatility_risk'] = {'noul': .2}
                return [{'answers': ans} for _ in states]

        pm._LAYA_CACHE.clear()
        states = [{'i': i} for i in range(3)]
        qs = [pm.laya_question('5m', reference=1.0)] * 2 + [pm.laya_question('5m', risk='r', reference=2.0)]
        with unittest.mock.patch('backend.engine.nodes.laya_nodes._agent', return_value=Agent()):
            out = pm._ask_laya(states, qs, 'ck', None)
        self.assertEqual(sorted(calls), [1, 2])
        self.assertNotIn('risk', out[0])
        self.assertEqual(out[2]['risk'], .2)


if __name__ == '__main__':
    import unittest.mock  # noqa: F401
    unittest.main()
