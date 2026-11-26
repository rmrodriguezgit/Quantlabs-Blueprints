"""Offline tests for the Polymarket connector and the LAYA UP/DOWN node (HTTP mocked)."""
import json
import os
import tempfile
import unittest
from unittest import mock

from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes import polymarket_nodes as pm

MARKET = {'slug': 'btc-updown-15m-1795374900', 'question': 'Bitcoin Up or Down - 15m', 'conditionId': '0x' + 'a' * 64,
          'outcomes': '["Up", "Down"]', 'outcomePrices': '["0.6", "0.4"]', 'clobTokenIds': '["111", "222"]',
          'eventStartTime': '2026-11-22T19:15:00Z', 'endDate': '2026-11-22T19:30:00Z', 'closed': False, 'active': True,
          'acceptingOrders': True, 'volumeNum': 1000, 'liquidityNum': 500, 'volume24hr': 900, 'events': [{'tags': [{'slug': '15M'}]}]}


CLOSED = {**MARKET, 'slug': 'btc-updown-15m-1795374000', 'closed': True, 'outcomePrices': '["1", "0"]'}


def fake_get(url, params=None, **_):
    if url.endswith('/markets'):
        if (params or {}).get('slug') == CLOSED['slug']:
            return [CLOSED] if (params or {}).get('closed') == 'true' else []
        return [MARKET]
    if url.endswith('/events'):
        return [{'markets': [MARKET]}]
    if url.endswith('/prices-history'):
        return {'history': [{'t': 1795374900 + 30 * i, 'p': 0.5 + 0.004 * i} for i in range(40)]}
    if url.endswith('/book'):
        return {'bids': [{'price': '0.65', 'size': '300'}], 'asks': [{'price': '0.66', 'size': '100'}]}
    if url.endswith('/klines'):
        t0 = int(params['startTime']) // 1000 // 60 * 60
        n = (int(params['endTime']) // 1000 - t0) // 60 + 1
        return [[(t0 + 60 * i) * 1000, str(60000 + i * (1 + (i % 3))), '0', '0', str(60000 + (i + 1) * (1 + (i % 3)))] for i in range(n)]
    if url.endswith('/crypto-price'):
        return {'openPrice': 60000.0, 'closePrice': None, 'completed': False}
    if url.endswith('/public-search'):
        return {'events': [{'title': 'x', 'markets': [MARKET], 'tags': []}]}
    raise AssertionError(url)


@mock.patch.object(pm, '_get', side_effect=fake_get)
class PolymarketTests(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(pm, 'chainlink_spot', return_value=[{'t': 1795374960.0, 'p': 60050.0}])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_target_signal(self, _):
        start = 1795374900
        live = pm.target_signal('btc', start, start + 120, start + 900)
        self.assertGreater(live['z'], 0)                       # price rose since the window open
        self.assertAlmostEqual(live['minutes_left'], 13.0)
        hist = pm.target_signal('btc', start, start + 120, start + 900, historical=True)
        self.assertLessEqual(hist['diff_pct'], live['diff_pct'] + 1e-9)   # historical uses only finished klines
        self.assertIsNone(pm.target_signal('btc', start, start - 10, start + 900))   # before the window opens
        self.assertIsNone(pm.target_signal('doesnotexist', start, start + 60, start + 900))

    def test_underlying_price_to_beat(self, _):
        frame, report = pm.polymarket_source({'market': MARKET['slug']})
        u = report['market']['underlying']
        self.assertEqual((u['asset'], u['price_to_beat'], u['spot'], u['leading']), ('BTC', 60000.0, 60050.0, 'UP'))
        self.assertAlmostEqual(u['diff'], 50.0)
        self.assertIsNone(pm.price_to_beat('btc', '30m', MARKET['eventStartTime'], MARKET['endDate']))   # no verified 30m variant

    def test_ref_parsing(self, _):
        self.assertEqual(pm._parse_ref('https://polymarket.com/event/foo-bar')[0], 'slug')
        self.assertEqual(pm._parse_ref('0x' + 'b' * 64)[0], 'condition_id')
        self.assertEqual(pm._parse_ref('9' * 70)[0], 'token_id')
        self.assertEqual(pm._parse_ref('bitcoin price')[0], 'text')

    def test_outcome_mapping(self, _):
        self.assertEqual(pm.map_outcomes(['No', 'Yes']), (1, 0))
        self.assertEqual(pm.map_outcomes(['Up', 'Down']), (0, 1))
        self.assertEqual(pm.map_outcomes(['A', 'B'], '1'), (1, 0))
        with self.assertRaises(ValueError):
            pm.map_outcomes(['a', 'b', 'c'])

    def test_window_detection_and_override(self, _):
        for win in ('5m', '15m', '30m', '1h'):
            frame, report = pm.polymarket_source({'market': MARKET['slug'], 'window': win})
            self.assertEqual(frame.attrs['polymarket']['window'], win)
        frame, _r = pm.polymarket_source({'market': MARKET['slug']})
        self.assertEqual(frame.attrs['polymarket']['window'], '15m')
        self.assertTrue(frame['Book_spread'].iloc[:-1].isna().all())   # snapshot only on last row

    def test_decision_trace_and_weights(self, _):
        frame, _r = pm.polymarket_source({'market': MARKET['slug']})
        out, rep = pm.laya_updown(frame, {'use_laya': False})
        d = rep['decision']
        self.assertEqual(d['decision'], 'UP')
        self.assertEqual(d['horizon'], '15m')
        self.assertEqual(d['laya']['used'], False)
        _o, flipped = pm.laya_updown(frame, {'use_laya': False, 'w_market': -3, 'w_momentum': -3, 'w_imbalance': -3})
        self.assertEqual(flipped['decision']['decision'], 'DOWN')
        with self.assertRaises(ValueError):
            pm.laya_updown(frame, {'use_laya': False, 'weight_source': 'calibrated'})

    def test_graph_execution(self, _):
        graph = {'nodes': [{'id': 'a', 'data': {'kind': 'polymarket_source', 'params': {'market': 'btc-updown-15m-1795374900'}}},
                           {'id': 'b', 'data': {'kind': 'laya_updown', 'params': {'use_laya': False, 'window': '1h'}}}],
                 'edges': [{'source': 'a', 'target': 'b', 'sourceHandle': 'frame', 'targetHandle': 'frame'}]}
        result = DAGRunner(graph).execute()
        self.assertEqual(result['decisions'][0]['horizon'], '1h')
        json.dumps(result, allow_nan=False)


    def test_online_learning_updates_after_resolution(self, _):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'QL_LEARNING_DIR': tmp}):
            frame, _r = pm.polymarket_source({'market': MARKET['slug']})
            params = {'use_laya': False, 'weight_source': 'online', 'online_lr': .1, 'minute_buckets': False}
            _o, first = pm.laya_updown(frame, params)
            learning = first['decision']['learning']
            self.assertEqual(learning['key'], 'btc-updown-15m')
            self.assertEqual(learning['updates'], 0)
            self.assertEqual(learning['pending'], [MARKET['slug']])
            # pretend that decision belonged to a previous window that has now resolved UP
            state = pm.load_learning('btc-updown-15m')
            state['pending'][CLOSED['slug']] = state['pending'].pop(MARKET['slug'])
            pm._save_learning('btc-updown-15m', state)
            _o, second = pm.laya_updown(frame, params)
            learning = second['decision']['learning']
            self.assertEqual(learning['updates'], 1)
            self.assertEqual(learning['history'][-1]['outcome'], 'UP')
            p, z = state['pending'][CLOSED['slug']]['p'], state['pending'][CLOSED['slug']]['z']
            expected = learning['initial_weights']['market'] - .1 * ((p - 1) * z['market'] + .001 * learning['initial_weights']['market'])
            self.assertAlmostEqual(learning['weights']['market'], expected, places=6)
            self.assertAlmostEqual(learning['weights']['bias'], -.1 * (p - 1), places=6)
            self.assertTrue(pm.reset_learning('btc-updown-15m'))
            self.assertIsNone(pm.load_learning('btc-updown-15m'))

    def test_online_learning_with_laya_enabled(self, _):
        class Agent:
            def predict(self, state, questions):
                return {'answers': {'direction': {'probabilities': {'up': .7, 'down': .3}}}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'QL_LEARNING_DIR': tmp}), \
                mock.patch('backend.engine.nodes.laya_nodes._agent', return_value=Agent()):
            frame, _r = pm.polymarket_source({'market': MARKET['slug']})
            _o, rep = pm.laya_updown(frame, {'weight_source': 'online'})
            self.assertTrue(rep['decision']['laya']['used'])
            b = rep['decision']['bucket']
            self.assertEqual(rep['decision']['buckets_total'], 15)            # 15m window, 60 s buckets
            self.assertEqual(rep['decision']['learning']['pending'], [f"{MARKET['slug']}#{b}"])

    def test_buckets_and_window_split(self, _):
        self.assertEqual(pm.decision_offsets('5m', 60, 0), [30, 90, 150, 210, 270])
        self.assertEqual(pm.bucket_seconds('1h'), 300)
        self.assertEqual(pm.bucket_of(299, 60, 300), 4)
        self.assertEqual(pm.bucket_of(10_000, 60, 300), 4)
        self.assertIsNone(pm.bucket_of(10, 0, 300))
        with self.assertRaises(ValueError):
            pm.bucket_seconds('5m', '70')
        raw = {'slug': 's', 'start': 1000, 'step': 300, 'question': 'q', 'y': 1.0,
               'prices': [[900 + 30 * i, .5 + .01 * i] for i in range(20)], 'klines': []}
        rec = pm.window_at(raw, 90, 5)
        self.assertTrue(all(t <= 1090 for t, _p in raw['prices'] if _p in rec['prices']))   # nothing after the decision point
        self.assertEqual(rec['p_market'], .5 + .01 * 6)

    def test_online_updates_only_its_bucket(self, _):
        state = {'key': 'k', 'weights': {**{k: 1.0 for k in pm.SIGNALS}, 'bias': 0.0}, 'initial_weights': {}, 'history': [],
                 'buckets': {str(b): {'weights': {**{k: 1.0 for k in pm.SIGNALS}, 'bias': 0.0}, 'initial_weights': {}} for b in range(3)},
                 'pending': {CLOSED['slug'] + '#1': {'t': 9e12, 'slug': CLOSED['slug'], 'bucket': 1, 'as_of': 'x', 'z': {k: 1.0 for k in pm.SIGNALS}, 'p': .3, 'decision': 'DOWN'}}}
        learned = pm.online_update(state, None, .1, 0, 'auto', 1)
        self.assertEqual(len(learned), 1)
        self.assertAlmostEqual(state['buckets']['1']['weights']['market'], 1.0 + .1 * .7)
        self.assertEqual(state['buckets']['0']['weights']['market'], 1.0)
        self.assertEqual(state['buckets']['2']['weights']['market'], 1.0)

    def test_unresolved_market_stays_pending(self, _):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'QL_LEARNING_DIR': tmp}):
            state = {'key': 'k', 'weights': {**{k: 1.0 for k in pm.SIGNALS}, 'bias': 0.0}, 'initial_weights': {}, 'history': [],
                     'pending': {'btc-updown-15m-1795374900': {'t': 9e12, 'as_of': 'x', 'z': {k: 1.0 for k in pm.SIGNALS}, 'p': .6, 'decision': 'UP'}}}
            self.assertEqual(pm.online_update(state, None, .1, 0, 'auto', 1), [])
            self.assertEqual(len(state['pending']), 1)


    def test_calibration_on_resolved_windows(self, _):
        import numpy as np
        from backend.engine.nodes.laya_nodes import laya_signal, laya_calibrate, laya_feedback
        rng = np.random.default_rng(0)
        fake = []
        for i in range(80):
            pm_ = float(rng.uniform(.2, .8))
            y = float(rng.random() < pm_)
            for b in range(3):          # three decision minutes per window
                fake.append({'slug': f'btc-updown-15m-{1795000000 + 900 * i}', 'start': 1795000000 + 900 * i, 'offset': 30 + 60 * b, 'bucket': b,
                             'decision_ts': 0, 'question': 'q', 'prices': [pm_] * 6, 'p_market': pm_, 'y': y, 'minutes_to_close': 14 - b})

        class Agent:
            def predict(self, state, questions):
                return {'answers': {'direction': {'probabilities': {'up': .8, 'down': .2}}}}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {'QL_LEARNING_DIR': tmp}), \
                mock.patch.object(pm, 'resolved_windows', return_value=fake), mock.patch('backend.engine.nodes.laya_nodes._agent', return_value=Agent()):
            frame, _r = pm.polymarket_source({'market': MARKET['slug']})
            sig, rep = laya_signal(frame, {'train_windows': 80})
            self.assertEqual(len(sig.attrs['polymarket_windows']), 240)
            self.assertEqual(len(rep['table']), 3)                       # one accuracy row per minute
            self.assertTrue(sig['Laya_P_up'].iloc[:-1].isna().all())
            self.assertAlmostEqual(rep['metrics']['laya_actual_p_up'], .8)
            cal, crep = laya_calibrate(sig, {'epochs': 200})
            wts = cal.attrs['updown_weights']
            self.assertEqual(sorted(cal.attrs['updown_weights_by_bucket']), ['0', '1', '2'])
            self.assertEqual(len(cal.attrs['updown_calibration']['buckets']), 3)
            self.assertGreater(wts['market'], 0)           # outcomes were drawn from the market price
            self.assertEqual(wts['imbalance'], 0.0)
            self.assertIn('logloss calibrada', crep['table'][0])
            self.assertIn('ventaja vs mercado', crep['table'][0])
            _o, d = pm.laya_updown(cal, {'weight_source': 'calibrated'})
            b = d['decision']['bucket']
            self.assertEqual(d['decision']['weights']['market'], cal.attrs['updown_weights_by_bucket'][str(b)]['market'])
            self.assertEqual(d['decision']['laya']['via'], 'Laya Signal')
            _o, d = pm.laya_updown(cal, {'weight_source': 'online', 'learning_key': 'seeded'})
            b = d['decision']['bucket']
            self.assertEqual(d['decision']['learning']['initial_weights']['market'], cal.attrs['updown_weights_by_bucket'][str(b)]['market'])
            with self.assertRaises(ValueError):
                laya_feedback(cal, {})



class PolymarketExplainTests(unittest.TestCase):
    def test_updown_reading_separates_bias_and_market(self):
        from backend.brain import explain
        rep = {'topic': 'laya_updown', 'metrics': {'p_up': .6}, 'table': [{}], 'decision': {
            'decision': 'UP', 'probability_up': .6, 'confidence': .6, 'actionable': True, 'horizon': '5m', 'bucket': 3, 'buckets_total': 5,
            'p_market': .4, 'contributions': {'market': -.2, 'target': .1, 'bias': .5}, 'laya': {'used': False},
            'bucket_stats': [{'minuto': 4, 'TE n': 24, 'logloss mercado': .49, 'logloss calibrada': .43, 'ventaja vs mercado': .06}]}}
        r = explain.rules(rep)
        self.assertIn('minuto 4 de 5', r['summary'])
        self.assertTrue(any('«market»' in f for f in r['findings']))          # bias is never named as a signal
        self.assertTrue(any('sesgo aprendido' in f for f in r['findings']))
        self.assertTrue(any('contradice al mercado' in c for c in r['cautions']))
        self.assertTrue(any('ventaja a favor' in f for f in r['findings']))

    def test_calibrate_reading_compares_with_market(self):
        from backend.brain import explain
        rows = [{'minuto': 1, 'TE n': 24, 'época VA': 300, 'convergió': 'no', 'acierto mercado': .5, 'acierto calibrada': .45,
                 'logloss mercado': .73, 'logloss calibrada': .70, 'ventaja vs mercado': .03},
                {'minuto': 2, 'TE n': 24, 'época VA': 80, 'convergió': 'sí', 'acierto mercado': .9, 'acierto calibrada': .9,
                 'logloss mercado': .20, 'logloss calibrada': .21, 'ventaja vs mercado': -.01}]
        rep = {'topic': 'laya_calibrate', 'splits': {'test': 24}, 'table': rows,
               'metrics': {'tramos': 2, 'ventanas': None, 'te_logloss_mercado': .5, 'te_logloss_calibrada': .48}}
        r = explain.rules(rep)
        self.assertIn('mejor que el mercado', r['summary'])
        self.assertTrue(any('menor log-loss que el mercado en 1 de 2' in f for f in r['findings']))
        self.assertTrue(any('más prudentes, no más certeras' in c for c in r['cautions']))
        self.assertTrue(any('no convergió' in c for c in r['cautions']))



class LayaTickerSyncTests(unittest.TestCase):
    def test_auto_ticker_follows_market_data(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        graph = json.loads((root / 'public/blueprints/laya-yfinance.json').read_text())['entries'][0]['graph']
        laya = next(n for n in graph['nodes'] if n['data']['kind'] == 'laya_signal')
        src = next(n for n in graph['nodes'] if n['data']['kind'] == 'source')
        self.assertEqual(laya['data']['params']['ticker'], 'auto')
        src['data']['params'].update({'source': 'yfinance', 'ticker': 'qqq'})
        self.assertEqual(DAGRunner(graph).source_ticker(laya['id']), 'QQQ')      # through Technical Features
        src['data']['params']['source'] = 'synthetic'
        self.assertEqual(DAGRunner(graph).source_ticker(laya['id']), 'SYNTHETIC')



def _dash_event(i, cat, vol, markets):
    return {'title': f'Evento {i}', 'slug': f'evento-{i}', 'volume24hr': vol, 'volume': vol * 3, 'openInterest': vol / 2,
            'endDate': '2099-01-01T00:00:00Z', 'tags': [{'label': cat}, {'label': 'Recurring'}], 'markets': markets}


def _dash_market(slug, price, change, vol, spread=.01, end='2099-01-01T00:00:00Z', group=None):
    return {'slug': slug, 'question': f'¿{slug}?', 'groupItemTitle': group, 'outcomes': '["Yes", "No"]', 'outcomePrices': json.dumps([str(price), str(1 - price)]),
            'bestBid': price - spread / 2, 'bestAsk': price + spread / 2, 'spread': spread, 'oneDayPriceChange': change, 'volume24hr': vol,
            'volumeNum': vol * 3, 'liquidityNum': vol, 'endDate': end, 'active': True, 'closed': False, 'acceptingOrders': True}


class PolymarketDashboardTests(unittest.TestCase):
    def test_dashboard_organises_markets(self):
        soon = (pm.pd.Timestamp.now(tz='UTC') + pm.pd.Timedelta(hours=3)).isoformat()
        events = [
            _dash_event(1, 'Politics', 5000, [_dash_market('a', .5, .02, 4000, group='Opción A'), _dash_market('b', .99, .4, 1000)]),
            _dash_event(2, 'Crypto', 2000, [_dash_market('btc-updown-5m-1795374900', .55, -.2, 1500, spread=.2), _dash_market('c', .2, .05, 500, end=soon)]),
        ]
        with mock.patch.object(pm, '_get', side_effect=lambda url, params=None, **_: events if params.get('offset') == 0 else []):
            frame, rep = pm.polymarket_dashboard({'pages': 2})
        d = rep['dashboard']
        self.assertEqual(d['kpis']['eventos'], 2)
        self.assertEqual(d['kpis']['mercados'], 4)
        self.assertEqual(d['kpis']['volumen_24h'], 7000)
        self.assertEqual(d['categories'][0]['categoría'], 'Politics')            # generic tags (Recurring) ignored
        self.assertEqual(d['top_markets'][0]['mercado'], 'Evento 1: Opción A')   # grouped markets keep their event
        self.assertNotIn('¿b?', [r['mercado'] for r in d['movers']])            # 99 % market is near-settled
        self.assertEqual(d['movers'][0]['cambio_24h'], -.2)
        self.assertEqual([r['mercado'] for r in d['ending_soon']], ['¿c?'])
        self.assertEqual(d['updown'][0]['ventana'], '5m')
        self.assertEqual(len(frame), 4)
        from backend.brain import explain
        r = explain.rules({**rep, 'title': 'd'})
        self.assertIn('2 eventos y 4 mercados', r['summary'])
        self.assertTrue(any('Politics' in f for f in r['findings']))

    def test_market_dashboard_from_connected_market(self):
        trades = {'data': [
            {'timestamp': 1795374960, 'side': 'BUY', 'outcome': 'Up', 'price': .6, 'size': 100, 'proxy_wallet': '0xabc', 'name': 'someone'},
            {'timestamp': 1795374950, 'side': 'BUY', 'outcome': 'Down', 'price': .4, 'size': 50, 'proxy_wallet': '0xdef'},
            {'timestamp': 1795374900, 'side': 'SELL', 'outcome': 'Down', 'price': .4, 'size': 25, 'proxy_wallet': '0x123'}]}
        holders = {'data': [{'token_id': '111', 'holders': [{'amount': 600, 'outcome_index': 0, 'name': 'whale', 'proxy_wallet': '0x1'}, {'amount': 400, 'outcome_index': 0}]}]}

        def get(url, params=None, **kw):
            if url.endswith('/v2/trades'):
                return trades
            if url.endswith('/v2/holders'):
                return holders
            return fake_get(url, params, **kw)
        with mock.patch.object(pm, '_get', side_effect=get):
            frame, _r = pm.polymarket_source({'market': MARKET['slug']})
            out, rep = pm.polymarket_dashboard({}, None, frame)
        d = rep['dashboard']
        self.assertEqual(d['mode'], 'market')
        self.assertEqual(d['kpis']['operaciones'], 3)
        # BUY Up (60 $) and SELL Down (10 $) push toward Up; BUY Down (20 $) toward Down
        self.assertAlmostEqual(d['kpis']['presión_up_usd'], 70.0)
        self.assertAlmostEqual(d['kpis']['presión_down_usd'], 20.0)
        self.assertEqual([t['hacia'] for t in d['trades']], ['UP', 'DOWN', 'UP'])
        self.assertNotIn('proxy_wallet', json.dumps(d))                       # no wallets or names leave the backend
        self.assertNotIn('whale', json.dumps(d))
        self.assertAlmostEqual(d['holders'][0]['mayor_tenedor_pct_top'], .6)
        self.assertTrue(d['depth'])
        self.assertEqual(sorted(d['book']), ['Down', 'Up'])
        from backend.brain import explain
        r = explain.rules({**rep, 'title': 'm'})
        self.assertIn('Bitcoin Up or Down', r['summary'])
        self.assertTrue(any('empujó hacia «Up»' in f for f in r['findings']))
        self.assertTrue(any('un solo tenedor' in c for c in r['cautions']))

    def test_dashboard_rejects_non_polymarket_input(self):
        with self.assertRaises(ValueError):
            pm.polymarket_dashboard({}, None, pm.pd.DataFrame({'Close': [1.0, 2.0]}))

    def test_dashboard_validates_params(self):
        with self.assertRaises(ValueError):
            pm.polymarket_dashboard({'category': 'nope'})


if __name__ == '__main__':
    unittest.main()
