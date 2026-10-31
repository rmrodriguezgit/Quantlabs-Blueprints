"""Deterministic synthetic tests: DAG validation, leakage and executable models."""
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient
from backend.server import app
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.data_nodes import source, features, merge_features
from backend.engine.nodes.stats_nodes import prepare
from backend.engine.nodes.dl_nodes import train, NetworkSpec, SequenceNet
from backend.engine.nodes.backtest_nodes import backtest
from backend.engine.nodes.plot_nodes import plot


def graph() -> dict:
    kinds = ['source', 'features', 'prepare', 'model', 'backtest']
    nodes = [{'id': str(i), 'data': {'kind': kind, 'params': {'years': 1} if i == 0 else {'epochs': 1, 'hidden': 8} if i == 3 else {}}} for i, kind in enumerate(kinds)]
    return {'nodes': nodes, 'edges': [{'source': str(i), 'target': str(i+1), 'sourceHandle': 'frame' if i < 2 else 'tensor', 'targetHandle': 'frame' if i < 2 else 'tensor'} for i in range(4)]}


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.frame = features(source({'years': 1}), {})

    def test_unordered_dag_executes(self) -> None:
        g = graph()
        g['nodes'].reverse()
        result = DAGRunner(g).execute()
        self.assertEqual(len(result['nodes']), 5)
        self.assertEqual(len(result['predictions']), len(result['backtest']))
        self.assertTrue(np.isfinite(result['metrics']['MSE']))
        self.assertTrue(set(row['signal'] for row in result['backtest']) <= {-1, 0, 1})

    def test_cycles_missing_nodes_and_pins(self) -> None:
        g = graph()
        # A self-loop with compatible tensor pins still must fail cycle validation.
        g['edges'][2] = {'source': '3', 'target': '3', 'sourceHandle': 'tensor', 'targetHandle': 'tensor'}
        with self.assertRaisesRegex(ValueError, 'Cycle'):
            DAGRunner(g).topological_sort()
        g = graph()
        g['edges'][0]['source'] = 'missing'
        with self.assertRaisesRegex(ValueError, 'missing'):
            DAGRunner(g)
        g = graph()
        g['edges'][0]['sourceHandle'] = 'tensor'
        with self.assertRaisesRegex(ValueError, 'pin'):
            DAGRunner(g)

    def test_scaler_does_not_see_test_data(self) -> None:
        original = prepare(self.frame, {})
        changed = self.frame.copy()
        changed.iloc[-20:, changed.columns.get_loc('Close')] *= 100
        modified = prepare(changed, {})
        np.testing.assert_array_equal(original.x_train, modified.x_train)
        np.testing.assert_array_equal(original.scaler.mean_, modified.scaler.mean_)
        expected = self.frame['LogReturn'].to_numpy()[-len(original.y_test):]
        np.testing.assert_allclose(original.y_test, expected)

    def test_all_architectures(self) -> None:
        prepared = prepare(self.frame, {})
        for architecture in ['Dense', 'CNN', 'LSTM', 'GRU', 'BiLSTM', 'BiGRU', 'CNN-LSTM', 'Autoencoder', 'DAE', 'DQN']:
            with self.subTest(architecture=architecture):
                result = train(prepared, {'architecture': architecture, 'hidden': 8, 'epochs': 1})
                self.assertEqual(result.predicted.shape, prepared.y_test.shape)
                self.assertTrue(np.isfinite(result.predicted).all())

    def test_costs_and_forecast_alignment(self) -> None:
        data = prepare(self.frame, {})
        result = train(data, {'epochs': 1, 'hidden': 8})
        result.predicted = np.ones_like(result.predicted)
        zero = backtest(result, {'fee_bps': 0, 'slippage_bps': 0})
        np.testing.assert_allclose([r['strategy'] for r in zero['backtest']], [r['market'] for r in zero['backtest']])
        costly = backtest(result, {'fee_bps': 20})
        self.assertLess(costly['trading']['total_return'], zero['trading']['total_return'])
        self.assertEqual(costly['trading']['turnover'], 2)

    def test_selectable_indicators_and_formulas(self) -> None:
        raw = source({'years': 1})
        all_features = features(raw, {})
        date = all_features.index[-1]
        tail = raw.loc[:date].tail(20)
        expected_vwap = (((tail.High + tail.Low + tail.Close) / 3) * tail.Volume).sum() / tail.Volume.sum()
        self.assertAlmostEqual(all_features.loc[date, 'VWAP'], expected_vwap)
        self.assertAlmostEqual(all_features.loc[date, 'SMA'], tail.Close.mean())
        self.assertAlmostEqual(all_features.loc[date, 'BBUpper'], tail.Close.mean() + 2 * tail.Close.std(ddof=0))
        np.testing.assert_allclose(all_features.MACDHistogram, all_features.MACD - all_features.MACDSignal)
        disabled = features(raw, {**{f'use_{key}': False for key in ('rsi','ema','sma','macd','vwap','bb')}, 'lags': 0})
        self.assertEqual(set(disabled.columns), {'Open','High','Low','Close','Volume','LogReturn','CumulativeLogReturn','Volatility'})
        # Future edits must not change any historical feature values.
        modified = raw.copy()
        modified.iloc[-1, modified.columns.get_loc('Close')] *= 2
        pd_features = features(modified, {})
        np.testing.assert_allclose(all_features.iloc[:-1].to_numpy(), pd_features.iloc[:-1].to_numpy())
        with self.assertRaisesRegex(ValueError, 'MACD'):
            features(raw, {'macd_fast': 30, 'macd_slow': 20})

    def test_cumulative_log_return_and_plot(self) -> None:
        frame = self.frame
        np.testing.assert_allclose(frame['CumulativeLogReturn'], frame['LogReturn'].cumsum())
        self.assertAlmostEqual(frame['CumulativeLogReturn'].iloc[0], frame['LogReturn'].iloc[0])
        chart = plot(frame, {'plot_source':'columns','plot_columns':'CumulativeLogReturn'}, 'cumulative')
        self.assertEqual(chart['series'],['CumulativeLogReturn'])
        np.testing.assert_allclose([row['CumulativeLogReturn'] for row in chart['rows']], frame['LogReturn'].cumsum())
        # The selected cumulative series can also be an input without changing the target.
        data = prepare(frame, {'feature_columns':'CumulativeLogReturn','lookback':1})
        self.assertEqual(data.feature_names,['CumulativeLogReturn'])
        self.assertEqual(data.target,'LogReturn')
        raw = source({'years':1})
        branches = merge_features(features(raw,{'use_vwap':False}), features(raw,{'vwap_window':40}))
        np.testing.assert_allclose(branches['CumulativeLogReturn'], branches['LogReturn'].cumsum())

    def test_feature_selection_and_custom_network(self) -> None:
        names = ['Open','High','Low','Close','Volume','LogReturn','Volatility','RSI','EMA','SMA','VWAP','MACD','MACDSignal','BBUpper','BBMiddle','BBLower']
        data = prepare(self.frame, {'feature_columns': ','.join(names), 'lookback': 1})
        self.assertEqual(data.x_train.shape[1:], (1, 16))
        spec = NetworkSpec(16, 'last', [{'layer_type':'Dense','units':64,'activation':'ReLU'}, {'layer_type':'Dense','units':32,'activation':'ReLU'}], True)
        model = SequenceNet(16, 1, {'_network': spec})
        linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]
        self.assertEqual([(layer.in_features, layer.out_features) for layer in linear], [(16,64),(64,32),(32,1)])
        result = train(data, {'architecture':'Custom','_network':spec,'epochs':1})
        self.assertTrue(np.isfinite(result.predicted).all())
        with self.assertRaisesRegex(ValueError, 'expects 16'):
            SequenceNet(15, 1, {'_network': spec})
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            prepare(self.frame, {'feature_columns':'Missing'})
        with self.assertRaisesRegex(ValueError, 'connected'):
            train(data, {'architecture':'Custom'})
        extended = NetworkSpec(16, 'flatten', [*spec.layers, {'layer_type':'Dropout','rate':.1}, {'layer_type':'BatchNorm'}, {'layer_type':'GELU'}], True)
        self.assertEqual(SequenceNet(16, 3, {'_network':extended})(torch.ones(2,3,16)).shape, (2,))

    def test_feature_branch_merge(self) -> None:
        raw = source({'years': 1})
        switches = {f'use_{key}':False for key in ('rsi','ema','sma','macd','vwap','bb')}
        left = features(raw, {**switches,'use_ema':True})
        right = features(raw, {**switches,'use_vwap':True})
        joined = merge_features(left, right)
        self.assertIn('EMA', joined)
        self.assertIn('VWAP', joined)
        self.assertEqual(len(joined.columns), len(set(joined.columns)))
        with self.assertRaisesRegex(ValueError, 'Conflicting'):
            merge_features(left, features(raw, {**switches,'use_ema':True,'ema_window':10}))

    def test_custom_graph_api_and_branching(self) -> None:
        from pathlib import Path
        import json
        example = json.loads((Path(__file__).parents[2] / 'examples/dense-16-64-32-1.json').read_text())
        for node in example['nodes']:
            if node['data']['kind'] == 'model':
                node['data']['params']['epochs'] = 1
        with TestClient(app) as client:
            response = client.post('/api/execute', json=example)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()['input_shape'], [1,16])
            self.assertEqual(len(response.json()['features']), 16)
        # Network output can branch into multiple trainers (shared architecture, independent weights).
        trainer = next(n for n in example['nodes'] if n['data']['kind']=='model')
        clone = {'id':'model-copy','data':trainer['data']}
        example['nodes'].append(clone)
        example['edges'].extend([{**edge,'target':'model-copy'} for edge in list(example['edges']) if edge['target']==trainer['id']])
        self.assertEqual(len(DAGRunner(example).execute()['nodes']), len(example['nodes']))

    def test_insufficient_rows_diagnostic_and_recovery(self) -> None:
        short = features(source({'years':1}), {'bb_window':252})
        with self.assertRaisesRegex(ValueError, '1 usable rows from 252 source rows') as error:
            prepare(short, {'lookback':20})
        self.assertIn('Increase Market Data history', str(error.exception))
        self.assertIn('train/validation/test=', str(error.exception))
        enough = features(source({'years':3}), {'bb_window':252})
        self.assertGreater(len(prepare(enough, {'lookback':20}).y_test), 3)
        with self.assertRaisesRegex(ValueError, 'Decomposition period'):
            prepare(self.frame.tail(30), {'lookback':1,'period':20})

    def test_forex_zero_volume_does_not_silently_discard_rows(self) -> None:
        raw = source({'years':1})
        raw['Volume'] = 0
        with self.assertRaisesRegex(ValueError,'Disable Enable VWAP'):
            features(raw,{})
        without_vwap = features(raw,{'use_vwap':False})
        self.assertGreater(len(without_vwap),200)
        self.assertNotIn('VWAP',without_vwap.columns)
        self.assertGreater(len(prepare(without_vwap,{}).y_test),3)

    def test_plot_sources_and_validation(self) -> None:
        data = prepare(self.frame,{})
        result = train(data, {'epochs':2,'hidden':8})
        market = backtest(result,{})
        cases = [(self.frame,'columns','Close,VWAP'),(data,'decomposition','TR,VA,TE'),
                 (result,'predictions','actual,predicted'),(result,'loss','train,validation'),
                 (market,'backtest','strategy,market'),(market,'signals','signal')]
        for value, mode, columns in cases:
            with self.subTest(mode=mode):
                chart = plot(value, {'plot_source':mode,'plot_columns':columns,'chart_type':'area'}, 'plot1')
                self.assertTrue(chart['rows'])
                self.assertEqual(chart['series'],columns.split(','))
                self.assertEqual(chart['chart_type'],'area')
        with self.assertRaisesRegex(ValueError,'unavailable'):
            plot(self.frame, {'plot_columns':'NotAColumn'}, 'plot1')
        with self.assertRaisesRegex(ValueError,'does not match'):
            plot(data, {'plot_source':'loss'}, 'plot1')
        with self.assertRaisesRegex(ValueError,'unique numeric'):
            plot(result, {'plot_source':'loss','plot_columns':'train,train'}, 'plot1')

    def test_progress_epochs_rows_and_plot_graph(self) -> None:
        g = graph()
        g['nodes'].extend([{'id':'plot-frame','data':{'kind':'plot','params':{'plot_columns':'Close,VWAP'}}},
                           {'id':'plot-equity','data':{'kind':'plot','params':{'plot_source':'backtest'}}}])
        g['edges'].extend([{'source':'1','target':'plot-frame','sourceHandle':'frame','targetHandle':'frame'},
                           {'source':'4','target':'plot-equity','sourceHandle':'tensor','targetHandle':'tensor'}])
        events = []
        result = DAGRunner(g).execute(events.append)
        self.assertEqual(len(result['plots']),2)
        self.assertTrue(any(e['status']=='epoch' and e['epoch']==1 for e in events))
        source_event = next(e for e in events if e['node']=='0' and e['status']=='complete')
        self.assertEqual(source_event['rows'],252)
        prep_event = next(e for e in events if e['node']=='2' and e['status']=='complete')
        self.assertIn('features',prep_event)
        self.assertTrue(all(n>=3 for n in prep_event['samples'].values()))
        g['nodes'][1]['data']['params']['bb_window'] = 252
        failed = []
        with self.assertRaisesRegex(ValueError,'Insufficient data'):
            DAGRunner(g).execute(failed.append)
        self.assertTrue(any(e['node']=='2' and e['status']=='error' for e in failed))

    def test_api(self) -> None:
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/health').status_code, 200)
            self.assertEqual(client.post('/api/execute', json={'nodes': []}).status_code, 422)
            self.assertEqual(client.post('/api/execute', json=graph()).status_code, 200)
            with client.websocket_connect('/ws/execute') as ws:
                ws.send_json(graph())
                events = []
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event.get('event') in ('result', 'error'):
                        break
                self.assertEqual(events[-1]['event'], 'result')
                self.assertEqual(sum(e.get('status') == 'complete' for e in events), 5)


if __name__ == '__main__':
    unittest.main()
