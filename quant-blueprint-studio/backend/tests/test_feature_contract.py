import unittest
import numpy as np
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.data_nodes import features,source

class FeatureContractTests(unittest.TestCase):
    def test_automatic_returns_and_volatility(self):
        raw=source({'source':'synthetic','years':2})
        result=features(raw,{'lags':0,'volatility_window':10})
        expected=np.log(raw.Close/raw.Close.shift(1))
        np.testing.assert_allclose(result.LogReturn,expected.loc[result.index])
        np.testing.assert_allclose(result.Volatility,expected.rolling(10).std().loc[result.index])
        np.testing.assert_allclose(result.CumulativeLogReturn,result.LogReturn.cumsum())

    def test_missing_frame_stops_before_source_executes(self):
        graph={'nodes':[{'id':'source','data':{'kind':'source','params':{}}},{'id':'features','data':{'kind':'features','label':'11 causal indicators','params':{}}}],'edges':[]}
        events=[]
        with self.assertRaisesRegex(ValueError,'Market Data'):
            DAGRunner(graph).execute(events.append)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['node'],'features')
        self.assertEqual(events[0]['status'],'error')

    def test_missing_ohlcv_and_close_only(self):
        raw=source({'source':'synthetic','years':2})[['Close']]
        with self.assertRaisesRegex(ValueError,'Missing columns'):features(raw,{})
        result=features(raw,{'use_vwap':False})
        self.assertIn('LogReturn',result)
        self.assertIn('Volatility',result)

    def test_volatility_bounds(self):
        with self.assertRaisesRegex(ValueError,'volatility_window'):
            features(source({'source':'synthetic','years':2}),{'volatility_window':1})
