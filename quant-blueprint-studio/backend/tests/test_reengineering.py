from copy import deepcopy
from pathlib import Path
import json
import unittest
import numpy as np
import pandas as pd
import torch
from backend.engine.dag_runner import DAGRunner, PINS, INPUTS
from backend.engine.nodes.data_nodes import source,features
from backend.engine.nodes.stats_nodes import prepare
from backend.engine.nodes.ml_nodes import train_ml,ALGORITHMS
from backend.engine.nodes.dl_nodes import SequenceNet
from backend.engine.nodes.research_nodes import data_quality,quant_features
from backend.engine.nodes.plot_nodes import plot

class ReengineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.frame=features(source({'source':'synthetic','years':2}),{'lags':2})
        cls.prepared=prepare(cls.frame,{'lookback':3,'period':10,'feature_columns':'LogReturn,RSI,Volatility'})

    def test_ml_every_algorithm_and_test_isolation(self):
        for algorithm in (*ALGORITHMS,'Compare'):
            with self.subTest(algorithm=algorithm):
                original,report=train_ml(self.prepared,{'algorithm':algorithm,'estimators':10})
                other=deepcopy(self.prepared);other.y_test=other.y_test*50+1
                repeated,second=train_ml(other,{'algorithm':algorithm,'estimators':10})
                self.assertEqual(report['selected'],second['selected'])
                self.assertEqual(report['table'],second['table'])
                np.testing.assert_allclose(original.predicted,repeated.predicted)
                self.assertTrue(np.isfinite(original.predicted).all())
                self.assertEqual(len(original.predicted),len(other.y_test))

    def test_transformer_shapes_gradients_and_seed(self):
        params={'architecture':'Transformer','hidden':16,'heads':4,'layers':1,'dropout':0}
        torch.manual_seed(1);model=SequenceNet(3,5,params)
        x=torch.randn(4,5,3);result=model(x)
        self.assertEqual(tuple(result.shape),(4,))
        result.sum().backward();self.assertIsNotNone(model.encoder.projection.weight.grad)
        with self.assertRaisesRegex(ValueError,'divisible'):SequenceNet(3,5,{**params,'hidden':15})

    def test_quality_causal_and_quant_reference(self):
        frame=pd.DataFrame({'Close':[100.,np.nan,110.,99.,100.,101.]},index=pd.date_range('2020-01-01',periods=6))
        result=data_quality(frame,{'missing':'ffill','ffill_limit':1})
        self.assertEqual(result.Close.iloc[1],100)
        quant=quant_features(result,{'window':2,'periods_per_year':12})
        np.testing.assert_allclose(quant.LogReturn.iloc[1:],np.log(result.Close/result.Close.shift(1)).iloc[1:])
        self.assertAlmostEqual(quant.Drawdown.iloc[3],-.1)
        constant=quant_features(result.assign(Close=100.),{'window':2})
        self.assertTrue(constant.RollingSharpe.isna().all())
        with self.assertRaisesRegex(ValueError,'Duplicate'):data_quality(pd.concat([frame,frame]),{})

    def test_histogram_counts(self):
        value=plot(self.frame,{'chart_type':'histogram','plot_columns':'LogReturn','hist_bins':15},'p')
        self.assertEqual(value['chart_type'],'bar')
        self.assertEqual(len(value['rows']),15)
        self.assertEqual(sum(row['count'] for row in value['rows']),len(self.frame))
        with self.assertRaisesRegex(ValueError,'exactly one'):plot(self.frame,{'chart_type':'histogram','plot_columns':'Close,LogReturn'},'p')

    def test_shared_contract_registry(self):
        root=Path(__file__).resolve().parents[2]
        contracts=json.loads((root/'src/renderer/data/node-contracts.json').read_text())
        self.assertEqual(set(contracts['nodes']),set(PINS))
        for kind,node in contracts['nodes'].items():
            for handle,payload in node['extraInputs'].items():self.assertEqual(INPUTS[kind][handle],payload)
        self.assertNotIn('rl_agent',contracts['tensorSources']['risk'])
        self.assertIn('ml_model',contracts['tensorSources']['risk'])
