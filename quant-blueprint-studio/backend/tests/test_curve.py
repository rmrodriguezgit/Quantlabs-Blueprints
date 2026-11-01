from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.data_nodes import source
from backend.engine.nodes.curve_nodes import monthly_curve, curve_fit, vectors


class CurveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def test_monthly_last_and_coordinate(self) -> None:
        daily=source({'source':'synthetic','years':15})
        monthly=monthly_curve(daily,{'exclude_incomplete_month':False})
        np.testing.assert_allclose(monthly.Close,daily.Close.resample('ME').last())
        self.assertEqual(monthly.Coordinate.iloc[0],-2)
        self.assertEqual(monthly.Coordinate.iloc[-1],2)
        self.assertGreaterEqual(len(monthly),170)

    def test_safe_metric_and_polynomial(self) -> None:
        self.assertEqual(vectors(np.array([[1.],[2.]]),np.array([1.,2.]))['MSE'],0)
        with self.assertRaises(ValueError):vectors(np.array([1.,2.]),np.array([1.]))
        with self.assertRaises(ValueError):vectors(np.ones((2,2)),np.ones(4))
        x=np.linspace(-2,2,120)
        frame=pd.DataFrame({'Close':1.2+.03*x+.01*x**3,'Coordinate':x},index=pd.date_range('2016-01-31',periods=120,freq='ME'))
        report=curve_fit(frame,{'variant':'ols'})
        self.assertLess(report['metrics']['MSE'],1e-20)
        self.assertEqual(report['splits'],{'train':120,'validation':0,'test':0})
        self.assertIn('in-sample',report['scope'])

    def test_temporal_mean_and_checkpoint_ignore_test(self) -> None:
        frame=monthly_curve(source({'years':15}),{})
        params={'variant':'dense','widths':'8,4','epochs':3,'evaluation':'temporal'}
        changed=frame.copy();b=int(len(frame)*.8)
        changed.iloc[b:,changed.columns.get_loc('Close')]+=10
        a=curve_fit(frame,params);other=curve_fit(changed,params)
        self.assertEqual(a['charts'][2]['rows'],other['charts'][2]['rows'])
        self.assertEqual(a['notes'][0],other['notes'][0])
        self.assertNotEqual(a['metrics']['MSE'],other['metrics']['MSE'])

    def test_notebook_graph(self) -> None:
        root=Path(__file__).resolve().parents[2]
        entry=json.loads((root/'public/blueprints/notebook-curve-quick.json').read_text())['entries'][0]
        graph=deepcopy(entry['graph'])
        for node in graph['nodes']:
            if node['data']['kind']=='source':node['data']['params']['source']='synthetic'
            if node['data']['kind']=='curve_fit':node['data']['params']['epochs']=2
        report=DAGRunner(graph).execute()
        self.assertEqual(len(report['studies']),7)
        self.assertEqual(len(report['plots']),3)
        self.assertEqual(report['studies'][1]['table'][0]['parameters'],769)
        self.assertEqual(report['studies'][2]['table'][0]['parameters'],6)
        json.dumps(report,allow_nan=False)

if __name__=='__main__':unittest.main()
