"""Behavioral checks for fitting boundaries, transforms and inverse target units."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from backend.engine.nodes.scaling_nodes import fit_scaler
from backend.engine.nodes.stats_nodes import prepare
from backend.engine.nodes.study_nodes import scaled, samples, study
from backend.engine.nodes.curve_nodes import monthly_curve, curve_fit
from backend.engine.nodes.data_nodes import source, features
from backend.engine.nodes.dl_nodes import train
from backend.engine.dag_runner import DAGRunner


class ScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.frame=features(source({'years':3}),{'lags':0})
        cls.monthly=monthly_curve(source({'years':15}),{})

    def test_standard_scaler_formula_constant_and_inverse(self) -> None:
        x=np.array([[1.,100.,7.],[2.,200.,7.],[6.,900.,7.],[10.,500.,7.]])
        scaler,audit=fit_scaler(x,['a','b','constant'])
        expected=(x-x.mean(axis=0))/np.where(x.std(axis=0)==0,1,x.std(axis=0,ddof=0))
        np.testing.assert_allclose(scaler.transform(x),expected,atol=1e-14)
        np.testing.assert_allclose(scaler.inverse_transform(expected),x)
        np.testing.assert_allclose(scaler.transform(x).mean(0),0,atol=1e-14)
        np.testing.assert_allclose(scaler.transform(x).std(0),[1,1,0],atol=1e-14)
        self.assertTrue(audit['columns'][2]['constant'])
        self.assertEqual(scaler.scale_[2],1)
        # Held-out observations use the same statistics, and need not lie in [-1,1].
        heldout=np.array([[1000.,-500.,7.]])
        np.testing.assert_allclose(scaler.transform(heldout),(heldout-scaler.mean_)/scaler.scale_)
        self.assertGreater(abs(scaler.transform(heldout)[0,0]),1)

    def test_invalid_scaler_and_nonfinite_training(self) -> None:
        with self.assertRaisesRegex(ValueError,'standard or minmax'):fit_scaler(np.ones((3,1)),['x'],'standrad')
        with self.assertRaisesRegex(ValueError,'finite'):fit_scaler(np.array([[np.nan]]),['x'])
        with self.assertRaisesRegex(ValueError,'feature names'):fit_scaler(np.ones((3,2)),['x'])

    def test_preparation_fits_unique_training_input_bars_only(self) -> None:
        p={'lookback':20,'feature_columns':'Close,LogReturn,RSI','scaler':'standard'}
        prepared=prepare(self.frame,p)
        stop=20+len(prepared.y_train)-1
        raw=self.frame[['Close','LogReturn','RSI']].to_numpy()
        np.testing.assert_allclose(prepared.scaler.mean_,raw[:stop].mean(0))
        self.assertEqual(prepared.scaler.n_samples_seen_,stop)
        # The first row outside input-fitting scope is already used by the last TR label;
        # its input must still be excluded from fitting X.
        changed=self.frame.copy()
        changed.iloc[stop:,changed.columns.get_loc('Close')]+=10000
        other=prepare(changed,p)
        np.testing.assert_array_equal(prepared.scaler.mean_,other.scaler.mean_)
        np.testing.assert_array_equal(prepared.x_train,other.x_train)
        self.assertFalse(np.array_equal(prepared.x_val,other.x_val))
        np.testing.assert_allclose(prepared.x_val[0],prepared.scaler.transform(raw[len(prepared.y_train):len(prepared.y_train)+20]),rtol=1e-6,atol=1e-7)

    def test_window_scaling_matches_unique_bar_reference(self) -> None:
        raw=np.column_stack([np.arange(100,dtype=float)**2,np.sin(np.arange(100))])
        windows=np.stack([raw[i:i+7] for i in range(94)])
        audits=[]
        output=scaled(windows,40,audits,['quadratic','sine'])
        reference=StandardScaler().fit(raw[:46])
        np.testing.assert_allclose(output,reference.transform(windows.reshape(-1,2)).reshape(windows.shape),rtol=1e-6,atol=1e-7)
        self.assertEqual(audits[0]['fit_rows'],46)
        changed=windows.copy();changed[40:]+=1e6
        np.testing.assert_array_equal(output[:40],scaled(changed,40)[:40])
        with self.assertRaises(ValueError):scaled(windows,0)

    def test_coordinate_interval_invariance_with_normalization(self) -> None:
        small=self.monthly.copy();small['Coordinate']/=2
        for variant in ('dense','polynomial_net'):
            p={'variant':variant,'widths':'8,4','epochs':4,'normalize':True,'evaluation':'temporal'}
            first=curve_fit(self.monthly,p);second=curve_fit(small,p)
            np.testing.assert_allclose([r['fitted'] for r in first['charts'][0]['rows']],[r['fitted'] for r in second['charts'][0]['rows']],rtol=1e-6,atol=1e-7)
            self.assertAlmostEqual(first['metrics']['MSE'],second['metrics']['MSE'],places=10)

    def test_polynomial_checkbox_and_fit_data(self) -> None:
        records=[];original=StandardScaler.fit
        def recording(scaler,x,*args,**kwargs):
            records.append(np.asarray(x).copy())
            return original(scaler,x,*args,**kwargs)
        params={'variant':'polynomial_net','epochs':2,'evaluation':'temporal','normalize':True,'degree':5}
        a=int(len(self.monthly)*.65)
        with patch.object(StandardScaler,'fit',new=recording):
            report=curve_fit(self.monthly,params)
        coordinate=self.monthly.Coordinate.to_numpy()[:a]
        self.assertEqual(len(records),1)
        np.testing.assert_allclose(records[0],np.column_stack([coordinate**d for d in range(1,6)]))
        self.assertEqual(report['scaling_audits'][0]['fit_rows'],a)
        with patch.object(StandardScaler,'fit',new=recording):
            disabled=curve_fit(self.monthly,{**params,'normalize':False})
        self.assertEqual(len(records),1,'Disabled normalization must not fit an input scaler.')
        self.assertEqual(disabled['scaling_audits'][0]['method'],'Disabled')
        self.assertEqual(disabled['table'][0]['X scaler'],'Disabled')
        with self.assertRaisesRegex(ValueError,'boolean'):
            curve_fit(self.monthly,{**params,'normalize':'false'})

    def test_original_branches_remain_explicitly_unscaled(self) -> None:
        for widths in ('256','256,256,256'):
            report=curve_fit(self.monthly,{'variant':'dense','epochs':1,'widths':widths,'normalize':False})
            self.assertEqual(report['scaling_audits'][0]['method'],'Disabled')
            self.assertEqual(report['scaling_audits'][1]['method'],'Center only')
            self.assertIn('in-sample',report['scaling_audits'][1]['scope'])

    def test_heldout_changes_do_not_affect_curve_scalers(self) -> None:
        a=int(len(self.monthly)*.65)
        changed=self.monthly.copy();changed.iloc[a:,changed.columns.get_loc('Coordinate')]+=200
        changed.iloc[a:,changed.columns.get_loc('Close')]+=100
        p={'variant':'polynomial_net','epochs':2,'evaluation':'temporal'}
        first=curve_fit(self.monthly,p);second=curve_fit(changed,p)
        self.assertEqual(first['scaling_audits'],second['scaling_audits'])

    def test_hpo_refits_each_fold_and_binary_target_stays_unscaled(self) -> None:
        p={'topic':'hyperparameters','epochs':1,'trials':2,'lookback':5}
        s=samples(self.frame,p)
        result=study(self.frame,p)
        folds=[a for a in result['scaling_audits'] if a['method']=='StandardScaler' and 'CV' in a['scope']]
        self.assertEqual(len(folds),4)
        expected=[int(s.train*f)+4 for _ in range(2) for f in (.5,.75)]
        self.assertEqual([a['fit_rows'] for a in folds],expected)
        classification=study(self.frame,{'topic':'regularization','epochs':1})
        self.assertEqual(classification['scaling_audits'][-1]['method'],'Disabled')
        self.assertIn('0/1',classification['scaling_audits'][-1]['note'])

    def test_generative_scalers_ignore_heldout_and_asset_ids(self) -> None:
        changed=self.frame.copy();a=int(len(changed)*.65)
        changed.iloc[a:,changed.columns.get_loc('LogReturn')]*=5
        changed.iloc[a:,changed.columns.get_loc('Volume')]*=10
        p={'topic':'generative','epochs':1}
        first=study(self.frame,p);second=study(changed,p)
        self.assertEqual(first['scaling_audits'],second['scaling_audits'])
        self.assertEqual([c['name'] for c in first['scaling_audits'][0]['columns']],['log_volatility','momentum','log_volume'])

    def test_curve_errors_undo_target_centering(self) -> None:
        result=curve_fit(self.monthly,{'variant':'dense','widths':'8','epochs':2,'evaluation':'temporal'})
        mean=result['scaling_audits'][1]['columns'][0]['center']
        predicted=np.array([row['fitted'] for row in result['charts'][0]['rows']])
        count=result['splits']['test']
        expected=np.mean((self.monthly.Close.to_numpy()[-count:]-(predicted[-count:]+mean))**2)
        self.assertAlmostEqual(result['metrics']['MSE'],float(expected),places=14)

    def test_model_errors_and_predictions_are_in_original_target_units(self) -> None:
        # Changing only test targets by a constant leaves predictions fixed;
        # reported error must match direct subtraction in original units.
        data=prepare(self.frame,{'lookback':3,'feature_columns':'LogReturn,RSI'})
        result=train(data,{'architecture':'Dense','epochs':2,'hidden':8,'dropout':0})
        self.assertAlmostEqual(result.metrics['MSE'],float(np.mean((data.y_test-result.predicted)**2)),places=10)
        changed=deepcopy(data);changed.y_test+=.5
        other=train(changed,{'architecture':'Dense','epochs':2,'hidden':8,'dropout':0})
        np.testing.assert_array_equal(result.predicted,other.predicted)

    def test_temporal_template_enables_neural_scalers(self) -> None:
        root=Path(__file__).resolve().parents[2]
        bundle=json.loads((root/'public/blueprints/notebook-curve-temporal.json').read_text())
        models=[n['data']['params'] for n in bundle['entries'][0]['graph']['nodes'] if n['data']['kind']=='curve_fit']
        self.assertEqual(len(models),7)
        self.assertTrue(all(p['normalize'] for p in models if p['variant']!='ols'))

    def test_dag_emits_serializable_audits(self) -> None:
        root=Path(__file__).resolve().parents[2]
        entry=json.loads((root/'public/blueprints/notebook-curve-quick.json').read_text())['entries'][0]
        graph=deepcopy(entry['graph'])
        for node in graph['nodes']:
            if node['data']['kind']=='source':node['data']['params']['source']='synthetic'
            if node['data']['kind']=='curve_fit':node['data']['params']['epochs']=1
        result=DAGRunner(graph).execute()
        self.assertEqual(len(result['scaling_audits']),14)
        json.dumps(result['scaling_audits'],allow_nan=False)
        methods=[a['method'] for a in result['scaling_audits'] if a['title'].startswith('Input')]
        self.assertEqual(methods.count('StandardScaler'),4)
        self.assertEqual(methods.count('Disabled'),3)

if __name__=='__main__':unittest.main()
