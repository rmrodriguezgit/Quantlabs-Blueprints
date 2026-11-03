"""Integration and holdout-isolation tests for every distributed example."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.dl_nodes import SequenceNet
from backend.engine.nodes.data_nodes import source, features
from backend.engine.nodes.study_nodes import samples, scaled, study, ConditionalGenerator, QNetwork

ROOT = Path(__file__).resolve().parents[2]


class LibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.entries = json.loads((ROOT/'public/blueprints/library.json').read_text())['entries']

    def test_all_distributed_graphs_execute(self) -> None:
        for entry in self.entries:
            with self.subTest(entry=entry['id']):
                graph=deepcopy(entry['graph'])
                for node in graph['nodes']:
                    if node['data']['kind']=='plot':node['data']['params']['skip_rows']=0
                    if node['data']['kind']=='source' and entry['id'].startswith('notebook-'):
                        node['data']['params']['source']='synthetic'
                    if node['data']['kind'] in ('study','model','curve_fit'):
                        node['data']['params']['epochs']=2
                        node['data']['params']['trials']=2
                result=DAGRunner(graph).execute()
                self.assertEqual(len(result['nodes']),len(graph['nodes']))
                json.dumps(result,allow_nan=False)
                if entry['id']=='notebook-keras-1-256-1':
                    report=result['studies'][0]
                    self.assertEqual(report['architecture']['total_parameters'],769)
                    self.assertEqual(len(result['plots']),3)
                    self.assertEqual(result['plots'][1]['x_key'],'coordinate')
                    self.assertEqual(len(result['plots'][1]['line_rows']),300)
                if entry.get('infographic') and not entry['id'].startswith('notebook-'):
                    report=result['studies'][0]
                    self.assertTrue(report['selected'])
                    self.assertTrue(report['metrics'])
                    self.assertTrue(all(report['splits'][k]>=8 for k in ('train','validation','test')))
                    self.assertEqual(len(report['features']),11)
                    self.assertGreaterEqual(len(report['charts']),2)

    def test_reference_integrity_and_bundled_copy(self) -> None:
        original=json.loads((ROOT/'public/infographics/manifest.json').read_text())
        files=[entry for item in original for entry in [item,*item.get('supplements',[])]]
        self.assertEqual(len(files),11)
        for entry in files:
            self.assertEqual(hashlib.sha256((ROOT/'public/infographics'/entry['asset']).read_bytes()).hexdigest(),entry['sha256'])
        public=json.loads((ROOT/'public/blueprints/library.json').read_text())
        bundled=json.loads((ROOT/'src/renderer/data/library.json').read_text())
        self.assertEqual(public,bundled)
        for entry in self.entries:
            self.assertEqual(json.loads((ROOT/f"public/blueprints/{entry['id']}.json").read_text())['entries'],[entry])

    def test_alignment_and_training_scale(self) -> None:
        frame=features(source({'years':2}),{'lags':0})
        s=samples(frame,{'lookback':20})
        np.testing.assert_allclose(s.y,10000*np.log(frame.Close.to_numpy()[20:]/frame.Open.to_numpy()[20:]))
        np.testing.assert_allclose(s.x[0,-1],frame[s.names].to_numpy()[19])
        changed=s.x.copy();changed[s.validation:]*=100
        np.testing.assert_array_equal(scaled(s.x,s.train)[:s.train],scaled(changed,s.train)[:s.train])

    def test_selection_does_not_consult_test(self) -> None:
        frame=features(source({'years':2}),{'lags':0})
        for topic in ('optimizers','regularization','hyperparameters','generative','double_dqn'):
            with self.subTest(topic=topic):
                params={'topic':topic,'epochs':2,'trials':2,'lookback':1}
                s=samples(frame,params)
                changed=frame.copy()
                start=len(frame)-len(s.y)+s.validation+2
                changed.iloc[start:,changed.columns.get_loc('Close')]*=1.2
                changed.iloc[start:,changed.columns.get_loc('LogReturn')]*=2
                original=study(frame,params); other=study(changed,params)
                self.assertEqual(original['selected'],other['selected'])
                if topic=='double_dqn':
                    self.assertEqual(original['charts'][1]['rows'],other['charts'][1]['rows'])
                else:
                    self.assertEqual(original['table'],other['table'])

    def test_bidirectional_final_states(self) -> None:
        x=torch.randn(3,20,11)
        for kind in ('BiLSTM','BiGRU'):
            net=SequenceNet(11,20,{'architecture':kind,'hidden':32,'dropout':0}).eval()
            with torch.no_grad():
                _,state=net.recurrent(x)
                h=state[0] if isinstance(state,tuple) else state
                expected=net.norm(torch.cat([h[-2],h[-1]],1))
                torch.testing.assert_close(net.encode(x),expected)

    def test_shapes_and_rejections(self) -> None:
        gen=ConditionalGenerator(16,4,True,3)
        conditions=torch.tensor([[0.,-1.,0.,0.],[0.,0.,0.,1.],[0.,1.,0.,2.]])
        self.assertEqual(tuple(gen(torch.randn(3,4),conditions).shape),(3,16))
        self.assertEqual(tuple(QNetwork(11)(torch.randn(3,12)).shape),(3,3))
        frame=features(source({'years':1}),{'lags':0})
        with self.assertRaisesRegex(ValueError,'epochs'):study(frame,{'topic':'feedforward','epochs':0})
        with self.assertRaisesRegex(ValueError,'horizon'):study(frame,{'topic':'generative','horizon':7})
        with self.assertRaisesRegex(ValueError,'feature'):study(frame,{'topic':'feedforward','feature_columns':'Future'})

if __name__=='__main__':unittest.main()
