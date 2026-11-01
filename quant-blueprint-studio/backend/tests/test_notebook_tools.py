from copy import deepcopy
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.curve_nodes import curve_fit, monthly_curve
from backend.engine.nodes.data_nodes import source
from backend.engine.nodes.dl_nodes import NetworkSpec, custom_network, describe_network
from backend.engine.nodes.plot_nodes import plot


class NotebookToolsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.frame=monthly_curve(source({'years':15}),{})

    def test_swish_layer_and_summary_preserve_state(self) -> None:
        spec=NetworkSpec(1,'last',[{'layer_type':'Dense','units':4,'activation':'Swish'},{'layer_type':'BatchNorm'},{'layer_type':'Dropout','rate':.2}],True)
        model,width=custom_network(1,1,spec)
        self.assertEqual(width,4)
        layer=next(m for m in model if isinstance(m,nn.SiLU))
        x=torch.tensor([-2.,0.,2.])
        torch.testing.assert_close(layer(x),x*torch.sigmoid(x))
        before=deepcopy(model.state_dict());rng=torch.random.get_rng_state()
        description=describe_network(model,torch.ones(2,1,1))
        self.assertTrue(model.training)
        for k,v in before.items():torch.testing.assert_close(model.state_dict()[k],v)
        torch.testing.assert_close(torch.random.get_rng_state(),rng)
        self.assertEqual(description['total_parameters'],16)
        self.assertIn('Swish',[r['operation'] for r in description['layers']])

    def test_curve_connected_network_matches_legacy_and_dimensions(self) -> None:
        p={'variant':'dense','widths':'256,256,128','activation':'Swish','optimizer':'Adam','epochs':2}
        layers=[{'layer_type':'Dense','units':w,'activation':'Swish'} for w in (256,256,128)]
        legacy=curve_fit(self.frame,p)
        connected=curve_fit(self.frame,{**p,'_network':NetworkSpec(1,'last',layers,True)})
        np.testing.assert_allclose([r['fitted'] for r in legacy['charts'][0]['rows']],[r['fitted'] for r in connected['charts'][0]['rows']],atol=1e-7)
        self.assertEqual(connected['architecture']['total_parameters'],99329)
        self.assertEqual(sum(r['parameters'] for r in connected['architecture']['layers']),99329)
        with self.assertRaisesRegex(ValueError,'expects 2'):curve_fit(self.frame,{**p,'_network':NetworkSpec(2,'last',layers,True)})
        with self.assertRaisesRegex(ValueError,'Network Output'):curve_fit(self.frame,{**p,'_network':NetworkSpec(1,'last',layers,False)})
        with self.assertRaisesRegex(ValueError,'OLS'):curve_fit(self.frame,{'variant':'ols','_network':NetworkSpec(1,'last',[],True)})
        poly=curve_fit(self.frame,{'variant':'polynomial_net','epochs':2,'_network':NetworkSpec(5,'last',[],True)})
        self.assertEqual(poly['architecture']['total_parameters'],6)

    def test_numeric_scatter_and_smooth_grid(self) -> None:
        result=curve_fit(self.frame,{'variant':'dense','widths':'8','epochs':2})
        chart=plot(result,{'chart_type':'scatter_line','x_column':'coordinate'},'p')
        self.assertEqual(chart['scatter_columns'],['actual'])
        self.assertEqual(len(chart['rows']),len(self.frame))
        self.assertEqual(len(chart['line_rows']),300)
        self.assertEqual(chart['x_key'],'coordinate')
        self.assertEqual(chart['line_rows'][0]['coordinate'],-2)
        frame=plot(self.frame,{'chart_type':'scatter','x_column':'Coordinate','plot_columns':'Close'},'p')
        self.assertEqual(frame['x_key'],'Coordinate')
        with self.assertRaisesRegex(ValueError,'numeric X'):plot(result,{'chart_type':'scatter_line'},'p')
        with self.assertRaisesRegex(ValueError,'Scatter columns'):plot(result,{'chart_type':'scatter_line','x_column':'coordinate','scatter_columns':'missing'},'p')

    def test_log_loss_skip_and_nonpositive_guard(self) -> None:
        result=curve_fit(self.frame,{'variant':'dense','widths':'8','epochs':4})
        chart=plot(result,{'study_chart':2,'y_scale':'log','skip_rows':2},'p')
        self.assertEqual(chart['y_scale'],'log')
        self.assertEqual(chart['rows'][0]['epoch'],3)
        self.assertEqual(len(chart['rows']),2)
        with self.assertRaisesRegex(ValueError,'positive'):plot(result,{'study_chart':0,'y_scale':'log'},'p')
        with self.assertRaisesRegex(ValueError,'Skip rows'):plot(result,{'study_chart':2,'skip_rows':4},'p')
        with self.assertRaisesRegex(ValueError,'Y scale'):plot(result,{'study_chart':2,'y_scale':'sqrt'},'p')

    def test_updated_notebook_graph_executes_connected_swish(self) -> None:
        root=Path(__file__).resolve().parents[2]
        graph=json.loads((root/'public/blueprints/notebook-curve-quick.json').read_text())['entries'][0]['graph']
        for node in graph['nodes']:
            if node['data']['kind']=='source':node['data']['params']['source']='synthetic'
            if node['data']['kind']=='curve_fit':node['data']['params']['epochs']=2
        runner=DAGRunner(graph);result=runner.execute()
        self.assertEqual(len(result['nodes']),17)
        capacity=next(r for r in result['studies'] if r['id']=='capacity')
        self.assertEqual(capacity['architecture']['total_parameters'],99329)
        self.assertEqual(result['plots'][0]['chart_type'],'scatter_line')
        self.assertEqual(result['plots'][2]['y_scale'],'log')
        json.dumps(result,allow_nan=False)

if __name__=='__main__':unittest.main()
