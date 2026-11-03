import unittest
from backend.engine.dag_runner import DAGRunner

class PayloadConnections(unittest.TestCase):
    def graph(self,source,target):
        return {'nodes':[{'id':'a','data':{'kind':source,'params':{}}},{'id':'b','data':{'kind':target,'params':{}}}], 'edges':[{'id':'ab','source':'a','target':'b','sourceHandle':'tensor','targetHandle':'tensor'}]}

    def test_reports_rejected_before_execution(self):
        for source in ('curve_fit','study','prepare'):
            for target in ('risk','backtest'):
                with self.subTest(source=source,target=target),self.assertRaisesRegex(ValueError,'Incompatible data'):
                    DAGRunner(self.graph(source,target))

    def test_supported_payloads(self):
        for source,target in [('curve_fit','plot'),('study','plot'),('prepare','plot'),('prepare','model'),('model','backtest'),('model','risk'),('backtest','risk')]:
            with self.subTest(source=source,target=target):
                self.assertEqual(DAGRunner(self.graph(source,target)).topological_sort(),['a','b'])
