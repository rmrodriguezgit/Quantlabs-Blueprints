"""Política de tres niveles del nodo LAYA UP/DOWN."""
import unittest

from backend.engine.nodes.polymarket_nodes import execution_policy, laya_question


class Policy(unittest.TestCase):
    def test_tiers(self):
        cases = [((.85, .2), 'EXECUTE'), ((.85, .4), 'ESCALATE'), ((.65, .1), 'ESCALATE'), ((.45, .1), 'CANCEL'), ((.9, None), 'EXECUTE')]
        for (c, r), want in cases:
            self.assertEqual(execution_policy(c, r, .8, .5, .3)[0], want)

    def test_risk_question(self):
        self.assertEqual(set(laya_question('5m')), {'direction'})
        q = laya_question('5m', risk='x')
        self.assertEqual(q['high_volatility_risk']['type'], 'noul')


if __name__ == '__main__':
    unittest.main()
