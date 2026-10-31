"""Independent numeric checks, optimization constraints, holdout leakage and graph integration."""
from copy import deepcopy
import unittest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from backend.server import app
from backend.engine.dag_runner import DAGRunner
from backend.engine.nodes.metrics_nodes import performance, analyze
from backend.engine.nodes.portfolio_nodes import PortfolioData, portfolio_data, build_portfolio, optimize_weights, weighted_returns
from backend.engine.nodes.plot_nodes import plot


class PortfolioTests(unittest.TestCase):
    def test_metric_definitions_and_ols(self) -> None:
        index = pd.bdate_range('2024-01-01', periods=5)
        series = pd.Series([.1,-.05,.02,-.01,.03],index=index)
        values = series.to_numpy()
        metrics = performance(series,None,{'confidence':.8})
        expected_roi = float(np.prod(1+values)-1)
        self.assertAlmostEqual(metrics['roi'],expected_roi)
        self.assertAlmostEqual(metrics['cagr'],(1+expected_roi)**(252/5)-1)
        self.assertAlmostEqual(metrics['var'],.018)
        self.assertAlmostEqual(metrics['cvar'],.05)
        self.assertAlmostEqual(metrics['volatility'],values.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(metrics['sharpe'],values.mean()/values.std(ddof=1)*np.sqrt(252))
        self.assertAlmostEqual(metrics['sortino'],values.mean()/np.sqrt(np.mean(np.minimum(values,0)**2))*np.sqrt(252))
        self.assertIsNone(metrics['beta'])
        benchmark = pd.Series(np.linspace(-.02,.02,100), index=pd.bdate_range('2024-01-01',periods=100), name='Reference')
        fit = performance(.0002+1.4*benchmark,benchmark,{})
        self.assertAlmostEqual(fit['beta'],1.4)
        self.assertAlmostEqual(fit['alpha'],.0002*252)
        self.assertAlmostEqual(fit['ols_r2'],1.)
        risk = analyze(benchmark.to_frame('Asset'),benchmark,{},'Test','historical sample')
        self.assertEqual(risk.report['benchmark'],'Reference')

    def test_zero_denominators_and_invalid_returns(self) -> None:
        series = pd.Series(np.zeros(20),index=pd.bdate_range('2024-01-01',periods=20))
        metrics = performance(series,series,{})
        self.assertIsNone(metrics['sharpe'])
        self.assertIsNone(metrics['sortino'])
        self.assertIsNone(metrics['beta'])
        with self.assertRaisesRegex(ValueError,'finite simple'):
            performance(pd.Series([.1,-1.,.1]),None,{})

    def test_rebalancing_costs_and_buy_hold_math(self) -> None:
        asset = pd.DataFrame({'A':[.1,-.02,.03],'B':[.02,.04,-.01]},index=pd.bdate_range('2024-01-01',periods=3))
        weights = np.array([.6,.4])
        daily = weighted_returns(asset,weights,'daily',0)
        np.testing.assert_allclose(daily,asset.to_numpy()@weights)
        held = weighted_returns(asset,weights,'buy_hold',0)
        self.assertAlmostEqual(float(np.prod(1+held)),float(np.cumprod(1+asset.to_numpy(),axis=0)[-1]@weights))
        cost = weighted_returns(asset,weights,'daily',10)
        self.assertTrue(np.all(cost<=daily))
        self.assertAlmostEqual(float(cost.iloc[0]),float(daily.iloc[0])-.001)

    def test_optimization_constraints_and_min_variance(self) -> None:
        rng=np.random.default_rng(7)
        train=pd.DataFrame(rng.normal([.0006,.0004,.0003],[.012,.009,.006],(500,3)),columns=['A','B','C'])
        result=optimize_weights(train,252,0,.6,'sample')
        covariance=train.cov().to_numpy()*252
        equal=np.full(3,1/3)
        for weights in result.values():
            self.assertAlmostEqual(float(weights.sum()),1.)
            self.assertGreaterEqual(weights.min(),-1e-7)
            self.assertLessEqual(weights.max(),.6+1e-7)
        self.assertLessEqual(result['min_variance']@covariance@result['min_variance'],equal@covariance@equal+1e-8)
        mean=train.mean().to_numpy()*252
        sharpe=lambda w:float(mean@w/np.sqrt(w@covariance@w))
        self.assertGreaterEqual(sharpe(result['max_sharpe']),sharpe(equal)-1e-6)
        unconstrained = optimize_weights(train,252,0,1,'sample')['min_variance']
        analytic = np.linalg.solve(covariance,np.ones(3))
        analytic /= analytic.sum()
        np.testing.assert_allclose(unconstrained,analytic,atol=2e-5)
        unique = optimize_weights(train,252,0,1/3,'sample')
        np.testing.assert_allclose(unique['max_sharpe'],np.ones(3)/3)
        with self.assertRaisesRegex(ValueError,'infeasible'):
            optimize_weights(train,252,0,.2,'sample')

    def test_portfolio_holdout_and_custom_weights(self) -> None:
        data=portfolio_data({'tickers':'AAA,BBB,CCC','benchmark':'REFERENCE','years':1})
        result=build_portfolio(data,{'weights':'.5,.3,.2'})
        self.assertIn('custom',result.report['weights'])
        self.assertEqual(len(result.report['metrics']),7)
        self.assertEqual(result.report['scope'],'holdout / out of sample')
        split=int((len(data.prices)-1)*.7)
        changed=deepcopy(data)
        changed.prices.iloc[split+1:] *= 1.03
        modified=build_portfolio(changed,{'weights':'.5,.3,.2'})
        self.assertEqual(result.report['weights'],modified.report['weights'])
        self.assertNotEqual(result.report['curves'],modified.report['curves'])
        self.assertLess(result.report['estimation']['end'],result.report['start'])
        with self.assertRaisesRegex(ValueError,'sum to 1'):
            build_portfolio(data,{'weights':'.5,.5,.5'})

    def test_common_dates_and_invalid_assets(self) -> None:
        with self.assertRaisesRegex(ValueError,'unique tickers'):
            portfolio_data({'tickers':'SPY,SPY'})
        data=portfolio_data({'tickers':'AAA,BBB','benchmark':'REFERENCE','years':1})
        self.assertEqual(data.prices.index.tolist(),data.benchmark.index.tolist())
        self.assertEqual(data.prices.columns.tolist(),['AAA','BBB'])
        # Constancy cannot silently produce a Sharpe solution.
        with self.assertRaisesRegex(ValueError,'no variance'):
            optimize_weights(pd.DataFrame(np.zeros((100,2))),252,0,1,'sample')

    def test_portfolio_plots_and_palette(self) -> None:
        data=portfolio_data({'tickers':'AAA,BBB','years':1})
        result=build_portfolio(data,{})
        for mode in ['equity','cumulative_log','drawdown','weights','asset_prices']:
            chart=plot(result,{'plot_source':mode,'color':'Coral','title':'Named chart'},'plot-a')
            self.assertTrue(chart['rows'])
            self.assertEqual(chart['title'],'Named chart')
            self.assertEqual(chart['colors'][chart['series'][0]],'#f07777')
        chart=plot(result,{'plot_source':'equity','plot_columns':'max_sharpe,min_variance','series_colors':'Blue,Orange','series_labels':'Growth,Defensive'},'plot-a')
        self.assertEqual(chart['labels'],{'max_sharpe':'Growth','min_variance':'Defensive'})
        self.assertEqual(chart['colors']['min_variance'],'#f2ad65')
        with self.assertRaisesRegex(ValueError,'Unknown plot color'):
            plot(result,{'color':'Invisible'},'plot-a')

    def test_portfolio_graph_and_api(self) -> None:
        graph={'nodes':[{'id':'data','data':{'kind':'portfolio_data','params':{'tickers':'AAA,BBB,CCC','years':1}}},
                        {'id':'allocation','data':{'kind':'portfolio','params':{}}},
                        {'id':'risk','data':{'kind':'risk','params':{}}},
                        {'id':'plot','data':{'kind':'plot','params':{'plot_source':'equity'}}}],
               'edges':[{'source':'data','target':'allocation','sourceHandle':'portfolio','targetHandle':'portfolio'},
                        {'source':'allocation','target':'risk','sourceHandle':'portfolio','targetHandle':'portfolio'},
                        {'source':'allocation','target':'plot','sourceHandle':'portfolio','targetHandle':'portfolio'}]}
        with TestClient(app) as client:
            response=client.post('/api/execute',json=graph)
            self.assertEqual(response.status_code,200,response.text)
        result=response.json()
        self.assertEqual(len(result['portfolios']),1)
        self.assertEqual(len(result['risk_reports']),1)
        self.assertEqual(len(result['plots']),1)
        self.assertEqual(result['plots'][0]['origin'],'AAA,BBB,CCC')
        self.assertEqual(result['portfolios'][0]['benchmark'],'SPY')

    def test_risk_metrics_on_frame_and_backtest_graph(self) -> None:
        from test_engine import graph
        payload=graph()
        payload['nodes'].extend([{'id':'asset-risk','data':{'kind':'risk','params':{}}},
                                 {'id':'strategy-risk','data':{'kind':'risk','params':{}}}])
        payload['edges'].extend([{'source':'1','target':'asset-risk','sourceHandle':'frame','targetHandle':'frame'},
                                {'source':'4','target':'strategy-risk','sourceHandle':'tensor','targetHandle':'tensor'}])
        result=DAGRunner(payload).execute()
        self.assertEqual(len(result['risk_reports']),2)
        self.assertEqual(result['risk_reports'][1]['scope'],'strategy holdout / out of sample')
        self.assertEqual(len(result['risk_reports'][1]['metrics']),2)


if __name__=='__main__':
    unittest.main()
