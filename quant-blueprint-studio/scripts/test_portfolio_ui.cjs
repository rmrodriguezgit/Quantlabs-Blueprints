const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const React=require('react');
const {renderToStaticMarkup}=require('react-dom/server');
const directory=fs.mkdtempSync(path.join(process.cwd(),'.portfolio-ui-'));
try{
 execFileSync(process.execPath,[require.resolve('typescript/bin/tsc'),'src/renderer/components/PortfolioDock.tsx','--outDir',directory,'--rootDir','src/renderer','--jsx','react-jsx','--module','commonjs','--target','es2022','--skipLibCheck','--esModuleInterop'],{stdio:'inherit'});
 const {PortfolioDock,portfolioMetrics}=require(path.join(directory,'components/PortfolioDock.js'));
 const metric=(name,roi)=>({name,roi,cagr:.08,var:.02,cvar:.03,sharpe:.7,sortino:.9,alpha:.01,beta:.8,volatility:.12,max_drawdown:-.1,ols_r2:.65,observations:100});
 const assets=[metric('AAA',.1),metric('BBB',-.02)];
 const allocations=[metric('equal_weight',.03),metric('max_sharpe',.04),metric('min_variance',.02),metric('custom',.07)];
 const report={id:'test',title:'Full portfolio',scope:'holdout / out of sample',start:'2025-01-01',end:'2025-05-20',confidence:.95,var_horizon:'1 trading period',risk_free_rate:0,periods_per_year:252,benchmark:'REFERENCE',assets:['AAA','BBB'],source:'synthetic',metrics:[...assets,...allocations],portfolio_metrics:allocations,asset_metrics:assets,benchmark_metrics:metric('REFERENCE',.05),weights:Object.fromEntries(allocations.map(row=>[row.name,{AAA:.5,BBB:.5}])),estimation:{start:'2024-01-01',end:'2024-12-31',observations:252,covariance:'shrinkage',long_only:true,max_weight:1},rebalance:'daily',cost_bps_per_traded_notional:0,curves:[]};
 assert.deepEqual(portfolioMetrics(report).map(row=>row.name),allocations.map(row=>row.name));
 const legacy={...report};delete legacy.portfolio_metrics;
 assert.deepEqual(portfolioMetrics(legacy).map(row=>row.name),allocations.map(row=>row.name));
 const html=renderToStaticMarkup(React.createElement(PortfolioDock,{portfolios:[report],reports:[]}));
 assert.ok(html.includes('Indicadores del portafolio completo'));
 assert.ok(html.includes('data-portfolio-strategy="custom"'));
 assert.equal((html.match(/data-portfolio-metric=/g)||[]).length,11);
 for(const key of ['roi','cagr','var','cvar','sharpe','sortino','alpha','beta','volatility','max_drawdown','ols_r2'])assert.ok(html.includes(`data-portfolio-metric="${key}"`));
 assert.match(html,/data-portfolio-metric="roi"[\s\S]*?<strong>7\.00%<\/strong>/);
 console.log('Portfolio UI checks passed: all 11 portfolio metrics, custom selection and legacy result compatibility.');
}finally{fs.rmSync(directory,{recursive:true,force:true});}
