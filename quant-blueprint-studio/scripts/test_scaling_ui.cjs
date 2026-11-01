const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {execFileSync}=require('node:child_process');
const React=require('react');
const {renderToStaticMarkup}=require('react-dom/server');
const directory=fs.mkdtempSync(path.join(process.cwd(),'.scaling-ui-'));
try {
 execFileSync(process.execPath,[require.resolve('typescript/bin/tsc'),'src/renderer/components/ScalingDock.tsx','--outDir',directory,'--rootDir','src/renderer','--jsx','react-jsx','--module','commonjs','--target','es2022','--skipLibCheck','--esModuleInterop'],{stdio:'inherit'});
 const {ScalingDock}=require(path.join(directory,'components/ScalingDock.js'));
 const base={id:'node',node_title:'Temporal MLP',title:'Input X',method:'StandardScaler',scope:'training',fit_rows:117,note:'Fit training only.',inverse_before_metrics:false,columns:[{name:'Coordinate',mean:2,variance:4,center:2,scale:2,transformed_mean:0,transformed_std:1,constant:false}]};
 const html=renderToStaticMarkup(React.createElement(ScalingDock,{audits:[base,{...base,title:'Input off',method:'Disabled',columns:[]},{...base,title:'Target y',method:'Center only',inverse_before_metrics:true}],onInspect:()=>{}}));
 for(const text of ['Normalización','StandardScaler','117 filas','training','Coordinate','Sin escalado','Center only','antes de calcular las métricas']) assert.ok(html.includes(text),text);
 assert.equal(renderToStaticMarkup(React.createElement(ScalingDock,{audits:[],onInspect:()=>{}})),'');
 console.log('Scaling UI checks passed: active/disabled X, target inverse units and legacy empty reports.');
} finally {fs.rmSync(directory,{recursive:true,force:true});}
