const assert=require('node:assert/strict');
const {execFileSync}=require('node:child_process');
const {mkdtempSync,rmSync}=require('node:fs');
const {tmpdir}=require('node:os');
const path=require('node:path');
const temp=mkdtempSync(path.join(tmpdir(),'quantlabs-connect-'));
try {
execFileSync(process.execPath,[require.resolve('typescript/bin/tsc'),'src/renderer/connections.ts','--outDir',temp,'--module','commonjs','--target','es2022','--skipLibCheck','--esModuleInterop'],{stdio:'inherit'});
const {validConnection}=require(path.join(temp,'connections.js'));
const {createNode}=require(path.join(temp,'types/blueprint.js'));
const source=createNode('source',0),features=createNode('features',1),merge=createNode('merge',2),plot=createNode('plot',3),prepare=createNode('prepare',4);
const nodes=[source,features,merge,plot,prepare];
const c=(a,b,input='frame',output='frame')=>({source:a.id,target:b.id,sourceHandle:output,targetHandle:input});
const edge=(a,b,input='frame',output='frame')=>({...c(a,b,input,output),id:crypto.randomUUID()});
assert(validConnection({nodes,edges:[]},c(source,features)));
assert(!validConnection({nodes,edges:[]},c(source,source)));
assert(!validConnection({nodes,edges:[]},c(source,features,'config')));
assert(!validConnection({nodes,edges:[]},c(source,features,'tensor','tensor')));
assert(!validConnection({nodes,edges:[edge(source,features)]},c(merge,features)));
assert(!validConnection({nodes,edges:[edge(features,merge)]},c(merge,features)));
assert(validConnection({nodes,edges:[edge(source,merge)]},c(features,merge,'frame_b')));
assert(!validConnection({nodes,edges:[edge(source,plot)]},c(prepare,plot,'tensor','tensor')));
const connected={nodes,edges:[edge(source,features)]};
const disconnected={...connected,edges:[]};
assert(validConnection(disconnected,c(source,features)));
assert.equal(disconnected.nodes.length,connected.nodes.length);
for (const kind of ['curve_fit','study','prepare']) {
 const src=createNode(kind,0),risk=createNode('risk',1),backtest=createNode('backtest',2),model=createNode('model',3),chart=createNode('plot',4);
 const g={nodes:[src,risk,backtest,model,chart],edges:[]};
 assert(!validConnection(g,c(src,risk,'tensor','tensor')));
 assert(!validConnection(g,c(src,backtest,'tensor','tensor')));
 assert.equal(validConnection(g,c(src,model,'tensor','tensor')),kind==='prepare');
 assert(validConnection(g,c(src,chart,'tensor','tensor')));
}
for (const kind of ['model','backtest']) {
 const src=createNode(kind,0),risk=createNode('risk',1);
 assert(validConnection({nodes:[src,risk],edges:[]},c(src,risk,'tensor','tensor')));
}
console.log('Connection checks passed: compatible types, occupied pins, cycles, merge inputs, exclusive plots, reconnection.');
} finally {rmSync(temp,{recursive:true,force:true});}
