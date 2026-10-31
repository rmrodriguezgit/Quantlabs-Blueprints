const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {mkdtempSync,rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const path = require('node:path');
const temporary=mkdtempSync(path.join(tmpdir(),'quantlabs-graph-'));
try {
 execFileSync(process.execPath,[require.resolve('typescript/bin/tsc'),'src/renderer/types/blueprint.ts','--outDir',temporary,'--module','commonjs','--target','es2022','--skipLibCheck','--esModuleInterop'],{stdio:'inherit'});
 const {networkExample,networkBindings,syncNetworkInputs,featureOptions}=require(path.join(temporary,'blueprint.js'));
 let graph=networkExample();
 let binding=networkBindings(graph)[0];
 assert.equal(binding.summary,'16 inputs → 64 ReLU → 32 ReLU → 1 output');
 assert.equal(binding.issue,undefined);
 const prep=graph.nodes.find(n=>n.id===binding.prepareId);
 prep.data.params.feature_columns=prep.data.params.feature_columns.split(',').slice(0,15).join(',');
 graph=syncNetworkInputs(graph);binding=networkBindings(graph)[0];
 assert.equal(binding.selected.length,15);assert.equal(binding.declared,15);assert.equal(binding.issue,undefined);
 graph.nodes.find(n=>n.id===binding.inputId).data.params={input_features:16,auto_features:false};
 graph=syncNetworkInputs(graph);binding=networkBindings(graph)[0];
 assert.match(binding.issue,/selected 15/);
 const features=graph.nodes.find(n=>n.data.kind==='features');features.data.params.use_vwap=false;
 assert.equal(featureOptions(graph,features.id).includes('VWAP'),false);
 assert.match(networkBindings(graph)[0].issue,/VWAP/);
 console.log('Graph checks passed: architecture summary, automatic dimensions, manual mismatch, disabled indicators.');
} finally {rmSync(temporary,{recursive:true,force:true});}
