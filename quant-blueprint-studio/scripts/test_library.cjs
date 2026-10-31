const assert = require('node:assert/strict');
const {execFileSync} = require('node:child_process');
const {mkdtempSync,rmSync,readFileSync} = require('node:fs');
const {tmpdir} = require('node:os');
const path = require('node:path');
const temporary=mkdtempSync(path.join(tmpdir(),'quantlabs-library-'));
try {
  execFileSync(process.execPath,[require.resolve('typescript/bin/tsc'),'src/renderer/library.ts','--outDir',temporary,'--module','commonjs','--target','es2022','--skipLibCheck','--esModuleInterop'],{stdio:'inherit'});
  const {parseLibrary,isGraph,readSaved,storeSaved}=require(path.join(temporary,'library.js'));
  const {networkBindings,syncNetworkInputs}=require(path.join(temporary,'types/blueprint.js'));
  const original=JSON.parse(readFileSync('public/blueprints/library.json','utf8'));
  assert.equal(parseLibrary(original).length,15);
  assert.equal(parseLibrary(original).filter(e=>e.infographic).length,10);
  for(const entry of original.entries) {
    assert.ok(isGraph(entry.graph),entry.id);
    assert.equal(networkBindings(syncNetworkInputs(entry.graph)).find(b=>b.issue),undefined,entry.id);
    assert.deepEqual(parseLibrary(JSON.parse(JSON.stringify({...original,entries:[entry]})))[0],entry);
  }
  assert.equal(parseLibrary(original.entries[0].graph)[0].area,'Personal');
  assert.throws(()=>parseLibrary({...original,schemaVersion:2}),/version/);
  const invalid=structuredClone(original);invalid.entries[0].graph.edges[0].target='missing';
  assert.throws(()=>parseLibrary(invalid),/invalid/);
  const cyclic=structuredClone(original.entries[0].graph);
  cyclic.edges.push({id:'cycle',source:'features',target:'features',sourceHandle:'frame',targetHandle:'frame'});
  assert.equal(isGraph(cyclic),false);
  const broken=structuredClone(original);broken.entries[0].graph.nodes[0].data.params.years=Infinity;
  assert.equal(isGraph(broken.entries[0].graph),false);
  const future=structuredClone(original);future.entries[0].infographic='../../private.png';
  assert.throws(()=>parseLibrary(future),/invalid/);
  const memory=new Map();global.localStorage={getItem:k=>memory.get(k)??null,setItem:(k,v)=>memory.set(k,v),removeItem:k=>memory.delete(k)};
  storeSaved(original.entries);assert.deepEqual(readSaved(),original.entries);
  storeSaved([]);assert.deepEqual(readSaved(),[]);
  console.log('Library checks passed: 15 graphs, 10 references, versioned/legacy roundtrip, persistent storage, malformed pins, dimensions and metadata.');
} finally {rmSync(temporary,{recursive:true,force:true});}
