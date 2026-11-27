const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),ts=require('../web/node_modules/typescript');
const scope={exports:{}};vm.runInNewContext(ts.transpileModule(fs.readFileSync('web/src/lib/experimentScores.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,scope);
const run={baseline_arn:'',variants:[{name:'one',run_id:'a'},{name:'two',run_id:'b',model_package_arn:'arn-b'}]};
const audits=[{id:'a',model:{model_package_arn:'arn-a'},metrics:{mAP50:0,precision:.8}},{id:'b',metrics:{mAP50:.4,recall:.7}}];
const models=[{model_package_arn:'arn-a',metrics:{mAP50:null}},{model_package_arn:'arn-b',metrics:{mAP50:.5,recall:null}}];
const result=scope.exports.experimentScores(run,models,audits);
assert.equal(result.length,2);assert.equal(result[0].values.mAP50,0);assert.equal(result[0].values.precision,.8);assert.equal(result[1].values.mAP50,.5);assert.equal(result[1].values.recall,.7);assert.equal(result[1].values.f1,null);
assert.equal(scope.exports.experimentScores({...run,baseline_arn:'base'},models,audits).length,3);
console.log('Audit fallback, missing ARN recovery, zero metrics, registry precedence and baseline-free comparison passed');
