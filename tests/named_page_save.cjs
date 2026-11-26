const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const ts = require('../web/node_modules/typescript');
const cleanup=[];
const scope={exports:{}, require:()=>({useRef:value=>({current:value}),useLayoutEffect:fn=>cleanup.push(fn())})};
vm.runInNewContext(ts.transpileModule(fs.readFileSync('web/src/lib/pageSave.ts','utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS}}).outputText,scope);
(async()=>{
 const api=scope.exports;
 await assert.rejects(api.saveNamedPage(),/named file/);
 const order=[];
 api.registerPageSave('mapping',async()=>order.push('mapping'));
 api.useNamedPageSave(async()=>{await api.saveEntirePage();order.push('named file');return 'helmet_sit_DP';});
 assert.equal(order.length,0); // registration and editing never autosave
 assert.equal(await api.saveNamedPage(),'helmet_sit_DP');
 assert.deepEqual(order,['mapping','named file']);
 cleanup.forEach(fn=>fn());
 await assert.rejects(api.saveNamedPage(),/named file/);
 console.log('Named page save ordering, explicit save and unmount checks passed');
})().catch(e=>{console.error(e);process.exitCode=1;});
