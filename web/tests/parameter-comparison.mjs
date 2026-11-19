import test from 'node:test';import assert from 'node:assert/strict';import {mkdir,writeFile,rm} from 'node:fs/promises';import path from 'node:path';import {createServer} from 'vite';import {chromium} from 'playwright';
test('parameter comparison highlights recorded differences, preserves zero/false and distinguishes missing values',async()=>{
const root=process.cwd(),h=path.join(root,`.parameter-comparison-${process.pid}.local`);await mkdir(h);
await writeFile(path.join(h,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
await writeFile(path.join(h,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {Track2ComparisonSummary} from '../src/components/Track2ComparisonSummary';import '../src/index.css';
const rows=[1,2,3].map(n=>({model_package_arn:'arn:'+n,registry:'helmet',generation:'11',size:'s',version:String(n)}));
const audits={'arn:1':{run_number:1,parameters:{epochs:10,img_size:640,batch:16,seed:42,patience:0},hyperparams:{optimizer:'auto',lr0:0.01,warmup_epochs:3,cos_lr:false,degrees:0,freeze:[0,1],future_param:7},training_options:{single_cls:false}},'arn:2':{run_number:2,parameters:{epochs:10,img_size:640,batch:16,seed:42,patience:0},hyperparams:{optimizer:'auto',lr0:0.02,warmup_epochs:1000,cos_lr:false,degrees:0,freeze:[0,1]},training_options:{single_cls:false}}};
Object.values(audits).forEach(a=>{a.train_dataset={file_count:10};a.eval_dataset={file_count:5};});
createRoot(document.getElementById('root')).render(<Track2ComparisonSummary rows={rows} audits={audits} status="ready"/>);`);
let server,browser;try{server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage({viewport:{width:1200,height:900}});await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(h)}/index.html`);await page.getByRole('table').waitFor();
assert.equal(await page.locator('tbody > tr').count(),3);
const firstSettings=page.locator('tbody > tr').first().locator('td').nth(2);
assert.match(await firstSettings.locator('[data-parameter="warmup_epochs"]').innerText(),/Different/);
assert.match(await firstSettings.locator('[data-parameter="cos_lr"]').innerText(),/false/);
assert.doesNotMatch(await firstSettings.locator('[data-parameter="cos_lr"]').innerText(),/Different/);
assert.match(await firstSettings.locator('[data-parameter="patience"]').innerText(),/0/);
assert.match(await firstSettings.locator('[data-parameter="future_param"]').innerText(),/7/);
const missing=page.locator('tbody > tr').last().locator('[data-parameter="warmup_epochs"]');
assert.match(await missing.innerText(),/Not recorded/);assert.doesNotMatch(await missing.innerText(),/Different/);
await page.getByRole('checkbox',{name:/Show differences only/}).check();
assert.equal(await firstSettings.locator('[data-parameter]').count(),2);
assert.equal(await page.locator('tbody > tr').count(),3);
assert.equal(await page.locator('[data-parameter="future_param"]').count(),0);

}finally{await browser?.close();await server?.close();assert.equal(path.dirname(h),root);await rm(h,{recursive:true,force:true});}});
