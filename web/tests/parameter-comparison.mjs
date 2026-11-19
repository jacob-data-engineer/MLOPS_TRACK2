import test from 'node:test';import assert from 'node:assert/strict';import {mkdir,writeFile,rm} from 'node:fs/promises';import path from 'node:path';import {createServer} from 'vite';import {chromium} from 'playwright';
test('parameter comparison highlights recorded differences, preserves zero/false and distinguishes missing values',async()=>{
const root=process.cwd(),h=path.join(root,`.parameter-comparison-${process.pid}.local`);await mkdir(h);
await writeFile(path.join(h,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
await writeFile(path.join(h,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {Track2ParameterComparison} from '../src/components/Track2ParameterComparison';import '../src/index.css';
const rows=[1,2,3].map(n=>({model_package_arn:'arn:'+n,registry:'helmet',generation:'11',size:'s',version:String(n)}));
const audits={'arn:1':{run_number:1,parameters:{epochs:10,img_size:640,batch:16,seed:42,patience:0},hyperparams:{optimizer:'auto',lr0:0.01,warmup_epochs:3,cos_lr:false,degrees:0,freeze:[0,1],future_param:7},training_options:{single_cls:false}},'arn:2':{run_number:2,parameters:{epochs:10,img_size:640,batch:16,seed:42,patience:0},hyperparams:{optimizer:'auto',lr0:0.02,warmup_epochs:1000,cos_lr:false,degrees:0,freeze:[0,1]},training_options:{single_cls:false}}};
createRoot(document.getElementById('root')).render(<Track2ParameterComparison rows={rows} audits={audits}/>);`);
let server,browser;try{server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage({viewport:{width:1200,height:900}});await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(h)}/index.html`);await page.getByRole('table').waitFor();
const warm=page.getByRole('row').filter({hasText:'Warmup epochs'});assert.match(await warm.innerText(),/Different/);assert.match(await warm.innerText(),/1000/);assert.match(await warm.innerText(),/Not recorded/);
const cosine=page.getByRole('row').filter({hasText:'Cosine schedule'});assert.doesNotMatch(await cosine.innerText(),/Different/);assert.equal(await cosine.getByRole('cell',{name:'false',exact:true}).count(),2);
const patience=page.getByRole('row').filter({hasText:'Patience'});assert.equal(await patience.getByRole('cell',{name:'0',exact:true}).count(),2);
assert.equal(await page.getByRole('row').filter({hasText:'future_param'}).count(),1);
await page.getByRole('checkbox',{name:/Show differences only/}).check();assert.equal(await page.locator('tbody tr').count(),2);assert.equal(await page.getByRole('row').filter({hasText:'future_param'}).count(),0);
}finally{await browser?.close();await server?.close();assert.equal(path.dirname(h),root);await rm(h,{recursive:true,force:true});}});
