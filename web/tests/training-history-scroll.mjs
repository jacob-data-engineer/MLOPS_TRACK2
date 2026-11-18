import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('history fits three complete sections, including expanded details, and scrolls to remaining runs', async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.history-scroll-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';
    import {Track2TrainingHistory} from '../src/components/Track2TrainingAuditPanel';import '../src/index.css';
    createRoot(document.getElementById('root')!).render(<Track2TrainingHistory/>);`);
  let server,browser;
  try{
    server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage({viewport:{width:1200,height:700}});
    const dataset={file_names:[],file_count:0,folder:'',s3_uri:''};
    const audits=Array.from({length:7},(_,i)=>({id:`audit-${i}`,run_number:100-i,status:'Succeeded',
      generation:'custom',size:'yolo11s',detector_slug:'test',submitted_at:'2026-11-18T00:00:00Z',
      parameters:{epochs:1,batch:16,img_size:640,seed:42},train_dataset:dataset,eval_dataset:dataset,base_model:{},model:null}));
    await page.route('**/track2/**',route=>route.fulfill({json:new URL(route.request().url()).pathname.endsWith('/pull-control')
      ? {config:{card_state:{training_audit:false}}} : {audits}}));
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    const region=page.getByRole('region',{name:'Training history'});
    await region.waitFor();
    await page.waitForFunction(()=>document.querySelector('[aria-label="Training history"]')?.style.maxHeight);
    const geometry=await region.evaluate(el=>{
      const bounds=el.getBoundingClientRect();const rows=[...el.children].map(x=>x.getBoundingClientRect());
      return {thirdBottom:rows[2].bottom,bottom:bounds.bottom,fourthTop:rows[3].top,overflow:el.scrollHeight>el.clientHeight};
    });
    assert.ok(Math.abs(geometry.thirdBottom-geometry.bottom)<2);
    assert.ok(geometry.fourthTop>geometry.bottom);
    assert.ok(geometry.overflow);
    const collapsedHeight=await region.evaluate(el=>el.clientHeight);
    await region.locator('[data-history-summary]').first().locator('button').first().click();
    await page.waitForFunction(height=>document.querySelector('[aria-label="Training history"]').clientHeight>height,collapsedHeight);
    const expanded=await region.evaluate(el=>({bottom:el.getBoundingClientRect().bottom,thirdBottom:el.children[2].getBoundingClientRect().bottom}));
    assert.ok(Math.abs(expanded.thirdBottom-expanded.bottom)<2);
    await region.evaluate(el=>{el.scrollTop=el.scrollHeight;});
    assert.ok(await region.evaluate(el=>el.scrollTop>0));
  }finally{
    await browser?.close();await server?.close();
    assert.equal(path.dirname(harness),root);await rm(harness,{recursive:true,force:true});
  }
});
