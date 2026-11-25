import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('explicit Save captures latest values and collapsed cards; edits never write; failures remain visible',async()=>{
 const root=process.cwd(),harness=path.join(root,`.page-save-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(harness,'main.tsx'),`import React,{useState} from 'react';import {createRoot} from 'react-dom/client';import {Track2DetectorsPanel} from '../src/components/Track2DetectorsPanel';import {PageSaveBar} from '../src/components/PageSaveBar';import {Card} from '../src/components/Card';import {usePageSnapshot} from '../src/lib/pageSnapshot';import '../src/index.css';
 function Stage(){const [value,setValue]=useState('prepared');usePageSnapshot('stage',{value},v=>setValue(v.value));return <input aria-label="Dataset stage" value={value} onChange={e=>setValue(e.target.value)}/>;}
 function Harness(){return <><PageSaveBar/><Track2DetectorsPanel/><Card title="Stage choices"><Stage/></Card></>;}createRoot(document.getElementById('root')).render(<Harness/>);`);
 let server,browser,config={card_state:{'model types and model names':true},default_training:{epochs:20,img_size:640,batch:16,seed:42,patience:0},default_detector:{slug:'helmet',architecture:'yolo26s'}},posts=[],fail=false;
 try{
  server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/track2/**',async route=>{
   const req=route.request(),p=new URL(req.url()).pathname;let body={};
   if(req.method()==='POST'){
    posts.push({path:p,body:req.postDataJSON()});
    assert.equal(p,'/track2/pull-control/config','Saving must not launch work');
    if(fail)return route.fulfill({status:503,json:{detail:'Storage unavailable'}});
    config={...config,...req.postDataJSON()};body={config};
   }else if(p==='/track2/pull-control')body={config};
   else if(p.endsWith('/detectors'))body={detectors:[{slug:'helmet',label:'Helmet',classes:{0:'helmet'},groups:{s:'yolo26s-helmet-detector'},groups_by_generation:{'26':{s:'yolo26s-helmet-detector'}}}],sizes:['s'],generations:[{generation:'26',label:'YOLO26',sizes:['s'],size_labels:{s:'Small'}}]};
   else if(p.endsWith('/buckets'))body={buckets:[]};
   await route.fulfill({json:body});
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
  await page.waitForFunction(()=>document.querySelector('input[aria-label="Epochs"]')?.value==='20');
  await page.getByLabel('Epochs',{exact:true}).fill('37');
  await page.getByLabel('Dataset stage').fill('saved-dataset-123');
  await page.getByRole('heading',{name:'Stage choices',exact:true}).locator('..').getByRole('button',{name:'Hide',exact:true}).click();
  await page.waitForTimeout(1100);
  assert.equal(posts.length,0,'typing, blur and collapse must not auto-save');
  await page.getByRole('button',{name:'Save',exact:true}).click();
  await page.getByRole('status').filter({hasText:'All page settings saved.'}).waitFor();
  assert.equal(config.default_training.epochs,37);
  assert.equal(config.card_state['stage choices'],false);
  assert.equal(Object.values(config.page_state).find(v=>v.value!==undefined).value,'saved-dataset-123');
  const written=posts.length;
  await page.reload();
  await page.waitForFunction(()=>document.querySelector('input[aria-label="Epochs"]')?.value==='37');
  await page.getByRole('heading',{name:'Stage choices',exact:true}).locator('..').getByRole('button',{name:'Show',exact:true}).click();
  assert.equal(await page.getByLabel('Dataset stage').inputValue(),'saved-dataset-123');
  assert.equal(posts.length,written,'reload and show must not save');
  await page.getByLabel('Epochs',{exact:true}).fill('41');
  // Click immediately, without relying on blur/debounce to persist the value.
  await page.getByRole('button',{name:'Save',exact:true}).click();
  await page.getByRole('status').filter({hasText:'All page settings saved.'}).waitFor();
  assert.equal(config.default_training.epochs,41);
  fail=true;
  await page.getByRole('button',{name:'Save',exact:true}).click();
  await page.getByRole('alert').waitFor();
  assert.equal(await page.getByRole('status').filter({hasText:'All page settings saved.'}).count(),0);
  assert.deepEqual(errors,[]);
 }finally{await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
