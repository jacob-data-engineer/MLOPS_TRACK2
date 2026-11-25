import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';
test('PT View reads uploads and registered models and clears stale classes',async()=>{
 const root=process.cwd(),harness=path.join(root,`.pt-view-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {Track2PtViewPage} from '../src/routes/Track2PtViewPage';import '../src/index.css';createRoot(document.getElementById('root')).render(<Track2PtViewPage/>);`);
 let server,browser;
 try{
  server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();
  let calls=[];
  await page.route('**/track2/**',route=>{
   const p=new URL(route.request().url()).pathname;
   if(p.endsWith('/metrics'))return route.fulfill({json:{rows:[{model_package_arn:'arn:test',registry:'helmet',generation:'26',size:'s',version:'12'}]}});
   if(!p.includes('/pt-view/'))return route.fulfill({json:{}});
   calls.push(p);
   if(p.endsWith('/registered'))assert.equal(route.request().postDataJSON().arn,'arn:test');
   return route.fulfill({json:{filename:'best.pt',size_bytes:1024,classes:[{id:0,name:'helmet'}],metadata:{epoch:20},notice:''}});
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
  await page.getByLabel('Checkpoint file').setInputFiles({name:'model.pt',mimeType:'application/octet-stream',buffer:Buffer.from('fixture')});
  await page.getByRole('button',{name:'Inspect checkpoint'}).click();
  await page.getByRole('cell',{name:'helmet',exact:true}).waitFor();
  await page.getByRole('radio',{name:'Registered model',exact:true}).check();
  assert.equal(await page.getByRole('cell',{name:'helmet',exact:true}).count(),0);
  await page.getByRole('combobox',{name:'Registered model',exact:true}).selectOption('arn:test');
  await page.getByRole('button',{name:'Inspect checkpoint'}).click();
  await page.getByRole('cell',{name:'helmet',exact:true}).waitFor();
  assert.deepEqual(calls,['/track2/pt-view/upload','/track2/pt-view/registered']);
 }finally{await browser?.close();await server?.close();assert.equal(path.dirname(harness),root);await rm(harness,{recursive:true,force:true});}
});
