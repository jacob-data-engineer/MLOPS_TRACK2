import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('training modes hide unused cards, preserve drafts and survive reload/back navigation',async()=>{
 const root=process.cwd(),harness=path.join(root,`.modes-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {BrowserRouter} from 'react-router-dom';import {Track2MultiAutoTrainingPage} from '../src/routes/Track2MultiAutoTrainingPage';import '../src/index.css';createRoot(document.getElementById('root')).render(<BrowserRouter><Track2MultiAutoTrainingPage/></BrowserRouter>);`);
 let server,browser;
 try{
  server=await createServer({root,plugins:[{name:'mode-child-fixtures',enforce:'pre',load(id){
   if(id.replaceAll('\\','/').endsWith('/routes/Track2ModelsPage.tsx'))return `import React from 'react';export function Track2ModelsPage(){return <label>Standard draft<input/></label>}`;
   if(id.replaceAll('\\','/').endsWith('/routes/Track2ExperimentsPage.tsx'))return `import React from 'react';export function Track2ExperimentsPage(){return <label>Experiment draft<input/></label>}`;
  }}],server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
  await page.getByLabel('Standard draft').fill('standard settings');
  assert.equal(await page.getByLabel('Experiment draft').count(),0);
  await page.getByRole('radio',{name:'Experiment: baseline + variants',exact:true}).click();
  await page.getByLabel('Experiment draft').fill('variant settings');
  assert.equal(await page.getByLabel('Standard draft').isVisible(),false);
  assert.equal(new URL(page.url()).searchParams.get('mode'),'experiment');
  await page.getByRole('radio',{name:'Standard training',exact:true}).click();
  await page.getByLabel('Standard draft').waitFor({state:'visible'});
  assert.equal(await page.getByLabel('Standard draft').inputValue(),'standard settings');
  assert.equal(await page.getByLabel('Experiment draft').isVisible(),false);
  await page.goBack();
  await page.getByLabel('Experiment draft').waitFor({state:'visible'});
  assert.equal(await page.getByLabel('Experiment draft').inputValue(),'variant settings');
  await page.reload();
  assert.equal(await page.getByRole('radio',{name:'Experiment: baseline + variants',exact:true}).isChecked(),true);
  assert.equal(await page.getByLabel('Standard draft').count(),0);
  const app=await readFile(path.join(root,'src/App.tsx'),'utf8');
  assert.match(app,/path="\/track2\/experiments" element=\{<Navigate to="\/track2\/multi-auto-training\?mode=experiment" replace/);
  const nav=await readFile(path.join(root,'src/components/Layout.tsx'),'utf8');
  assert.ok(!nav.includes('to: "/track2/experiments"'));
 }finally{await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
