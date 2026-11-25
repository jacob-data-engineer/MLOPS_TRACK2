import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('experiment review, explicit submission, saved recovery, parameter differences and results',async()=>{
 const root=process.cwd(),harness=path.join(root,`.experiment-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {Track2ExperimentsPage} from '../src/routes/Track2ExperimentsPage';import '../src/index.css';createRoot(document.getElementById('root')).render(<Track2ExperimentsPage/>);`);
 const arn='arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/yolo26s-helmet-detector/9';
 let server,browser,records=[],submissions=[];
 const score=(id,role,label,value)=>({id,role,label,'mAP50-95':value,mAP50:value,precision:value,recall:value,f1:value,fp:20,fn:30,delta_pp:{'mAP50-95':0,mAP50:0,precision:0,recall:0,f1:0},per_class:[]});
 try{
  server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage({viewport:{width:1500,height:1000}});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/track2/**',route=>{
   const p=new URL(route.request().url()).pathname;
   let body={};
   if(p==='/track2/models/detectors')body={detectors:[{slug:'helmet',label:'Helmet',groups_by_generation:{'26':{s:'yolo26s-helmet-detector'}}}]};
   else if(p==='/track2/retention/versions')body={versions:[{version_id:'fixed-sql-data',run_number:73,train_images:100,val_images:20}]};
   else if(p==='/track2/pt-view/registered')body={models:[{arn,label:'Helmet · YOLO26s · v9',status:'Completed'}]};
   else if(p==='/track2/experiments'){
    if(route.request().method()==='POST'){
     const request=route.request().postDataJSON();submissions.push(request);
     const record={...request,state:'Completed',created_at:'2026-11-25',variants:request.variants.map((v,i)=>({index:i,name:v.name,state:'Succeeded',run_number:74+i,settings:{...request.shared,...v.overrides},model_package_arn:arn.replace('/9',`/${10+i}`)})),comparison:{status:'Completed',result:{rows:[score(arn,'baseline','Baseline v9',.4),score('variant-a','trained','Variant A',.6)]}}};
     records=[record];body=record;
    }else body={experiments:records};
   }else if(p.includes('runs-by-model'))body={runs:{}};
   else if(p.includes('metrics'))body={rows:[]};
   return route.fulfill({json:body});
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
  await page.getByLabel('Experiment name',{exact:true}).fill('Learning rate experiment');
  await page.getByLabel('Model name',{exact:true}).selectOption('helmet');
  await page.getByLabel('Architecture',{exact:true}).selectOption('yolo26s');
  await page.getByLabel('Baseline and starting checkpoint',{exact:true}).selectOption(arn);
  await page.getByLabel('Saved dataset',{exact:true}).selectOption('fixed-sql-data');
  await page.getByRole('button',{name:'Review experiment',exact:true}).click();
  assert.equal(submissions.length,0);
  await page.getByRole('button',{name:'Run experiment',exact:true}).click();
  await page.getByRole('cell',{name:'Run 74',exact:true}).waitFor();
  assert.equal(submissions.length,1);
  assert.equal(submissions[0].baseline_arn,arn);
  assert.equal(submissions[0].dataset_version_id,'fixed-sql-data');
  assert.equal(submissions[0].variants[0].overrides.hyperparams.lr0,.01);
  assert.equal(submissions[0].variants[1].overrides.hyperparams.lr0,.005);
  await page.getByText('Compare training parameters',{exact:true}).click();
  assert.ok(await page.locator('td.bg-amber-100').count()>0);
  await page.getByLabel('Rank comparison by').selectOption('fp');
  await page.reload();
  await page.getByLabel('Saved experiment',{exact:true}).selectOption(records[0].id);
  await page.getByRole('cell',{name:'Run 75',exact:true}).waitFor();
  assert.equal(submissions.length,1);
  assert.deepEqual(errors,[]);
  await page.getByRole('heading',{name:'Saved experiments',exact:true}).scrollIntoViewIfNeeded();
  await mkdir(path.join(root,'..','tmp'),{recursive:true});
  await page.screenshot({path:path.join(root,'..','tmp','experiments-browser.png')});
 }finally{await browser?.close();await server?.close();assert.equal(path.dirname(harness),root);await rm(harness,{recursive:true,force:true});}
});
