import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('history shows dataset identity and keeps CSV, checkpoint download and S3 export working', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.join(root, `.history-${process.pid}.local`);
  await mkdir(harness);
  const dataset = {folder:'_datasets/dv-test',s3_uri:'s3://bucket/_datasets/dv-test',dataset_version_id:'dv-test',
    file_count:1,file_names:['s3://bucket/images/train/a.jpg'],truncated:false};
  const audit = {id:'audit-test',run_number:17517,status:'Succeeded',detector_slug:'helmet',generation:'custom',size:'yolo11s',
    submitted_at:'2026-11-17T00:00:00Z',started_at:null,ended_at:null,source:'queue',build_id:'',notes:'',
    parameters:{epochs:1,img_size:640,batch:2,seed:42},train_dataset:dataset,eval_dataset:dataset,
    eval_is_train_dataset:false,base_model:{folder:'s3://bucket/base.pt',source:'file'},actual:null,metrics:null,
    model:{group:'helmet',version:1,approval_status:'Approved',model_package_arn:'arn:test:package/helmet/1'}};
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';
    import {AuditRow} from '../src/components/Track2TrainingAuditPanel';import '../src/index.css';
    createRoot(document.getElementById('root')!).render(<AuditRow audit={${JSON.stringify(audit)}} onDeleted={()=>{}}/>);`);
  let server,browser;
  try {
    server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage();const exports=[];const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/track2/**',async route=>{
      const req=route.request(),url=new URL(req.url());let response={};
      if(url.pathname.endsWith('/pull-control')) response={config:{card_state:{training_audit:true}}};
      else if(url.pathname.endsWith('/export.csv')) return route.fulfill({contentType:'text/csv',body:'Run ID,audit-test\nDataset record ID,dv-test\n'});
      else if(url.pathname.endsWith('/weights-url')) response={url:`${url.origin}/test-model.pt`};
      else if(url.pathname==='/track2/export' && req.method()==='GET') response={folders:['exports'],registries:[]};
      else if(url.pathname==='/track2/export') {exports.push(req.postDataJSON());response={message:'Exported to s3://bucket/exports/selected.pt',new_key:'exports/selected.pt'};}
      await route.fulfill({json:response});
    });
    await page.route('**/test-model.pt',route=>route.fulfill({headers:{'Content-Disposition':'attachment; filename="model.pt"'},contentType:'application/octet-stream',body:'test-checkpoint'}));
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByText('Dataset record ID:',{exact:true}).waitFor();
    assert.equal(await page.getByText('dv-test',{exact:true}).count(),1);
    assert.equal(await page.getByText('Approval:',{exact:true}).count(),0);
    const csvPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Export training record as CSV',exact:true}).click();
    assert.match((await csvPromise).suggestedFilename(),/audit-test\.csv$/);
    const ptPromise=page.waitForEvent('download');await page.getByRole('button',{name:'Download .pt',exact:true}).click();
    assert.equal((await ptPromise).suggestedFilename(),'model.pt');
    await page.getByRole('button',{name:'Export .pt to S3',exact:true}).click();
    const row=page.getByRole('button',{name:'exports',exact:true}).locator('..');await row.getByRole('checkbox').check();
    await page.getByRole('button',{name:'Export here',exact:true}).click();
    await page.getByText('Exported to s3://bucket/exports/selected.pt',{exact:true}).waitFor();
    assert.deepEqual(exports,[{model_package_arn:'arn:test:package/helmet/1',destination_folder:'exports'}]);
    assert.deepEqual(errors,[]);
  } finally {await browser?.close();await server?.close();assert.equal(path.dirname(harness),root);await rm(harness,{recursive:true,force:true});}
});
