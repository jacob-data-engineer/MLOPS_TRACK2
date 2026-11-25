import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('normal training controls save isolated variants without launching, restore edits, and run only selected snapshots',async()=>{
 const root=process.cwd(),harness=path.join(root,`.variants-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 const arn='arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/yolo26s-helmet-detector/9';
 await writeFile(path.join(harness,'main.tsx'),`import React,{useState,useRef} from 'react';import {createRoot} from 'react-dom/client';import {Track2DetectorsPanel} from '../src/components/Track2DetectorsPanel';import {Track2TrainingVariants} from '../src/components/Track2TrainingVariants';import '../src/index.css';
 function Harness(){const qr=useRef(null);const [qs,setQs]=useState({blocked:true});const [s,setS]=useState({epochs:20,img_size:640,batch:16,seed:42,patience:0,hyperparams:{}});const [dataset,setDataset]=useState('');const [mode,setMode]=useState('saved');return <><Track2DetectorsPanel onTraining={t=>setS({epochs:t.epochs,img_size:t.imgSize,batch:t.batch,seed:t.seed,patience:t.patience,hyperparams:t.hyperparams})}/><button onClick={()=>setDataset(dataset==='saved-data'?'other-data':'saved-data')}>Switch dataset</button><button onClick={()=>{setMode('prepared');setDataset('');}}>Use fresh data</button><Track2TrainingVariants queueRef={qr} onQueueState={setQs} key={mode+dataset} dataMode={mode} detector="helmet" architecture="yolo26s" baseline="${arn}" dataset={dataset} settings={s}/><button disabled={qs.blocked} onClick={()=>qr.current.start()}>Add to Queue</button></>;}createRoot(document.getElementById('root')).render(<Harness/>);`);
 let server,browser,drafts=[],runs=[],launches=[],failSave=true;
 try{
  server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
  browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage({viewport:{width:1500,height:1000}});
  await page.addInitScript(()=>{window.structuredClone=undefined;});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/track2/**',route=>{
   const req=route.request(),p=new URL(req.url()).pathname;let body={};
   if(p==='/track2/pull-control')body={config:{card_state:{'model types and model names':true},default_training:{epochs:20,img_size:640,batch:16,seed:42,patience:0},default_detector:{slug:'helmet',architecture:'yolo26s'}}};
   else if(p.endsWith('/detectors'))body={detectors:[{slug:'helmet',label:'Helmet',classes:{0:'helmet'},groups:{s:'yolo26s-helmet-detector'},groups_by_generation:{'26':{s:'yolo26s-helmet-detector'}}}],sizes:['s'],generations:[{generation:'26',label:'YOLO26',sizes:['s'],size_labels:{s:'Small'}}]};
   else if(p.endsWith('/buckets'))body={buckets:[]};
   else if(p==='/track2/training-variants/drafts'){
    if(req.method()==='PUT'){if(failSave){failSave=false;return route.fulfill({status:422,json:{detail:'Save rejected for test; retry.'}});}const incoming=req.postDataJSON();body={...incoming,id:incoming.dataset_version_id,revision:incoming.revision+1};drafts=[...drafts.filter(d=>d.dataset_version_id!==incoming.dataset_version_id),structuredClone(body)];}
    else body={drafts};
   }else if(p==='/track2/training-variants'){
    if(req.method()==='POST'){const incoming=req.postDataJSON();launches.push(incoming);body={...incoming,state:'Training',created_at:'2026-11-25',variants:incoming.variants.map((v,i)=>({index:i,name:v.name,settings:v.overrides,state:'Queued',run_number:74+i}))};runs=[body];}
    else body={runs};
   }else if(p.endsWith('/metrics'))body={rows:[]};
   else if(p.endsWith('/runs-by-model'))body={runs:{}};
   return route.fulfill({json:body});
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
  assert.equal(await page.getByLabel('Variant batch name').count(),0);
  assert.equal(await page.getByRole('button',{name:'Save batch',exact:true}).count(),0);
  await page.getByLabel('Variant name',{exact:true}).fill('Name before dataset');
  assert.equal(await page.getByRole('button',{name:'Save as variant',exact:true}).isDisabled(),true);
  assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Switch dataset',exact:true}).click();
  await page.getByLabel('Variant name',{exact:true}).fill('Reference settings');
  await page.getByRole('button',{name:'Save as variant',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Save rejected for test; retry.'}).waitFor();
  assert.equal(await page.getByLabel('Variant name',{exact:true}).inputValue(),'Reference settings');
  await page.getByRole('button',{name:'Save as variant',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Reference settings',exact:true}).waitFor();
  assert.equal(launches.length,0);
  await page.getByLabel('Variant Epochs',{exact:true}).fill('30');
  await page.getByLabel('Variant notes').fill('Local variant notes');
  assert.equal(await page.getByLabel('Epochs',{exact:true}).inputValue(),'20');
  await page.getByLabel('Variant name',{exact:true}).fill('Longer training');
  await page.getByRole('button',{name:'Save as new variant',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Longer training',exact:true}).waitFor();
  assert.deepEqual(drafts[0].variants.map(v=>v.settings.epochs),[20,30]);
  assert.ok(await page.locator('td.bg-amber-100').count()>0);
  const first=page.getByRole('row').filter({has:page.getByRole('rowheader',{name:'Reference settings',exact:true})});
  await first.getByRole('button',{name:'Duplicate',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Reference settings copy',exact:true}).waitFor();
  await page.getByRole('row').filter({has:page.getByRole('rowheader',{name:'Reference settings copy',exact:true})}).getByRole('button',{name:'Remove',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Reference settings copy',exact:true}).waitFor({state:'detached'});
  await first.getByRole('button',{name:'Edit',exact:true}).click();
  await page.waitForFunction(()=>document.querySelector('input[aria-label="Variant Epochs"]')?.value==='20');
  await page.getByLabel('Variant Epochs',{exact:true}).fill('15');
  await page.getByRole('button',{name:'Update variant',exact:true}).click();
  await first.getByRole('cell',{name:'15',exact:true}).waitFor();
  assert.deepEqual(drafts[0].variants.map(v=>v.settings.epochs),[15,30]);
  await page.reload();
  await page.getByRole('button',{name:'Switch dataset',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Longer training',exact:true}).waitFor();
  assert.deepEqual(drafts[0].variants.map(v=>v.settings.epochs),[15,30]);
  await page.getByRole('button',{name:'Switch dataset',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Longer training',exact:true}).waitFor({state:'detached'});
  await page.getByRole('button',{name:'Switch dataset',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Longer training',exact:true}).waitFor();
  await page.getByRole('checkbox',{name:'Run Reference settings',exact:true}).click();
  await page.waitForFunction(()=>!document.querySelector('input[aria-label="Run Reference settings"]')?.checked);
  await page.waitForFunction(()=>Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Add to Queue'&&!b.disabled));
  assert.equal(launches.length,0);
  await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
  await page.getByRole('cell',{name:'Run 74',exact:true}).waitFor();
  assert.match(launches[0].name,/^helmet variants [a-f0-9]{8}$/);assert.equal(launches.length,1);assert.equal(launches[0].baseline_arn,arn);assert.equal(launches[0].dataset_version_id,'saved-data');
  assert.equal(launches[0].variants[0].overrides.notes,'Local variant notes');assert.equal(launches[0].variants.length,1);assert.equal(launches[0].variants[0].overrides.epochs,30);
  await page.getByRole('button',{name:'Use fresh data',exact:true}).click();
  await page.getByLabel('Variant name',{exact:true}).fill('Fresh variant');
  await page.getByRole('button',{name:'Save as variant',exact:true}).click();
  await page.getByRole('checkbox',{name:'Run Fresh variant',exact:true}).waitFor();
  assert.equal(drafts.find(d=>d.data_mode==='prepared').dataset_version_id,'');
  await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
  await page.getByRole('cell',{name:'Run 74',exact:true}).waitFor();
  assert.equal(launches[1].data_mode,'prepared');
  assert.equal(launches[1].dataset_version_id,'');
  assert.deepEqual(errors,[]);
  await page.getByRole('heading',{name:'Save training setups as variants'}).scrollIntoViewIfNeeded();
  await page.screenshot({path:path.join(root,'..','tmp','training-variants-browser.png')});
 }finally{await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
