import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('training mode gates editors and queues only the selected workflow with a shared base',async()=>{
 const root=process.cwd(),dir=path.join(root,`.training-mode-${process.pid}.local`);await mkdir(dir);
 await writeFile(path.join(dir,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 const arn='arn:aws:sagemaker:test:model-package/helmet/2';
 await writeFile(path.join(dir,'main.tsx'),`import React,{useState} from 'react';import {createRoot} from 'react-dom/client';import {TrainingForm} from '../src/routes/Track2ModelsPage';import {Track2DetectorsPanel} from '../src/components/Track2DetectorsPanel';import {Track2TrainingMode} from '../src/components/Track2TrainingMode';import {Track2BaseModelPicker} from '../src/components/Track2BaseModelPicker';import {usePageSnapshot} from '../src/lib/pageSnapshot';import {PageSaveBar} from '../src/components/PageSaveBar';
 function Harness(){const [mode,setMode]=useState('single');usePageSnapshot('training-mode',{trainingMode:mode},v=>setMode(v.trainingMode));const [selection,setSelection]=useState({epochs:25,imgSize:640,batch:8,seed:42,patience:10,hyperparams:{},datasetFolder:'',evalFolder:'',detectorSlug:'helmet',architecture:'yolo11s',basePackageArn:'${arn}',notes:''});return <><PageSaveBar/><Track2DetectorsPanel trainingDisabled={mode==='experiments'}><Track2BaseModelPicker detectorSlug="helmet" detectorLabel="Helmet" architecture="yolo11s" generation="11" size="s" groupName="helmet" value={selection.basePackageArn} onChange={a=>setSelection(s=>({...s,basePackageArn:a}))} modeControl={<Track2TrainingMode value={mode} onChange={setMode}/>}/></Track2DetectorsPanel><TrainingForm selection={selection} setSelection={p=>setSelection(s=>({...s,...p}))} onSubmitted={()=>{}} detectorsVersion={0} allowVariants trainingMode={mode}/></>;}createRoot(document.getElementById('root')).render(<Harness/>);`);
 let browser,server,config={default_training:{epochs:25,img_size:640,batch:8,seed:42,patience:10},central_store:{folder:'site'},dataset_assemble:{train_sources:['site'],label_classes:{0:'helmet'}},card_state:{'start the training run':true,'model types and model names':true}},launches=[],rejectVariant=true;
 const settings={epochs:10,img_size:640,batch:8,seed:42,patience:10,hyperparams:{}};
 let draft={id:'draft',revision:1,batch_name:'',baseline_arn:arn,data_mode:'prepared',dataset_version_id:'',detector_slug:'helmet',architecture:'yolo11s',variants:[{id:'a',name:'Ten',selected:true,settings},{id:'b',name:'Twenty',selected:true,settings:{...settings,epochs:20}}]};
 try{
  server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/track2/**',async route=>{const req=route.request(),p=new URL(req.url()).pathname;let body={};
   if(p.endsWith('/pull-control'))body={config};
   else if(p.endsWith('/pull-control/config')){config={...config,...req.postDataJSON()};body={config};}
   else if(p.endsWith('/detectors'))body={detectors:[{slug:'helmet',label:'Helmet',classes:{0:'helmet'},groups:{s:'helmet'}}],sizes:['s']};
   else if(p.endsWith('/classes'))body={classes:{0:'helmet'}};
   else if(p.endsWith('/metrics'))body={rows:[{model_package_arn:arn,group_name:'helmet',registry:'helmet',generation:'11',size:'s',version:2,status:'Approved',metrics:{}}]};
   else if(p.endsWith('/runs-by-model'))body={runs:{}};
   else if(p.endsWith('/comparisons/baselines'))body={versions:[]};
   else if(p.endsWith('/retrain/preparation'))body={job:null};
   else if(p.endsWith('/training-variants/drafts')){if(req.method()==='PUT'){draft={...req.postDataJSON(),id:'draft',revision:draft.revision+1};body=draft;}else body={drafts:[draft]};}
   else if(p.endsWith('/training-variants')){if(req.method()==='POST'){if(rejectVariant){await route.fulfill({status:422,json:{detail:'Baseline is not ready. Choose another version.'}});return;}launches.push({kind:'experiments',payload:req.postDataJSON()});body={id:'run-1'};}else body={runs:[]};}
   else if(p.endsWith('/retrain/run')){launches.push({kind:'single',payload:req.postDataJSON()});body={ok:true,steps:[]};}
   else if(p.endsWith('/buckets'))body={buckets:[]};
   await route.fulfill({json:body});
  });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(dir)}/index.html`);
  await page.getByRole('checkbox',{name:'Run Ten',exact:true}).waitFor();
  assert.equal(await page.getByLabel('Epochs',{exact:true}).isEnabled(),true);
  assert.equal(await page.getByLabel('Variant name',{exact:true}).isDisabled(),true);
  assert.equal(await page.locator('input[name="track2-base-package"]').isEnabled(),true);
  await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
  assert.equal(launches[0].kind,'single');
  assert.equal(launches[0].payload.training.base_version,arn);
  await page.getByRole('radio',{name:'Train a series of experiments',exact:true}).check();
  assert.equal(await page.getByLabel('Epochs',{exact:true}).isDisabled(),true);
  assert.equal(await page.getByLabel('Variant name',{exact:true}).isEnabled(),true);
  assert.equal(await page.locator('input[name="track2-base-package"]').isEnabled(),true);
  await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
  await page.getByRole('alert').filter({hasText:'Baseline is not ready. Choose another version.'}).last().waitFor();
  assert.equal(launches.length,1);
  rejectVariant=false;
  await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
  await page.getByRole('status').filter({hasText:'Submitted 2 variants.'}).waitFor();
  assert.match(launches[1].payload.name,/^helmet variants [a-f0-9]{8}$/);
  assert.equal(launches[1].kind,'experiments');assert.equal(launches[1].payload.baseline_arn,arn);assert.equal(launches[1].payload.variants.length,2);
  await page.getByRole('checkbox',{name:'Run Ten',exact:true}).uncheck();await page.getByRole('checkbox',{name:'Run Twenty',exact:true}).uncheck();
  assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
  await page.getByRole('button',{name:'Save',exact:true}).click();await page.getByRole('status').filter({hasText:'All page settings saved.'}).waitFor();
  await page.reload();await page.getByRole('radio',{name:'Train a series of experiments',exact:true}).waitFor();
  await page.waitForFunction(()=>document.querySelector('input[name="training-mode"]:checked')?.parentElement.textContent.includes('experiments'));
  assert.equal(launches.length,2);assert.deepEqual(errors,[]);
 }finally{await browser?.close();await server?.close();await rm(dir,{recursive:true,force:true});}
});
