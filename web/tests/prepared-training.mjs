import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('training supports prepared data, checked saved SQL data and exact previous runs', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.join(root, `.prepared-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness, 'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness, 'main.tsx'), `import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import {TrainingForm} from '../src/routes/Track2ModelsPage';
    import '../src/index.css';
    function Harness(){const [selection,setSelection]=useState({epochs:25,imgSize:640,batch:8,seed:42,patience:10,
      hyperparams:{lr0:0.001},datasetFolder:'',evalFolder:'',detectorSlug:'helmet',architecture:'yolo11s',
      basePackageArn:'arn:aws:sagemaker:test:model-package/helmet/2',notes:'repeatable test'});
      return <TrainingForm selection={selection} setSelection={p=>setSelection(s=>({...s,...p}))}
        onSubmitted={()=>document.body.dataset.submitted='yes'} detectorsVersion={0}/>;}
    createRoot(document.getElementById('root')!).render(<Harness/>);`);
  let server, browser;
  try {
    server = await createServer({ root, server: { host: '127.0.0.1', port: 0 } });
    await server.listen();
    browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE });
    const page = await browser.newPage();
    const queued = [], repeats = [], errors = [];
    let rejectCheck = false;
    let job = null;
    page.on('pageerror', e => errors.push(e.message));
    await page.route('**/track2/**', async route => {
      const req = route.request(), url = new URL(req.url());
      let response = {};
      if (url.pathname.endsWith('/pull-control')) response = { config: {
        central_store: { folder: 'site' }, dataset_assemble: {train_sources:['site'],label_classes:{0:'helmet'}},
        card_state:{'start the training run':true} } };
      else if (url.pathname.endsWith('/detectors')) response = {detectors:[{slug:'helmet',label:'Helmet',classes:{0:'helmet'},groups:{s:'helmet'}}]};
      else if (url.pathname.endsWith('/classes')) response = {classes:{0:'helmet'}};
      else if (url.pathname.endsWith('/metrics')) response = {rows:[]};
      else if (url.pathname.endsWith('/queue-after-pull')) response = {settings:{enabled:false,triggers:[]},next:{act:false,reason:'Manual'}};
      else if (url.pathname.endsWith('/retrain')) response = {settings:{enabled:false},next:{}};
      else if (url.pathname.endsWith('/retrain/preparation')) response = {job};
      else if (url.pathname.endsWith('/versions')) response = {versions:[{version_id:'dv-saved',created_at:'2026-11-17',
        train_images:100,val_images:20,classes:['helmet'],central_recipe:{store_uri:'s3://original/store'}}]};
      else if (url.pathname.endsWith('/audit')) response = {audits:[{id:'run-old',run_number:10,detector_slug:'helmet',
        dataset_version_id:'dv-saved',submitted_at:'2026-11-17',status:'Succeeded',generation:'11',size:'s',
        parameters:{epochs:3,img_size:320,batch:2,seed:7},base_model:{source:'s3://original/model.pt'},train_dataset:{}}]};
      else if (url.pathname.endsWith('/dv-saved/check')) {
        assert.deepEqual(req.postDataJSON(),{detector_slug:'helmet'});
        if (rejectCheck) {await route.fulfill({status:400,json:{detail:'Preserved image is missing'}});return;}
        response = {version_id:'dv-saved',store_uri:'s3://original/store',remapped:true};
      }
      else if (url.pathname.endsWith('/run-old/repeat')) {repeats.push(req.postDataJSON());response={entry_id:'repeated'};}
      else if (url.pathname.endsWith('/retrain/run')) {
        queued.push(req.postDataJSON());
        job = {id:'job-test',state:'running',stage:'Preserving images and labels',completed:50,total:100,percent:47,updated_at:'2026-11-17T00:00:00Z'};
        response = {ok:true,job,steps:[{step:'background',ok:true,detail:'saved'}]};
      }
      await route.fulfill({json:response});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByText(/Training uses the already-prepared decayed dataset/).waitFor();
    assert.equal(await page.getByRole('radio').count(), 3);
    assert.equal(await page.getByText('Pick a snapshot', {exact:true}).count(), 0);
    const cleanup = page.getByRole('checkbox', {name: /Delete the decayed dataset after this run/});
    assert.equal(await cleanup.isChecked(), false);
    await cleanup.check();
    await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
    await page.getByRole('progressbar', {name:'Dataset preparation'}).waitFor();
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'),'47');
    await page.reload();
    await page.getByText('50 / 100 image and label pairs',{exact:true}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Preserving the prepared dataset and queueing...'}).isDisabled(),true);
    job = {...job,state:'completed',stage:'Added to training queue',percent:100,completed:100};
    await page.waitForFunction(()=>document.body.dataset.submitted==='yes');
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'),'100');
    assert.equal(queued.length,1);
    assert.deepEqual(queued[0], {submit:'enqueue',prepared_only:true,training:{epochs:25,img_size:640,batch:8,
      seed:42,patience:10,hyperparams:{lr0:0.001},detector_slug:'helmet',architecture:'yolo11s',notes:'repeatable test',
      base_version:'arn:aws:sagemaker:test:model-package/helmet/2',delete_decay_after_run:true}});
    await page.getByRole('radio',{name:'Use a saved dataset',exact:true}).check();
    await page.getByLabel('Saved dataset record').selectOption('dv-saved');
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
    assert.equal(await cleanup.count(),0);
    rejectCheck = true;
    await page.getByRole('button',{name:'Check saved dataset',exact:true}).click();
    await page.getByText('Preserved image is missing',{exact:true}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
    rejectCheck = false;
    await page.getByRole('button',{name:'Check saved dataset',exact:true}).click();
    await page.getByText(/Files verified. Class IDs will be remapped/).waitFor();
    await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
    await page.waitForFunction(()=>document.querySelector('[aria-label="Dataset preparation"]')?.getAttribute('aria-valuenow')==='47');
    assert.equal(queued[1].training.dataset_version_id,'dv-saved');
    assert.equal(queued[1].training.epochs,25);
    assert.equal(queued[1].training.delete_decay_after_run,false);
    job = {...job,state:'completed',percent:100};
    await page.getByRole('button',{name:'Add to Queue',exact:true}).waitFor();
    await page.getByRole('radio',{name:'Repeat a previous run',exact:true}).check();
    await page.getByLabel('Previous training run').selectOption('run-old');
    await page.getByText('3 epochs · 320px · batch 2 · seed 7',{exact:true}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).count(),0);
    await page.getByRole('button',{name:'Queue recorded run again'}).click();
    await page.getByText(/Queued repeated with the recorded dataset/).waitFor();
    assert.equal(repeats.length,1);
    // History shortcut opens the card even if it was collapsed.
    await page.getByRole('button',{name:'Hide',exact:true}).click();
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('track2-use-saved-dataset',{detail:'dv-saved'})));
    await page.getByLabel('Saved dataset record').waitFor();
    assert.equal(await page.getByLabel('Saved dataset record').inputValue(),'dv-saved');
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
    assert.deepEqual(errors,[]);
  } finally {
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root);
    await rm(harness,{recursive:true,force:true});
  }
});
