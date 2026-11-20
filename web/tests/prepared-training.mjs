import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('training supports prepared data and checked saved SQL data', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.join(root, `.prepared-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness, 'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness, 'main.tsx'), `import React, {useState} from 'react';
    import {createRoot} from 'react-dom/client';
    import {TrainingForm} from '../src/routes/Track2ModelsPage';
    import {Track2TrainingAdvanced,effectiveHyperparams} from '../src/components/Track2TrainingAdvanced';
    import {TrainingParameterInput} from '../src/components/TrainingParameterInput';
    import '../src/index.css';
    function Harness(){const [selection,setSelection]=useState({epochs:25,imgSize:640,batch:8,seed:42,patience:10,
      hyperparams:{lr0:0.001},datasetFolder:'',evalFolder:'',detectorSlug:'helmet',architecture:'yolo11s',
      basePackageArn:'arn:aws:sagemaker:test:model-package/helmet/2',notes:'repeatable test'});
      const [extra,setExtra]=useState({});
      return <><TrainingForm selection={selection} setSelection={p=>setSelection(s=>({...s,...p}))}
        onSubmitted={()=>document.body.dataset.submitted='yes'} detectorsVersion={0}/>
        <TrainingParameterInput parameter="epochs" label="Test epochs" value={selection.epochs} onChange={raw=>setSelection(s=>({...s,epochs:Number(raw)}))} onCommit={()=>{}}/>
        <Track2TrainingAdvanced value={extra} enabled={true} onEnabledChange={()=>{}} onCommit={()=>{}}
          onChange={next=>{setExtra(next);setSelection(s=>({...s,hyperparams:effectiveHyperparams(next,true)}));}}/></>;}
    createRoot(document.getElementById('root')!).render(<Harness/>);`);
  let server, browser;
  try {
    server = await createServer({ root, server: { host: '127.0.0.1', port: 0 } });
    await server.listen();
    browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE });
    const page = await browser.newPage({timezoneId:'America/Los_Angeles'});
    const queued = [], errors = [];
    let rejectCheck = false;
    let releaseDatasets;
    const datasetsGate = new Promise(resolve => {releaseDatasets = resolve;});
    let releaseHistory;
    const historyGate = new Promise(resolve => {releaseHistory = resolve;});
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
      else if (url.pathname.endsWith('/versions')) {await datasetsGate; response = {versions:[{version_id:'dv-saved',run_number:70,created_at:'2026-11-17T18:30:00Z',
        train_images:100,val_images:20,classes:['helmet'],central_recipe:{store_uri:'s3://original/store'}}]};}
      else if (url.pathname.endsWith('/audit')) {await historyGate; response = {audits:[{id:'run-old',run_number:10,detector_slug:'helmet',
        dataset_version_id:'dv-saved',submitted_at:'2026-11-17',status:'Succeeded',generation:'11',size:'s',
        parameters:{epochs:3,img_size:320,batch:2,seed:7},base_model:{source:'s3://original/model.pt'},train_dataset:{}}]};}
      else if (url.pathname.endsWith('/dv-saved/check')) {
        assert.deepEqual(req.postDataJSON(),{detector_slug:'helmet'});
        if (rejectCheck) {await route.fulfill({status:400,json:{detail:'Preserved image is missing'}});return;}
        response = {version_id:'dv-saved',store_uri:'s3://original/store',remapped:true};
      }
      else if (url.pathname.endsWith('/retrain/run')) {
        queued.push(req.postDataJSON());
        job = {id:'job-test',state:'running',stage:'Preserving images and labels',completed:50,total:100,percent:47,updated_at:'2026-11-17T00:00:00Z'};
        response = {ok:true,job,steps:[{step:'background',ok:true,detail:'saved'}]};
      }
      await route.fulfill({json:response});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByText(/Training uses the already-prepared decayed dataset/).waitFor();
    assert.equal(await page.getByRole('radio').count(), 2);
    assert.equal(await page.getByRole('radio',{name:'Repeat a previous run',exact:true}).count(),0);
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
    const loading = page.getByRole('status').filter({hasText:'Loading datasets…'});
    await loading.waitFor();
    assert.equal(await page.getByLabel('Saved dataset record').isDisabled(),true);
    assert.equal(await loading.evaluate(el => getComputedStyle(el).animationName),'pulse');
    releaseDatasets();
    assert.match(await page.locator('option[value="dv-saved"]').textContent(), /^Run 70/);
    await page.getByLabel('Saved dataset record').selectOption('dv-saved');
    assert.equal(await loading.count(),0);
    // The picker becomes usable before the slower source-run history returns.
    releaseHistory();
    assert.match(await page.locator('option[value="dv-saved"]').textContent(), /18 Nov 2026, 2:30 AM SGT \(GMT\+8\)/);
    const selectStyle = await page.getByLabel('Saved dataset record').evaluate(el => ({border:getComputedStyle(el).borderTopWidth,height:el.getBoundingClientRect().height,background:getComputedStyle(el).backgroundColor}));
    assert.equal(selectStyle.border,'2px');
    assert.ok(selectStyle.height >= 48);
    assert.equal(selectStyle.background,'rgb(255, 255, 255)');
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

    // History shortcut opens the card even if it was collapsed.
    await page.getByRole('button',{name:'Hide',exact:true}).click();
    await page.evaluate(()=>window.dispatchEvent(new CustomEvent('track2-use-saved-dataset',{detail:'dv-saved'})));
    await page.getByLabel('Saved dataset record').waitFor();
    assert.equal(await page.getByLabel('Saved dataset record').inputValue(),'dv-saved');
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
    await page.getByRole('radio',{name:'Use newly prepared data',exact:true}).check();
    await page.getByRole('button',{name:'Show the rest of the training parameters'}).click();
    const mosaic = page.getByRole('textbox',{name:'Mosaic',exact:true});
    await mosaic.fill('5.1');
    assert.equal(await mosaic.getAttribute('aria-invalid'),'true');
    await page.getByRole('alert').filter({hasText:'Mosaic:'}).waitFor();
    assert.match(await page.getByRole('alert').filter({hasText:'Mosaic:'}).textContent(),/Minimum: 0; maximum: 1/);
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),true);
    await mosaic.fill('1');
    assert.equal(await mosaic.getAttribute('aria-invalid'),'false');
    assert.equal(await page.getByRole('button',{name:'Add to Queue',exact:true}).isDisabled(),false);
    for (const [name,bad,good,expected] of [
      ['Dataset fraction','0','1','greater than 0'],['Nominal batch','2.5','64','whole number'],
      ['Rotation','361','-360','maximum: 360'],['Shear','-181','180','Minimum: -180'],
      ['Box','-1','7.5','maximum: no fixed limit'],['Freeze layers','-1','0,1,2','Minimum: 0'],
      ['Test epochs','1001','25','maximum: 1000'],
    ]) {
      const input=page.getByRole('textbox',{name,exact:true});
      await input.fill(bad);
      assert.equal(await input.getAttribute('aria-invalid'),'true',name);
      assert.match(await page.getByRole('alert').filter({hasText:name+':'}).textContent(),new RegExp(expected));
      await input.fill(good);
      assert.equal(await input.getAttribute('aria-invalid'),'false',name);
    }
    assert.deepEqual(errors,[]);
  } finally {
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root);
    await rm(harness,{recursive:true,force:true});
  }
});
