import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, writeFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { chromium } from 'playwright';

test('training queues the prepared dataset with the selected parameters and no source selector', async () => {
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
    const queued = [], errors = [];
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
      else if (url.pathname.endsWith('/retrain/run')) {
        queued.push(req.postDataJSON()); response = {ok:true,steps:[{step:'queue',ok:true,detail:'queued'}]};
      }
      await route.fulfill({json:response});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByText(/already-prepared decayed dataset/).waitFor();
    assert.equal(await page.getByRole('radio').count(), 0);
    assert.equal(await page.getByText('Pick a snapshot', {exact:true}).count(), 0);
    const cleanup = page.getByRole('checkbox', {name: /Delete the decayed dataset after this run/});
    assert.equal(await cleanup.isChecked(), false);
    await cleanup.check();
    await page.getByRole('button',{name:'Add to Queue',exact:true}).click();
    await page.waitForFunction(()=>document.body.dataset.submitted==='yes');
    assert.equal(queued.length,1);
    assert.deepEqual(queued[0], {submit:'enqueue',prepared_only:true,training:{epochs:25,img_size:640,batch:8,
      seed:42,patience:10,hyperparams:{lr0:0.001},detector_slug:'helmet',architecture:'yolo11s',notes:'repeatable test',
      base_version:'arn:aws:sagemaker:test:model-package/helmet/2',delete_decay_after_run:true}});
    assert.deepEqual(errors,[]);
  } finally {
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root);
    await rm(harness,{recursive:true,force:true});
  }
});
