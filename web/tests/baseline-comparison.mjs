import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir, writeFile, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('baseline registry, optional selection, saved-run results and retry', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.join(root, `.baseline-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness, 'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness, 'main.tsx'), `import React from 'react';import {createRoot} from 'react-dom/client';
    import {Track2RunComparisons} from '../src/components/Track2RunComparisons';import '../src/index.css';
    createRoot(document.getElementById('root')!).render(<Track2RunComparisons/>);`);
  let server, browser;
  try {
    server = await createServer({root, server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser = await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport:{width:1280,height:900}});
    const errors = [], requests = [];
    page.on('pageerror', e => errors.push(e.message));
    let registered = false;
    const row = (role, score) => ({id:role,role,label:role, 'mAP50-95':score,mAP50:score,
      precision:score,recall:score,f1:score,fp:2,fn:3,
      delta_pp:Object.fromEntries(['mAP50-95','mAP50','precision','recall','f1'].map(k=>[k,role==='baseline'?0:5])),
      per_class:[{class:'helmet','mAP50-95':score,precision:score,recall:score,fp:2,fn:3}]});
    let status = 'No comparisons', comparison;
    await page.route('**/track2/**', async route => {
      const req = route.request(), url = new URL(req.url());
      let response;
      if (url.pathname.endsWith('/metrics')) response={rows:[{model_package_arn:'source-v1',group_name:'helmet',version:1}]};
      else if (url.pathname.endsWith('/baselines') && req.method()==='POST') {
        assert.deepEqual(req.postDataJSON(), {source_arn:'source-v1',label:'Reference'});
        registered=true; response={arn:'baseline-v1'};
      } else if (url.pathname.endsWith('/baselines')) response={versions:registered?[{arn:'baseline-v1',version:1,label:'Reference'}]:[]};
      else if (url.pathname.endsWith('/r1/run')) {
        requests.push(req.postDataJSON());
        status='Completed'; comparison={status,result:{status,validation_images:20,dataset_version_id:'saved',snapshot_id:'fixed',
          settings:{confidence:0.25,ap_conf:0.001,imgsz:640},ultralytics_version:'test',rows:[row('baseline',0.61),row('trained',0.66)]}};
        response={status};
      } else response={runs:[{id:'r1',label:'Helmet',submitted_at:'2026-11-24T00:00:00Z',status,can_compare:true,comparison}]};
      await route.fulfill({json:response});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByLabel('Training run').selectOption('r1');
    await page.getByRole('status').filter({hasText:'No comparisons'}).waitFor();
    await page.getByText('Compare this completed run',{exact:true}).click();
    await page.getByText(/Baseline Model registry/).click();
    await page.getByLabel('Source model version').selectOption('source-v1');
    await page.getByLabel('Baseline name',{exact:true}).fill('Reference');
    await page.getByRole('button',{name:'Register baseline version'}).click();
    await page.getByLabel('Baseline Model version').waitFor();
    assert.equal(await page.getByLabel('Baseline Model version').inputValue(),'baseline-v1');
    await page.getByLabel('Other comparison models').selectOption('source-v1');
    await page.getByRole('button',{name:'Run comparison',exact:true}).click();
    await page.getByRole('status').filter({hasText:'Completed'}).waitFor();
    assert.deepEqual(requests[0], {selection:{enabled:true,baseline_arn:'baseline-v1',other_arns:['source-v1']}});
    assert.equal(await page.getByText('+5.00 pp',{exact:false}).count(),5);
    await page.getByText('Per-class results',{exact:true}).click();
    assert.equal(await page.getByRole('cell',{name:'helmet',exact:true}).count(),2);
    await page.screenshot({path:path.join(root,'../tmp/baseline-comparison-results.png'),fullPage:true});
    // Reopen a failed saved comparison and retry without changing its pins.
    status='Failed'; comparison={status,error:'Temporary evaluation failure'};
    await page.reload(); await page.getByLabel('Training run').selectOption('r1');
    await page.getByText('Compare this completed run',{exact:true}).click();
    await page.getByRole('button',{name:'Retry saved comparison'}).click();
    await page.getByRole('status').filter({hasText:'Completed'}).waitFor();
    assert.deepEqual(requests[1], {});
    assert.deepEqual(errors,[]);
  } finally {
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root);
    await rm(harness,{recursive:true,force:true});
  }
});
