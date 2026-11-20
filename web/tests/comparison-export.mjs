import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir, writeFile, rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('comparison ranks models and exports exact selections with independent options', async () => {
  const root = process.cwd();
  const harness = path.join(root, `.comparison-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness, 'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness, 'main.tsx'), `import React from 'react';
    import {createRoot} from 'react-dom/client'; import {Track2MetricsExportCard} from '../src/components/Track2MetricsExportCard';
    import '../src/index.css';
    const rows=[1,2].map(n=>({model_package_arn:'arn:'+n,group_name:'helmet',registry:'helmet',generation:'11',size:'s',version:String(n),status:'Approved',metrics:{mAP50:n/10,recall:(3-n)/10,dataset:{dataset_version:'dataset-'+n,validation_count:100}}}));
    function App(){const [settings,setSettings]=React.useState({export_mode:'selected',registry:'helmet',generation:'11',size:'s',group_name:'helmet',criterion:'mAP50',destination:'local',bucket:'',prefix:'',retention:'keep_all'});
      return <Track2MetricsExportCard rows={rows} selectedRows={rows.slice(0,1)} settings={settings} onChange={setSettings} runsByArn={{'arn:1':{run_id:'r1',run_number:1,detector:'helmet'},'arn:2':{run_id:'r2',run_number:2,detector:'test-model'}}}/>;}
    createRoot(document.getElementById('root')!).render(<App/>);`);
  let server, browser;
  const requests = []; let unavailable = false;
  const scheduleRequests=[]; const savedSchedules={}; const autoRequests=[];
  try {
    server = await createServer({root, server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser = await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport:{width:1200,height:1000}});
    await page.route('**/track2/**', route => {
      const req = route.request(); const url = new URL(req.url());
      if(url.pathname.endsWith('/selection/run')) {autoRequests.push(req.postDataJSON());return route.fulfill({json:{export_id:'auto-package',run_id:'auto-run',url:'',filename:'',written_to:''}});}
      if(url.pathname.endsWith('/exports') && url.searchParams.get('run_id')==='auto-run') return route.fulfill({json:{exports:[{id:'auto-package',label:'Run 1',state:'ready',format:'zip',files:3,bytes:100,missing:0}]}});
      if (url.pathname.endsWith('/models/detectors')) return route.fulfill({json:{detectors:[{slug:'helmet',label:'Helmet',classes:{0:'helmet'}},{slug:'test-model',label:'test_model',classes:{0:'a',1:'b',2:'c',3:'d',4:'e',5:'f'}}]}});
      if (url.pathname.endsWith('/pull-control')) return route.fulfill({json:{config:{}}});
      if (url.pathname.endsWith('/export/schedules')) return route.fulfill({json:{schedules:Object.values(savedSchedules)}});
      if (url.pathname.includes('/export/schedules/')) {
        const scope=url.pathname.split('/').at(-1),body=req.postDataJSON();scheduleRequests.push({scope,...body});
        savedSchedules[scope]={...savedSchedules[scope],...body,scope,next_run_at:'2026-12-01T01:00:00Z'};
        return route.fulfill({json:{schedule:savedSchedules[scope]}});
      }
      if(url.pathname.endsWith('/selection/preview'))return route.fulfill({json:{best:{group_name:'helmet',model_package_arn:'arn:1',version:'1',criterion:'mAP50',value:.1},considered:1}});
      if (url.pathname.endsWith('/export/groups')) return route.fulfill({json:{groups:[{registry:'helmet',generation:'11',size:'s',group_name:'helmet',export_stem:'helmet'}]}});
      if (url.pathname.endsWith('/export/auto')) return route.fulfill({json:{settings:{enabled:false}}});
      if (url.pathname.endsWith('/export/preview')) return route.fulfill({json:{best:null,considered:0,reason:'No evaluated models'}});
      if (url.pathname.endsWith('/buckets')) return route.fulfill({json:{buckets:[]}});
      if (url.pathname.endsWith('/metadata')) return route.fulfill({json:{checkpoint_available:!unavailable, unavailable:[], groups:{}}});
      if (url.pathname.endsWith('/dataset-availability')) return route.fulfill({json:{state:'ready', detail:'Dataset files verified.'}});
      if (req.method()==='POST') {
        const body=req.postDataJSON(); requests.push(body);
        return route.fulfill({json:{id:body.id,label:body.id,state:'building'}});
      }
      const id=url.searchParams.get('run_id');
      return route.fulfill({json:{exports:[{id,label:id,run_id:id,state:'ready',files:3,bytes:100,missing:0,format:'zip',destination:'local'}]}});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByLabel('Ranking metric').selectOption('mAP50');
    await page.getByLabel('Filter manual models by name').selectOption('test-model');
    assert.equal(await page.getByLabel('Filter manual models by name').locator('option:checked').innerText(), 'test_model (6 classes)');
    await page.getByLabel('Filter manual models by YOLO type').selectOption('11');
    await page.getByLabel('Filter manual models by size').selectOption('s');
    assert.equal(await page.locator('tbody tr').count(), 1);
    assert.match(await page.locator('tbody tr').innerText(), /Run 2/);
    await page.locator('tbody input').check();
    await page.getByLabel('Filter manual models by name').selectOption('helmet');
    assert.equal(await page.getByLabel('Filter manual models by YOLO type').inputValue(), '');
    assert.equal(await page.getByLabel('Filter manual models by size').inputValue(), '');
    assert.match(await page.locator('tbody tr').innerText(), /Run 1/);
    await page.getByRole('radio', {name: 'Show by runs', exact: true}).check();
    assert.equal(await page.getByLabel('Filter manual models by name').count(), 0);
    assert.equal(await page.getByLabel('Filter manual models by YOLO type').count(), 0);
    assert.equal(await page.getByLabel('Filter manual models by size').count(), 0);
    assert.equal(await page.locator('tbody tr').count(), 2);
    assert.equal(await page.getByLabel('Ranking metric').count(), 0);
    assert.match(await page.locator('tbody tr').first().innerText(), /Run 2/);
    await page.getByLabel('Run number sort order').selectOption('asc');
    assert.match(await page.locator('tbody tr').first().innerText(), /Run 1/);
    assert.deepEqual(await page.getByLabel('Filter manual models by run').locator('option').allTextContents(), ['All runs', 'Run 1', 'Run 2']);
    await page.getByLabel('Run number sort order').selectOption('desc');
    assert.match(await page.locator('tbody tr').first().innerText(), /Run 2/);
    await page.getByLabel('Filter manual models by run').selectOption('r2');
    assert.equal(await page.locator('tbody tr').count(), 1);
    assert.equal(await page.locator('tbody input').isChecked(), true);
    await page.getByRole('radio', {name: 'Show by model name', exact: true}).check();
    assert.equal(await page.getByLabel('Filter manual models by run').count(), 0);
    assert.match(await page.locator('tbody tr').innerText(), /Run 1/);
    await page.getByRole('button', {name: 'Clear filters', exact: true}).click();
    for (const name of ['Run', 'Submitted', 'Status', 'Model name', 'Model type', 'Size', 'Model version', 'Dataset version', 'Images', 'Registered as']) {
      assert.equal(await page.getByRole('columnheader', {name, exact: true}).count(), 1);
    }
    await page.getByLabel('Find manual model').fill('Run 2');
    // Search includes the run number even when the word "Run" is omitted.
    await page.getByLabel('Find manual model').fill('2');
    assert.equal(await page.locator('tbody tr').count(),1);
    await page.getByLabel('Find manual model').fill('');
    assert.match(await page.locator('tbody tr').first().innerText(), /Run 2/);
    await page.getByLabel('Ranking metric').selectOption('recall');
    assert.match(await page.locator('tbody tr').first().innerText(), /Run 1/);
    const checks=page.locator('tbody input'); await checks.nth(0).check(); await checks.nth(1).check();
    await page.getByLabel('Actual images and labels').uncheck();
    await page.getByLabel('Metadata', {exact:true}).uncheck();
    await page.getByRole('button',{name:'Review selected packages'}).click();
    await page.getByRole('button',{name:'Export 2 package(s)',exact:true}).click();
    await page.getByRole('button',{name:'Download ZIP',exact:true}).nth(1).waitFor();
    assert.deepEqual(requests.map(r=>r.model_package_arn), ['arn:1','arn:2']);
    assert.ok(requests.every(r=>r.include_model && !r.include_images && !r.include_metadata));
    unavailable=true;
    await page.getByRole('button',{name:'Review selected packages'}).click();
    await page.getByRole('alert').filter({hasText:'no published checkpoint'}).waitFor();
    assert.equal(await page.getByRole('button',{name:'Export 2 package(s)',exact:true}).count(),0);
    await page.getByRole('radio',{name:'Auto',exact:true}).check();
    await page.getByRole('radio',{name:'Best of compared models',exact:true}).waitFor();
    assert.equal(await page.getByLabel('Ranking metric').count(),0);
    await page.getByRole('button',{name:'Preview best model',exact:true}).click();
    await page.getByRole('button',{name:'Export this winner now',exact:true}).waitFor();
    await page.getByRole('button',{name:'Export this winner now',exact:true}).click();
    await page.getByRole('button',{name:'Download package ZIP',exact:true}).waitFor();
    assert.equal(autoRequests[0].package_options.include_images,true);
    assert.equal(autoRequests[0].package_options.metadata_groups.length,14);
    await page.getByLabel('Actual images and labels').uncheck();
    await page.getByLabel('Auto S3 bucket').fill('test-exports');
    await page.getByRole('button',{name:'Save and enable schedule',exact:true}).click();
    await page.getByRole('button',{name:'Stop comparison schedule',exact:true}).waitFor();
    assert.deepEqual(scheduleRequests[0].model_package_arns,['arn:1']);
    assert.equal(scheduleRequests[0].package_options.include_images,false);
    assert.equal(scheduleRequests[0].package_options.include_model,true);
    assert.equal(scheduleRequests[0].package_options.include_metadata,true);
    await page.getByRole('radio',{name:'Best for a Model Name',exact:true}).check();
    await page.getByLabel('Auto Model Name').selectOption('helmet');
    await page.getByLabel('Auto export destination').selectOption('s3');
    await page.getByLabel('Auto package format').selectOption('folder');
    await page.getByRole('button',{name:'Save and enable schedule',exact:true}).click();
    await page.getByRole('button',{name:'Stop Model Name schedule',exact:true}).waitFor();
    assert.equal(scheduleRequests[1].model_name,'helmet');
    assert.deepEqual(scheduleRequests[1].model_package_arns,[]);
    assert.equal(scheduleRequests[1].package_options.format,'folder');
    assert.equal(savedSchedules.comparison.enabled,true);
    assert.equal(savedSchedules.model_name.enabled,true);
    await page.getByRole('button',{name:'Stop comparison schedule',exact:true}).click();
    await page.getByText('Best of compared models: Off',{exact:true}).waitFor();
    assert.equal(savedSchedules.model_name.enabled,true);
    await page.getByRole('radio',{name:'Manual',exact:true}).check();
    await page.getByLabel('Ranking metric').waitFor();
    assert.equal(await page.getByRole('radio',{name:'Best of compared models',exact:true}).count(),0);
  } finally {
    await browser?.close(); await server?.close(); await rm(harness,{recursive:true,force:true});
  }
});
