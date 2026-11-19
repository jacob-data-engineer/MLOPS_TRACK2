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
    import {createRoot} from 'react-dom/client'; import {Track2ComparisonExport} from '../src/components/Track2ComparisonExport';
    import '../src/index.css';
    const rows=[1,2].map(n=>({model_package_arn:'arn:'+n,group_name:'helmet',registry:'helmet',generation:'11',size:'s',version:String(n),status:'Approved',metrics:{mAP50:n/10,recall:(3-n)/10,dataset:{dataset_version:'dataset-'+n,validation_count:100}}}));
    createRoot(document.getElementById('root')!).render(<Track2ComparisonExport rows={rows} runs={{'arn:1':{run_id:'r1',run_number:1},'arn:2':{run_id:'r2',run_number:2}}}/>);`);
  let server, browser;
  const requests = []; let unavailable = false;
  try {
    server = await createServer({root, server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser = await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page = await browser.newPage({viewport:{width:1200,height:1000}});
    await page.route('**/track2/**', route => {
      const req = route.request(); const url = new URL(req.url());
      if (url.pathname.endsWith('/pull-control')) return route.fulfill({json:{config:{}}});
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
  } finally {
    await browser?.close(); await server?.close(); await rm(harness,{recursive:true,force:true});
  }
});
