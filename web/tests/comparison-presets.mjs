import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('page Save and Load restore comparison models and metric checkboxes', async()=>{
  const root=process.cwd(), harness=path.join(root,`.comparison-presets-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {Track2MetricsPage} from '../src/routes/Track2MetricsPage';createRoot(document.getElementById('root')).render(<Track2MetricsPage/>);`);
  let server,browser,saved;
  const rows=[1,2].map(n=>({model_package_arn:'arn:'+n,group_name:'helmet',registry:'helmet',generation:'11',size:'s',version:String(n),status:'Approved',description:'',metrics:{mAP50:.5,recall:.6,f1:.5,precision:.6,'mAP50-95':.4,classes:['helmet'],dataset:{dataset_version:'dataset',validation_count:100}}}));
  try {
    server=await createServer({root,plugins:[{name:'catalog-test-stub',enforce:'pre',load(id){if(id.replaceAll('\\','/').endsWith('/routes/Track2CatalogPage.tsx'))return 'export function Track2CatalogPage(){return null}';}}],server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();
    await page.route('**/track2/**',route=>{
      const p=new URL(route.request().url()).pathname;
      let json={};
      if(p.endsWith('/metrics'))json={rows};
      else if(p.endsWith('/settings/save')){saved=route.request().postDataJSON().settings;json={name:'comparison'};}
      else if(p.endsWith('/settings/load'))json={name:'comparison',settings:saved};
      else if(p.endsWith('/settings'))json={writable:true,files:saved?[{name:'comparison'}]:[]};
      else if(p.endsWith('/pull-control'))json={config:{}};
      else if(p.endsWith('/runs-by-model'))json={runs:{}};
      else if(p.endsWith('/audit'))json={audits:[]};
      else if(p.endsWith('/export/groups'))json={groups:[]};
      else if(p.endsWith('/export/auto'))json={settings:{enabled:false}};
      else if(p.endsWith('/buckets'))json={buckets:[]};
      return route.fulfill({json});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByRole('button',{name:'Select this',exact:true}).click();
    await page.getByRole('combobox',{name:'Version',exact:true}).selectOption('1');
    await page.getByRole('button',{name:'Select this',exact:true}).click();
    await page.locator('#comparison').getByLabel('Precision',{exact:true}).uncheck();
    await page.getByPlaceholder('new file name').fill('comparison');
    await page.getByRole('button',{name:'Save',exact:true}).click();
    await page.getByText('Saved comparison with 2 compared model(s)',{exact:true}).waitFor();
    assert.deepEqual(saved.comparison_versions,['helmet/2','helmet/1']);
    assert.ok(!saved.comparison_metrics.includes('precision'));
    await page.getByRole('button',{name:'New',exact:true}).click();
    assert.equal(await page.locator('#comparison').getByLabel('Precision',{exact:true}).isChecked(),true);
    await page.getByRole('button',{name:'Load',exact:true}).click();
    await page.getByText('Loaded comparison with 2 compared model(s)',{exact:true}).waitFor();
    assert.equal(await page.locator('#comparison').getByLabel('Precision',{exact:true}).isChecked(),false);
    await page.getByRole('radio',{name:'Selected models',exact:true}).check();
    assert.equal(await page.locator('[aria-label="Rank and export compared models"] tbody tr').count(),2);
    saved.comparison_versions.push('deleted-model/99');
    await page.getByRole('button',{name:'Load',exact:true}).click();
    await page.getByRole('alert').filter({hasText:'deleted-model/99'}).waitFor();
  } finally {await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
