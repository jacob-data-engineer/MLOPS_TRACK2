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
  let server,browser,saved,lastLoaded='';
  const rows=[1,2].map(n=>({model_package_arn:'arn:'+n,group_name:'helmet',registry:'helmet',generation:'11',size:'s',version:String(n),status:'Approved',description:'',metrics:{mAP50:.5,recall:.6,f1:.5,precision:.6,'mAP50-95':.4,classes:['helmet'],dataset:{dataset_version:'dataset',validation_count:100}}}));
  try {
    server=await createServer({root,plugins:[{name:'catalog-test-stub',enforce:'pre',load(id){if(id.replaceAll('\\','/').endsWith('/routes/Track2CatalogPage.tsx'))return 'export function Track2CatalogPage(){return null}';}}],server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();
    await page.route('**/track2/**',route=>{
      const p=new URL(route.request().url()).pathname;
      let json={};
      if(p.endsWith('/metrics'))json={rows};
      else if(p.endsWith('/comparisons'))json={runs:[]};
      else if(p.endsWith('/export/defaults'))json={destination:'s3',bucket:'track2-s3',prefix:'_exports',format:'zip'};
      else if(p.endsWith('/settings/save')){saved=route.request().postDataJSON().settings;lastLoaded='comparison';json={name:'comparison'};}
      else if(p.endsWith('/settings/forget')){lastLoaded='';json={last_loaded:''};}
      else if(p.endsWith('/settings/load')){
        if(!saved)return route.fulfill({status:404,json:{detail:'File was deleted'}});
        lastLoaded='comparison';json={name:'comparison',settings:saved};
      }
      else if(p.endsWith('/settings'))json={last_loaded:lastLoaded,writable:true,files:saved?[{name:'comparison'}]:[]};
      else if(p.endsWith('/detectors'))json={detectors:[]};
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
    await page.getByRole('radio',{name:'Manual',exact:true}).check();
    await page.getByLabel('Actual images and labels').uncheck();
    await page.getByText('Metadata categories (14)',{exact:true}).click();
    await page.getByLabel('Identity',{exact:true}).uncheck();
    await page.getByLabel('Package destination').selectOption('s3');
    await page.getByLabel('Export bucket').fill('preset-bucket');
    await page.getByLabel('Export prefix').fill('helmet/packages');
    await page.getByLabel('Package format').selectOption('folder');
    assert.equal(await page.getByRole('button',{name:'Save package options',exact:true}).count(),0);
    await page.getByPlaceholder('new file name').fill('comparison');
    await page.getByRole('button',{name:'Save',exact:true}).click();
    await page.getByText('Saved comparison with 2 compared model(s)',{exact:true}).waitFor();
    assert.deepEqual(saved.comparison_versions,['helmet/2','helmet/1']);
    assert.ok(!saved.comparison_metrics.includes('precision'));
    assert.equal(saved.package_options.include_images,false);
    assert.equal(saved.package_options.format,'folder');
    assert.ok(!saved.package_options.metadata_groups.includes('Identity'));
    assert.equal(saved.bucket,'preset-bucket');
    await page.getByRole('button',{name:'New',exact:true}).click();
    await page.getByText('Blank settings. Nothing saved until you press Save.',{exact:true}).waitFor();
    assert.equal(await page.locator('#comparison').getByLabel('Precision',{exact:true}).isChecked(),true);
    await page.getByRole('button',{name:'Load',exact:true}).click();
    await page.getByText('Loaded comparison with 2 compared model(s)',{exact:true}).waitFor();
    assert.equal(await page.locator('#comparison').getByLabel('Precision',{exact:true}).isChecked(),false);
    await page.getByRole('radio',{name:'Manual',exact:true}).check();
    assert.equal(await page.locator('[aria-label="Rank and export compared models"] tbody tr').count(),2);
    await page.reload();
    await page.getByText('Automatically loaded comparison with 2 compared model(s)',{exact:true}).waitFor();
    assert.equal(await page.getByLabel('Actual images and labels').isChecked(),false);
    await page.getByText('Metadata categories (13)',{exact:true}).click();
    assert.equal(await page.getByLabel('Identity',{exact:true}).isChecked(),false);
    assert.equal(await page.getByLabel('Export bucket').inputValue(),'preset-bucket');
    assert.equal(await page.getByLabel('Export prefix').inputValue(),'helmet/packages');
    assert.equal(await page.getByLabel('Package format').inputValue(),'folder');
    assert.equal(await page.locator('#comparison').getByLabel('Precision',{exact:true}).isChecked(),false);
    assert.equal(await page.locator('[aria-label="Rank and export compared models"] tbody tr').count(),2);
    saved.comparison_versions.push('deleted-model/99');
    await page.getByRole('button',{name:'Load',exact:true}).click();
    await page.getByRole('alert').filter({hasText:'deleted-model/99'}).waitFor();
    await page.getByRole('button',{name:'New',exact:true}).click();
    await page.getByText('Blank settings. Nothing saved until you press Save.',{exact:true}).waitFor();
    await page.reload();
    await page.getByRole('button',{name:'Select this',exact:true}).waitFor();
    await page.locator('#comparison').getByText('Pick a version above and press', {exact:false}).waitFor();
    lastLoaded='comparison';saved=undefined;
    await page.reload();
    await page.getByText('Could not restore comparison: File was deleted',{exact:true}).waitFor();
    assert.equal(lastLoaded,'');
  } finally {await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
