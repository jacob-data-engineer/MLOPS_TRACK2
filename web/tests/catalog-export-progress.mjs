import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('ZIP export shows preparation, completion, incomplete and stopped states beside the action', async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.export-progress-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';
    import {createRoot} from 'react-dom/client'; import {Track2CatalogMetadata} from '../src/components/Track2CatalogMetadata';
    import '../src/index.css'; const root=createRoot(document.getElementById('root')!);
    window.renderExports=(exportsList)=>root.render(<Track2CatalogMetadata selection={{kind:'run',id:'run-1',label:'Run 1'}}
      destination={{destination:'local',format:'zip',bucket:'',prefix:''}} exportsList={exportsList}
      onExported={()=>{}} renderDestination={()=>null}/>); window.renderExports([]);`);
  let server,browser;
  try{
    server=await createServer({root,server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage({viewport:{width:1100,height:800}});
    await page.route('**/track2/**',route=>{
      const p=new URL(route.request().url()).pathname;
      const json=p.endsWith('/pull-control') ? {config:{catalog_export:{output_type:'dataset'}}}
        : p.endsWith('/metadata') ? {groups:{},unavailable:[],checkpoint_available:false}
        : {id:'export-1',label:'run-1',state:'building'};
      return route.fulfill({json});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByRole('button',{name:'Prepare ZIP export'}).click();
    await page.getByText('Preparing export files. Progress updates automatically.').waitFor();
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'),null);
    const item={id:'export-1',run_id:'run-1',format:'zip',state:'ready'};
    await page.evaluate(e=>window.renderExports([e]),item);
    await page.getByRole('button',{name:'Download ZIP',exact:true}).waitFor();
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'),'100');
    await page.evaluate(e=>window.renderExports([{...e,state:'partial'}]),item);
    await page.getByRole('button',{name:'Download incomplete ZIP'}).waitFor();
    await page.evaluate(e=>window.renderExports([{...e,state:'cancelled'}]),item);
    await page.getByText('Export stopped.').waitFor();
    assert.equal(await page.getByRole('progressbar').count(),0);
    await page.evaluate(e=>window.renderExports([{...e,state:'building'}]),item);
    await page.screenshot({path:path.join(root,'../tmp/catalog-export-progress.png')});
  }finally{
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root); await rm(harness,{recursive:true,force:true});
  }
});
