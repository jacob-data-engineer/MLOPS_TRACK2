import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir, writeFile, rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('completion panels open results and S3 link without downloads or navigation', async () => {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const harness = path.join(root, `.completion-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'), '<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'), `import React from 'react';import {createRoot} from 'react-dom/client';
    import {Track2CompletionNotices} from '../src/components/Track2CompletionNotices';import '../src/index.css';
    createRoot(document.getElementById('root')!).render(<><input aria-label="Unsaved work"/><Track2CompletionNotices/></>);`);
  let server, browser;
  try {
    server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage({viewport:{width:1280,height:900}});
    await page.clock.install();
    const errors=[];page.on('pageerror',e=>errors.push(e.message));
    let done=false, downloads=0, polls=0;
    page.on('download',()=>downloads++);
    await page.route('**/track2/**', async route=>{
      const exports=new URL(route.request().url()).pathname.endsWith('/exports');
      if(!exports)polls++;
      await route.fulfill({json:exports ? {exports:[{id:'e1',label:'Run 72',state:done?'ready':'building',destination_uri:'s3://track2-s3/export/run-72.zip'}]} : {
        runs:[{id:'r1',label:'Test model',training_status:done?'Succeeded':'Executing',status:done?'Completed':'Pending',
          can_compare:false,comparison:done?{result:{status:'Completed',validation_images:1788,dataset_version_id:'saved',snapshot_id:'fixed',rows:[]}}:undefined}]}});
    });
    const url=`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`;
    await page.goto(url);
    await page.getByLabel('Unsaved work').fill('keep this');
    await page.waitForFunction(()=>sessionStorage.getItem('track2-completion-observer-v1'));
    assert.equal(await page.getByText('Training Succeeded:',{exact:false}).count(),0);
    done=true;
    await page.clock.runFor(11000);
    await page.getByText('Training Succeeded: Test model').waitFor();
    await page.getByText('1788 validation images',{exact:false}).waitFor();
    assert.match(await page.getByRole('link',{name:'Open export in S3'}).getAttribute('href'), /prefix=export%2Frun-72.zip/);
    assert.equal(await page.getByLabel('Unsaved work').inputValue(),'keep this');
    assert.equal(page.url(),url);assert.equal(downloads,0);
    await page.clock.runFor(11000);
    assert.equal(await page.getByText('Training Succeeded: Test model').count(),1);
    await page.screenshot({path:path.resolve(root,'../tmp/completion-notices.png')});
    await page.reload();await page.clock.runFor(11000);
    assert.equal(await page.getByText('Training Succeeded: Test model').count(),0);
    assert.ok(polls>=3);assert.deepEqual(errors,[]);
  } finally {await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
