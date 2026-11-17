import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('one training schedule saves and reloads interval, Singapore time and optional start date', async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.schedule-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';
    import {createRoot} from 'react-dom/client'; import {Track2PinnedBar} from '../src/components/Track2PinnedBar';
    import '../src/index.css'; createRoot(document.getElementById('root')!).render(<Track2PinnedBar scope="at"/>);`);
  let server,browser;
  try{
    server=await createServer({root,server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage({viewport:{width:1250,height:600}});
    let saved={trigger:'off',every_days:1,at:'02:00',from_date:''}; const posts=[],errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    await page.route('**/track2/**',async route=>{
      const req=route.request(),p=new URL(req.url()).pathname;
      let response;
      if(p.endsWith('/retrain')){
        if(req.method()==='POST'){posts.push(req.postDataJSON());saved={...saved,...req.postDataJSON()};}
        response={trigger:saved.trigger,settings:saved,next_runs:['2026-11-20T06:50:00+00:00']};
      }else if(p.endsWith('/pull-control/settings'))response={files:[]};
      else if(p.endsWith('/pull-control'))response={config:{}};
      else return route.fulfill({status:503,json:{detail:'Fixture: unrelated card unavailable'}});
      await route.fulfill({json:response});
    });
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByRole('button',{name:'On a schedule',exact:true}).click();
    await page.getByLabel('Training interval in days').fill('3');
    await page.getByLabel('Training time', {exact:true}).fill('14:50');
    await page.getByLabel('Training start date').fill('2026-11-20');
    await page.getByRole('button',{name:'Apply schedule',exact:true}).click();
    await page.getByText('Training schedule saved. Pulls use this schedule too.').waitFor();
    assert.deepEqual(posts,[{trigger:'clock',every_days:3,at:'14:50',from_date:'2026-11-20'}]);
    await page.reload();
    await page.getByLabel('Training start date').waitFor();
    assert.equal(await page.getByLabel('Training start date').inputValue(),'2026-11-20');
    assert.equal(await page.getByLabel('Training time',{exact:true}).getAttribute('type'),'time');
    assert.equal(await page.getByText('PULL SCHEDULE',{exact:true}).count(),0);
    assert.equal(await page.getByRole('button',{name:'After each data pull',exact:true}).count(),0);
    await page.screenshot({path:path.join(root,'../tmp/unified-training-schedule.png')});
    assert.deepEqual(errors,[]);
  }finally{
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root); await rm(harness,{recursive:true,force:true});
  }
});
