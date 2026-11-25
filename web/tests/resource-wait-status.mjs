import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('status bar displays resource wait and clears it when resources are available', async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.resource-wait-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react'; import {createRoot} from 'react-dom/client';
    import {Track2ChainStatusBar} from '../src/components/Track2ChainStatusBar';
    createRoot(document.getElementById('root')!).render(<Track2ChainStatusBar/>);`);
  let server,browser;
  let status={running:false,stages:[],pull:null,gpu:{known:true,busy:false,jobs:[]},cpu:{known:true,busy:false,jobs:[]},resources:{waiting:true,detail:'SageMaker quota reached'}};
  try{
    server=await createServer({root,server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage();
    await page.route('**/track2/retention/chain-status', route=>route.fulfill({json:status}));
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.getByText('Waiting for resources',{exact:true}).waitFor();
    assert.equal(await page.getByRole('status').getAttribute('title'),'SageMaker quota reached');
    status={...status,resources:{waiting:false,detail:''}};
    await page.getByText('Waiting for resources',{exact:true}).waitFor({state:'detached'});
    await page.getByText('GPU free',{exact:true}).waitFor();
  }finally{
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root); await rm(harness,{recursive:true,force:true});
  }
});
