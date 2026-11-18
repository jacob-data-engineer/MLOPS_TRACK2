import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('recording progress follows committed counts and disappears after submission',async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.recording-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React,{useState} from 'react';import {createRoot} from 'react-dom/client';
    import {Track2RecordingProgress} from '../src/components/Track2RecordingProgress';import '../src/index.css';
    function App(){const [progress,setProgress]=useState(null);window.setProgress=setProgress;return <Track2RecordingProgress progress={progress}/>;}
    createRoot(document.getElementById('root')).render(<App/>);`);
  let server,browser;
  try {
    server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage();
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.waitForFunction(()=>!!window.setProgress);
    await page.evaluate(()=>window.setProgress({entry_id:'entry',stage:'Checking preserved dataset',completed:0,total:0}));
    await page.getByText('Preparing the training submission…').waitFor();
    await page.evaluate(()=>window.setProgress({entry_id:'entry',stage:'Recording files',completed:500,total:24824}));
    await page.getByText('Recording files: 500 / 24,824',{exact:true}).waitFor();
    assert.equal(await page.getByRole('progressbar').getAttribute('aria-valuenow'),'500');
    await page.evaluate(()=>window.setProgress({entry_id:'entry',stage:'File records saved — submitting training',completed:24824,total:24824}));
    await page.getByText('File records saved — submitting training: 24,824 / 24,824',{exact:true}).waitFor();
    await page.evaluate(()=>window.setProgress(null));
    await page.getByRole('status').waitFor({state:'detached'});
  } finally {await browser?.close();await server?.close();assert.equal(path.dirname(harness),root);await rm(harness,{recursive:true,force:true});}
});
