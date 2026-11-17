import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('queue progress distinguishes completed training images, epochs and downloads', async()=>{
  const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
  const harness=path.join(root,`.progress-${process.pid}.local`);
  await mkdir(harness);
  await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
  await writeFile(path.join(harness,'main.tsx'),`import React from 'react';
    import {createRoot} from 'react-dom/client'; import {Track2TrainingRunProgress} from '../src/components/Track2TrainingRunProgress';
    import '../src/index.css'; const root=createRoot(document.getElementById('root')!);
    window.updateProgress=(progress)=>root.render(<div style={{padding:30,maxWidth:900}}><Track2TrainingRunProgress progress={progress}/></div>);`);
  let server,browser;
  try{
    server=await createServer({root,server:{host:'127.0.0.1',port:0}}); await server.listen();
    browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});
    const page=await browser.newPage({viewport:{width:1000,height:450}});
    await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
    await page.waitForFunction(()=>typeof window.updateProgress==='function');
    const progress={status:'Executing',steps:{TrainYolo:'Executing'},epoch_progress:{current:2,total:50,completed:true},image_progress:{current:10700,total:531200,per_epoch:10624}};
    await page.evaluate(p=>window.updateProgress(p),progress);
    await page.getByText('Epochs completed: 2 / 50').waitFor();
    assert.equal(await page.getByRole('progressbar',{name:'Training images processed'}).getAttribute('aria-valuenow'),'10700');
    await page.screenshot({path:path.join(root,'../tmp/training-queue-progress.png')});
    await page.evaluate(p=>window.updateProgress({...p,image_progress:null,epoch_progress:null,download_progress:{current:100,total:500}}),progress);
    await page.getByText('Downloading dataset: 100 / 500 images').waitFor();
    assert.equal(await page.getByRole('progressbar',{name:'Training images processed'}).count(),0);
    await page.evaluate(p=>window.updateProgress({...p,status:'Succeeded',steps:{TrainYolo:'Succeeded'}}),progress);
    await page.getByText('Training finished before the planned epoch limit.').waitFor();
  }finally{
    await browser?.close(); await server?.close();
    assert.equal(path.dirname(harness),root); await rm(harness,{recursive:true,force:true});
  }
});
