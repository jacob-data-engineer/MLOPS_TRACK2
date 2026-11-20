import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';
test('scheduled overview shows all files and independently toggles scheduler categories without writes',async()=>{
 const root=process.cwd(),harness=path.join(root,`.scheduled-overview-${process.pid}.local`);
 await mkdir(harness);
 await writeFile(path.join(harness,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(harness,'main.tsx'),`import React from 'react';import {createRoot} from 'react-dom/client';import {MemoryRouter} from 'react-router-dom';import {Track2ScheduledPage} from '../src/routes/Track2ScheduledPage';import '../src/index.css';createRoot(document.getElementById('root')!).render(<MemoryRouter><Track2ScheduledPage/></MemoryRouter>);`);
 let server,browser;const requests=[];
 try{
 server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();
 browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();
 await page.route('**/track2/**',route=>{
 const req=route.request(),u=new URL(req.url());requests.push({method:req.method(),url:u});let data={};
 if(u.pathname.endsWith('/scheduled')){
 const events=['helmet_AT','test_AT'].map((file,i)=>({file,live:!i,at:'2026-11-21T14:00:00Z',kind:'train',label:'Training job',sources:['Pull']}));
 data={now:'2026-11-20T00:00:00Z',until:'2027-02-20T00:00:00Z',events,available:['helmet_AT','test_AT'],files:events.map(ev=>({name:ev.file,live:ev.live,next:ev,next_pull:null,notes:[],sources:['Pull'],error:''}))};
 }else if(u.pathname.endsWith('/sensor'))data={config:{enabled:true,every_days:1,at:'22:00'},next_at:'2026-11-21T14:00:00Z'};
 else if(u.pathname.endsWith('/schedules'))data={schedules:[{scope:'model_name',model_name:'helmet',enabled:true,time_sgt:'22:00',next_run_at:'2026-11-21T14:00:00Z',last_result:{written_to:'s3://exports/helmet.pt'}}]};
 else if(u.pathname.endsWith('/auto'))data={settings:{enabled:false,group_name:'helmet'}};
 return route.fulfill({json:data});
 });
 await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(harness)}/index.html`);
 await page.getByText('s3://exports/helmet.pt',{exact:true}).waitFor();
 await page.getByText('Calendar',{exact:true}).waitFor();
 assert.equal(await page.getByText('Upcoming training and pull jobs',{exact:true}).count(),0);
 assert.equal(requests.find(r=>r.url.pathname.endsWith('/scheduled')).url.searchParams.get('files'),'*');
 assert.ok(await page.getByText('test_AT',{exact:true}).count()>0);
 await page.getByLabel('SQL pulls',{exact:true}).uncheck();assert.equal(await page.getByText('SQL image and label pull',{exact:true}).count(),0);
 await page.getByLabel('Model exports',{exact:true}).uncheck();assert.equal(await page.getByText('s3://exports/helmet.pt',{exact:true}).count(),0);
 await page.getByLabel('Training and pulls',{exact:true}).uncheck();assert.equal(await page.getByText('Calendar',{exact:true}).count(),0);
 await page.getByLabel('Model exports',{exact:true}).check();await page.getByText('s3://exports/helmet.pt',{exact:true}).waitFor();
 assert.ok(requests.every(r=>r.method==='GET'));
 }finally{await browser?.close();await server?.close();await rm(harness,{recursive:true,force:true});}
});
