import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdir,writeFile,rm} from 'node:fs/promises';
import path from 'node:path';
import {createServer} from 'vite';
import {chromium} from 'playwright';

test('experiment queue and review include every variant, live status, comparison and export',async()=>{
 const root=process.cwd(),dir=path.join(root,`.experiment-review-${process.pid}.local`);await mkdir(dir);
 await writeFile(path.join(dir,'index.html'),'<div id="root"></div><script type="module" src="./main.tsx"></script>');
 await writeFile(path.join(dir,'main.tsx'),`import '../src/index.css';import React from 'react';import {createRoot} from 'react-dom/client';import {Track2ExperimentRuns} from '../src/components/Track2ExperimentRuns';createRoot(document.getElementById('root')).render(<><Track2ExperimentRuns queue/><Track2ExperimentRuns/></>);`);
 const base='arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/yolo26s-helmet-detector/';
 const variants=Array.from({length:6},(_,i)=>({index:i,name:'Variant '+(i+1),state:i===0?'Queued':'Pending',training_status:i===0?'Waiting for resources':undefined,settings:{epochs:(i+1)*10,batch:16,hyperparams:{lr0:.01*(i+1)}},run_number:i===0?81:undefined}));
 let stopped=[];
 let run={id:'experiment-a',name:'Helmet experiment',created_at:'2026-11-25T11:00:00Z',state:'Training',baseline_arn:base+'1',dataset_version_id:'shared-data',shared:variants[0].settings,variants},models=[],browser,server;
 try{
 server=await createServer({root,server:{host:'127.0.0.1',port:0}});await server.listen();browser=await chromium.launch({executablePath:process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE});const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/track2/**',route=>{const p=new URL(route.request().url()).pathname;let body={};if(p.endsWith('/train/stop-selected')){stopped.push(route.request().postDataJSON());body={message:'Stop requested for selected run.'};}else if(route.request().method()==='DELETE'){run={...run,state:'Cancelled',variants:run.variants.map(v=>({...v,state:'Cancelled'}))};body=run;}else if(p.endsWith('/training-variants'))body={runs:[run]};else if(p.endsWith('/metrics'))body={rows:models};else if(p.endsWith('/runs-by-model'))body={runs:{}};else if(p.endsWith('/buckets'))body={buckets:[]};return route.fulfill({json:body});});
 await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/${path.basename(dir)}/index.html`);
 const queue=page.getByLabel('Queued experiments'),review=page.getByLabel('Experiment review');
 await queue.getByText('Waiting for resources',{exact:true}).waitFor();assert.equal(await queue.getByRole('table',{name:'Experiment variants'}).getByRole('row').count(),7);
 await queue.getByText('Run 81',{exact:true}).waitFor();await review.getByText('Comparison results will appear after training and baseline evaluation finish.').waitFor();
 assert.equal(await queue.getByRole('progressbar').getAttribute('aria-valuenow'),'0');
 run={...run,variants:run.variants.map((v,i)=>i===0?{...v,training_status:'Executing',run_id:'run-81'}:v)};
 await queue.getByRole('button',{name:'Stop training: Variant 1',exact:true}).waitFor({timeout:12000});
 await queue.getByRole('button',{name:'Stop training: Variant 1',exact:true}).click();
 await queue.getByText('Stop requested for selected run.',{exact:true}).waitFor();
 assert.deepEqual(stopped,[{run_id:'run-81'}]);
 run={...run,state:'Preparing',stage:'Epoch1: decay (100/200)'};
 await queue.getByRole('progressbar',{name:'Dataset preparation progress'}).waitFor({timeout:12000});
 assert.equal(await queue.getByRole('progressbar').getAttribute('aria-valuenow'),'100');
 assert.equal(await queue.getByRole('progressbar').getAttribute('aria-valuemax'),'200');
 const metrics={'mAP50-95':.3,mAP50:.5,precision:.7,recall:.6,f1:.65,fp:12,fn:20,delta_pp:{'mAP50-95':0,mAP50:0,precision:0,recall:0,f1:0},per_class:[]};
 models=Array.from({length:7},(_,i)=>({model_package_arn:base+(i+1),group_name:'yolo26s-helmet-detector',registry:'helmet',generation:'26',size:'s',version:i+1,status:'Approved',metrics:{'mAP50-95':.3}}));
 run={...run,state:'Completed with failures',variants:variants.map((v,i)=>({...v,state:i===5?'Failed':'Succeeded',run_number:81+i,model_package_arn:i===5?undefined:base+(i+2),error:i===5?'Training failed':undefined})),comparison:{status:'Completed',result:{status:'Completed',rows:models.slice(0,6).map((m,i)=>({...metrics,id:m.model_package_arn,role:i===0?'baseline':'other',label:i===0?'Original base':'Registry model '+i}))}}};
 // Polling must update an open page without a reload; failures remain visible.
 await queue.getByText('No active experiments.',{exact:true}).waitFor({timeout:12000});
 await review.getByLabel('Baseline comparison bar charts').waitFor();
 assert.equal(await review.getByLabel('mAP50 comparison',{exact:true}).getByRole('img').count(),6);
 await review.getByText('6 available models · 6 selected for export',{exact:true}).waitFor({timeout:20000});
 assert.equal(await review.getByRole('checkbox',{name:/^Export /}).count(),6);
 await queue.getByText('No active experiments.',{exact:true}).waitFor();
 await queue.getByText('Finished experiments / needs attention',{exact:true}).click();await queue.getByText('Training failed',{exact:true}).waitFor();
 await review.getByText('Compare training parameters',{exact:true}).click();assert.ok(await review.locator('td.bg-amber-100').count()>0);
 assert.equal(await queue.getByRole('progressbar').getAttribute('aria-valuenow'),'6');
 assert.equal(await review.getByRole('progressbar').count(),0);
 run={...run,id:'pending-experiment',name:'Waiting experiment',state:'Pending',variants:variants.map(v=>({...v,state:'Pending',run_number:undefined})),comparison:undefined};
 await queue.getByRole('button',{name:'Remove from queue',exact:true}).waitFor({timeout:12000});
 await queue.getByRole('button',{name:'Remove from queue',exact:true}).click();
 await queue.getByText('No active experiments.',{exact:true}).waitFor();
 assert.equal(run.state,'Cancelled');
 assert.equal(await queue.getByRole('button',{name:'Remove from queue',exact:true}).count(),0);
 assert.deepEqual(errors,[]);
 }finally{await browser?.close();await server?.close();await rm(dir,{recursive:true,force:true});}
});
