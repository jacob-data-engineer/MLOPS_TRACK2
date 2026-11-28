import {RunStatusBadge} from './RunStatusBadge';
import {ConfirmDialog} from './ConfirmDialog';
import {useEffect,useState} from 'react';
import {ApiError,getTrainingControl,stopAllTraining,type TrainingControl} from '../lib/api';
export function TrainingStopAll({running,loaded}:{running:boolean;loaded:boolean}){
 const [state,setState]=useState<TrainingControl|null>(null),[confirm,setConfirm]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState('');
 useEffect(()=>{let live=true;const poll=()=>getTrainingControl().then(x=>{if(live)setState(x);}).catch(()=>{if(live)setState(null);});void poll();const timer=setInterval(poll,5000);return()=>{live=false;clearInterval(timer);};},[]);
 async function act(){setBusy(true);try{const x=await stopAllTraining();setState(x);setMessage((x.problems??[]).filter(Boolean).join(' '));setConfirm(false);}catch(e){setMessage(e instanceof ApiError?e.detail:'Control request failed. Retry.');}finally{setBusy(false);}}
 return <div className="relative flex items-center gap-2">
  <span role="status"><RunStatusBadge status={busy||state?.halted?'Stopping':!state||!loaded?'Checking':running?'Running':'Ready'}/></span>
  <button type="button" disabled={busy} onClick={()=>setConfirm(true)} className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-50">Stop All</button>
  <ConfirmDialog open={confirm} title="Stop All?" destructive busy={busy} confirmLabel="Confirm Stop All" cancelLabel="Keep running" onConfirm={()=>void act()} onCancel={()=>setConfirm(false)} body={<p>Cancel dev preparation, queued work, training and comparisons? Preserved data and completed results remain.</p>}/>
  {message&&<p role="status" className="absolute right-0 top-full z-10 mt-2 w-80 rounded border bg-surface p-2 text-right text-xs font-normal">{message}</p>}
 </div>;
}
