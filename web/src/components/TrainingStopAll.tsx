import {ConfirmDialog} from './ConfirmDialog';
﻿import {useEffect,useState} from 'react';
import {ApiError,getTrainingControl,stopAllTraining,type TrainingControl} from '../lib/api';
export function TrainingStopAll(){
 const [state,setState]=useState<TrainingControl|null>(null),[confirm,setConfirm]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState('');
 useEffect(()=>{let live=true;const poll=()=>getTrainingControl().then(x=>{if(live)setState(x);}).catch(()=>{});void poll();const timer=setInterval(poll,5000);return()=>{live=false;clearInterval(timer);};},[]);
 async function act(){setBusy(true);try{const x=await stopAllTraining();setState(x);setMessage([x.message,...(x.problems??[])].filter(Boolean).join(' '));setConfirm(false);}catch(e){setMessage(e instanceof ApiError?e.detail:'Control request failed. Retry.');}finally{setBusy(false);}}
 return <div className="flex flex-col items-end gap-1">
  <button type="button" disabled={busy} onClick={()=>setConfirm(true)} className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-50">Stop All</button>
  {state?.halted&&<span className="max-w-xs text-right text-xs">Automation paused while stopping. It re-enables automatically after all jobs stop.</span>}
  <ConfirmDialog open={confirm} title="Stop All?" destructive busy={busy} confirmLabel="Confirm Stop All" cancelLabel="Keep running" onConfirm={()=>void act()} onCancel={()=>setConfirm(false)} body={<p>Cancel dev preparation, queued work, training and comparisons? Preserved data and completed results remain. Automation re-enables automatically after all jobs stop; cancelled work stays cancelled.</p>}/>
  {message&&<p role="status" className="max-w-sm text-right text-xs font-normal">{message}</p>}
 </div>;
}
