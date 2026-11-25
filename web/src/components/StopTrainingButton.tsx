import {useState} from 'react';
import {ApiError,stopSelectedTraining} from '../lib/api';
export function StopTrainingButton({runId,label='training run'}:{runId:string;label?:string}){
 const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[error,setError]=useState('');
 async function stop(){setBusy(true);setError('');try{setMessage((await stopSelectedTraining(runId)).message);}catch(e){setError(e instanceof ApiError?e.detail:'Could not stop this run.');}finally{setBusy(false);}}
 return <span className="inline-flex flex-col gap-1"><button type="button" aria-label={`Stop training: ${label}`} disabled={busy||!!message} onClick={()=>void stop()} className="rounded bg-red-600 px-2 py-1 text-xs font-medium text-white disabled:opacity-50">{busy?'Stopping...':'Stop training'}</button>{message&&<span role="status" className="text-xs">{message}</span>}{error&&<span role="alert" className="text-xs text-critical">{error}</span>}</span>;
}
