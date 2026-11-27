import {CheckCircle2,CircleDashed,XCircle} from 'lucide-react';
export function RunStatusBadge({status}:{status:string}){
 const state=status.toLowerCase();
 const success=['succeeded','completed','complete','evaluated','ready'].includes(state);
 const failed=/fail|stop|cancel|attention|error/.test(state);
 const color=success?'border-green-700 bg-green-700 text-white':failed?'border-red-600 bg-red-600 text-white':'border-yellow-700 bg-yellow-700 text-white';
 const Icon=success?CheckCircle2:failed?XCircle:CircleDashed;
 return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${color}`}><Icon size={13} strokeWidth={2.25}/>{status}</span>;
}
