import {useId,useState,type ReactNode} from 'react';
/** Keep controls mounted while collapsed so drafts and background status survive. */
export function SaveSection({children}:{children:ReactNode}){
 const [open,setOpen]=useState(true);const id=useId();
 return <section className="w-full min-w-0"><div className="mb-1 flex items-center justify-between gap-2"><span className="text-xs font-medium">Saved settings</span><button type="button" aria-expanded={open} aria-controls={id} className="rounded border border-hairline px-3 py-1 text-xs" onClick={()=>setOpen(v=>!v)}>{open?'Hide saved settings':'Show saved settings'}</button></div><div id={id} hidden={!open}>{children}</div></section>;
}
