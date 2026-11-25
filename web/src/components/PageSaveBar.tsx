import {useState} from 'react';
import {saveEntirePage} from '../lib/pageSave';
import '../lib/pageSnapshot';
export function PageSaveBar(){
 const [busy,setBusy]=useState(false),[message,setMessage]=useState(''),[error,setError]=useState('');
 return <section className="mb-4 flex flex-wrap items-center gap-3 rounded border border-hairline p-3" aria-label="Page settings">
  <button type="button" disabled={busy} className="rounded bg-accent px-3 py-1.5 text-white" onClick={async()=>{setBusy(true);setMessage('');setError('');try{await saveEntirePage();setMessage('All page settings saved.');}catch(e){setError(e instanceof Error?e.message:'Save failed.');}finally{setBusy(false);}}}>{busy?'Saving…':'Save'}</button>
  <span className="text-xs">Save captures this page’s settings and card layout. Edits are not saved automatically.</span>
  {message&&<span role="status">{message}</span>}{error&&<span role="alert">{error}</span>}
 </section>;
}
