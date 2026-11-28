import { useId, useState } from 'react';
import { trainingParameterError } from '../lib/trainingValidation';

export function TrainingParameterInput({parameter,label,value,fallback,onChange,onCommit}: {
  parameter:string; label:string; value:string | number; fallback?:string;
  onChange:(value:string)=>void; onCommit:()=>void;
}) {
  const id = useId();
  const [focused,setFocused] = useState(false);
  // Keep intermediate input such as '0.' and '1e-' while the parent stores numbers.
  const [draft,setDraft] = useState(String(value));
  const displayed = focused ? draft : value;
  const error = trainingParameterError(parameter,displayed);
  return <span className="relative block">
    <input type="text" inputMode={parameter === 'freeze' ? 'text' : 'decimal'} aria-label={label}
      aria-invalid={!!error} aria-describedby={error ? id : undefined}
      className={`tabular w-full rounded-md border bg-page px-2 py-1.5 text-xs ${error ? 'border-critical ring-1 ring-critical' : 'border-black/15'}`}
      value={displayed} placeholder={fallback} onFocus={()=>{setDraft(String(value));setFocused(true);}}
      onChange={e=>{setDraft(e.target.value);onChange(e.target.value);}} onBlur={()=>{setFocused(false); if(!error) onCommit();}} />
    {error && <span id={id} role="alert" className={`${focused ? 'absolute left-0 top-full z-30 mt-1 w-64 max-w-[80vw] rounded-md border border-critical bg-white p-3 shadow-lg' : 'mt-1 block'} text-xs text-critical`}>
      <strong>{label}: </strong>{error}
    </span>}
  </span>;
}
