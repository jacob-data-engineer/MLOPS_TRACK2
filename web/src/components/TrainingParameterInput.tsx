import { useId, useState } from 'react';
import { trainingParameterError } from '../lib/trainingValidation';

export function TrainingParameterInput({parameter,label,value,fallback,onChange,onCommit}: {
  parameter:string; label:string; value:string | number; fallback?:string;
  onChange:(value:string)=>void; onCommit:()=>void;
}) {
  const id = useId();
  const [focused,setFocused] = useState(false);
  const error = trainingParameterError(parameter,value);
  return <span className="relative block">
    <input type="text" inputMode={parameter === 'freeze' ? 'text' : 'decimal'} aria-label={label}
      aria-invalid={!!error} aria-describedby={error ? id : undefined}
      className={`tabular w-full rounded-md border bg-page px-2 py-1.5 text-xs ${error ? 'border-critical ring-1 ring-critical' : 'border-black/15'}`}
      value={value} placeholder={fallback} onFocus={()=>setFocused(true)}
      onChange={e=>onChange(e.target.value)} onBlur={()=>{setFocused(false); if(!error) onCommit();}} />
    {error && <span id={id} role="alert" className={`${focused ? 'absolute left-0 top-full z-30 mt-1 w-64 max-w-[80vw] rounded-md border border-critical bg-white p-3 shadow-lg' : 'mt-1 block'} text-xs text-critical`}>
      <strong>{label}: </strong>{error}
    </span>}
  </span>;
}
