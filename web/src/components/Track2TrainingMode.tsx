export type TrainingMode = 'single' | 'experiments';
export function Track2TrainingMode({value,onChange}:{value:TrainingMode;onChange:(value:TrainingMode)=>void}){
 return <fieldset className="mb-4 flex flex-wrap gap-5 border-b border-hairline pb-3">
  <legend className="mb-2 text-sm font-medium">Training mode</legend>
  <label><input type="radio" name="training-mode" checked={value==='single'} onChange={()=>onChange('single')}/> Train a single model</label>
  <label><input type="radio" name="training-mode" checked={value==='experiments'} onChange={()=>onChange('experiments')}/> Train a series of experiments</label>
  <p className="w-full text-xs text-ink-muted">{value==='single'?'Use Training Settings below for one model. Saved variants are inactive.':'Use Save training setups as variants below. Every selected variant starts from the base model chosen here and uses the same dataset.'}</p>
 </fieldset>;
}
