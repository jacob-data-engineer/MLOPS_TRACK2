export type TrainingMode = 'single' | 'experiments';
export function Track2TrainingMode(){
 return <div className="mb-4 border-b border-hairline pb-3">
  <p className="mb-2 text-sm font-medium">Train a series of experiments</p>
  <p className="text-xs text-ink-muted">Use Save training setups as variants below. Every selected variant starts from the base model chosen here and uses the same dataset.</p>
 </div>;
}
