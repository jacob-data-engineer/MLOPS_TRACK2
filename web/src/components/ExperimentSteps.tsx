import type {VariantRunRecord} from '../lib/api';

type Step={id:string;label:string;state:'Waiting'|'In progress'|'Complete'|'Failed'|'Stopped'|'Cancelled';percent?:number;detail?:string};
export function experimentSteps(run:VariantRunRecord):Step[]{
 const stage=run.stage??'';
 const counts=/\((\d+)\s*\/\s*(\d+)\)/.exec(stage);
 const measured=counts&&Number(counts[2])>0?Math.min(100,Math.round(Number(counts[1])/Number(counts[2])*100)):undefined;
 const detail=counts?`${Number(counts[1]).toLocaleString()} / ${Number(counts[2]).toLocaleString()} items`:undefined;
 const steps:Step[]=[];
 const phase=(labels:string[],current:number,prefix:string,allDone=false,active=false,failed=false)=>labels.forEach((label,i)=>{
  const state:Step['state']=allDone||active&&i<current?'Complete':active&&i===current?(failed?'Failed':'In progress'):'Waiting';
  steps.push({id:`${prefix}-${i}`,label,state,percent:state==='Complete'?100:active&&i===current?measured:undefined,detail:active&&i===current?detail:undefined});
 });
 if(run.trigger==='schedule')steps.push({id:'pull',label:'Pull fresh source data',state:run.pull_done?'Complete':/pull/i.test(stage)?'In progress':'Waiting',percent:run.pull_done?100:undefined});
 if(run.data_mode!=='saved'){
  const current=/Saving dataset records/i.test(stage)?3:/Preserving|Reading prepared/i.test(stage)?2:/decay/i.test(stage)?1:0;
  phase(['List source images and labels','Build the combined / decayed dataset','Preserve the selected images and labels','Save the shared dataset in SQL'],current,'shared',!!run.dataset_version_id,['Preparing','Needs attention'].includes(run.state)&&!run.dataset_version_id&&!/pull/i.test(stage),run.state==='Needs attention');
 }
 for(const variant of run.variants){
  const ownsStage=stage.startsWith(variant.name+':');
  const current=/Saving dataset records/i.test(stage)?2:/Rebuilding saved dataset/i.test(stage)?1:0;
  phase([`${variant.name}: check central-store files`,`${variant.name}: rebuild training input files`,`${variant.name}: save run dataset records`],current,`variant-${variant.index}`,!!variant.run_id||variant.state==='Succeeded',!!run.dataset_version_id&&ownsStage&&['Preparing','Failed'].includes(variant.state),variant.state==='Failed'||run.state==='Needs attention');
  const state:Step['state']=variant.state==='Succeeded'?'Complete':variant.state==='Failed'?'Failed':variant.state==='Stopped'?'Stopped':variant.state==='Queued'&&variant.training_status&&!['Queued','Pending'].includes(variant.training_status)?'In progress':'Waiting';
  steps.push({id:`train-${variant.index}`,label:`${variant.name}: train model (${String(variant.settings.epochs??'?')} epochs)`,state,percent:state==='Complete'?100:undefined,detail:variant.state==='Queued'?variant.training_status??'Queued':undefined});
 }
 const comparison=run.comparison?.status;
 steps.push({id:'compare',label:'Compare trained variants with the baseline',state:comparison==='Completed'?'Complete':['Failed','Unavailable'].includes(comparison??'')?'Failed':run.state==='Comparing'?'In progress':'Waiting',percent:comparison==='Completed'?100:undefined});
 if(run.state==='Cancelled')for(const step of steps)if(step.state!=='Complete')step.state='Cancelled';
 return steps;
}
export function ExperimentSteps({run}:{run:VariantRunRecord}){
 const steps=experimentSteps(run);
 return <div aria-label="Experiment steps" className="space-y-2 rounded border border-hairline p-3"><h4 className="text-sm font-semibold">Steps and progress</h4><p className="text-xs text-ink-muted">Percentages apply to each step, not the whole experiment. Steps without measured counts show In progress; earlier completed steps stay at 100%.</p><ol className="space-y-2">{steps.map((step,i)=><li key={step.id} className="grid items-center gap-2 text-xs sm:grid-cols-[minmax(12rem,2fr)_minmax(8rem,1fr)_10rem]"><span>{i+1}. {step.label}{step.detail&&<span className="block text-ink-muted">{step.detail}</span>}</span><div role="progressbar" aria-label={step.label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={step.percent} aria-valuetext={step.percent==null?step.state:`${step.percent}% - ${step.state}`} className="h-2 overflow-hidden rounded bg-black/10"><div className={`h-full ${step.state==='Failed'?'bg-red-600':'bg-accent'} ${step.state==='In progress'&&step.percent==null?'w-1/3 animate-pulse':''}`} style={step.percent==null?undefined:{width:`${step.percent}%`}}/></div><span>{step.state}{step.percent==null?'':` - ${step.percent}%`}</span></li>)}</ol></div>;
}
