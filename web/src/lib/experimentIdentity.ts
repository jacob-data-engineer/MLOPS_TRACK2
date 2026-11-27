import type {VariantRunRecord} from './api';
/** Display identity only. Internal run IDs and model ARNs remain stable. */
export function experimentIdentity(experiments:VariantRunRecord[]){
 const byRun=new Map<string,{run_number:number;variant_name:string;experiment_id:string}>();
 for(const experiment of experiments){
  if(experiment.run_number==null)continue;
  for(const variant of experiment.variants)if(variant.run_id)byRun.set(variant.run_id,{run_number:experiment.run_number,variant_name:variant.name,experiment_id:experiment.id});
 }
 return byRun;
}
export function displayRun<T extends {run_number?:number|null}>(record:T,id:string,identities:ReturnType<typeof experimentIdentity>):T & {variant_name?:string;experiment_id?:string}{
 const identity=identities.get(id);return identity?{...record,...identity}:record;
}
