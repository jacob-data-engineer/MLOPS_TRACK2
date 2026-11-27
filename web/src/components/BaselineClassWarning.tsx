import {useEffect,useState} from 'react';
import {inspectPtRegistered,ApiError} from '../lib/api';
type Classes={id:number;name:string}[];
const inspected=new Map<string,Promise<Classes>>();
function classesFor(arn:string){
 let pending=inspected.get(arn);
 if(!pending){pending=inspectPtRegistered(arn).then(r=>r.classes).catch(e=>{inspected.delete(arn);throw e;});inspected.set(arn,pending);}
 return pending;
}
export function classDifferences(expected:Record<string,string>,actual:Classes){
 const names=new Set(Object.values(expected));
 const rows=Object.entries(expected).map(([id,name])=>{
  const found=actual.filter(c=>c.name===name);
  return {name,expected:id,actual:found.map(c=>String(c.id)).join(', ')||'Missing',status:found.length===0?'Missing from model':found.length>1?'Duplicate model class':String(found[0].id)!==id?'Different ID (remapped)':'Matches'};
 });
 for(const c of actual)if(!names.has(c.name))rows.push({name:c.name,expected:'Not in dataset',actual:String(c.id),status:'Extra model class'});
 return rows;
}
export function BaselineClassWarning({arns,expected,onChecked}:{arns:string[];expected?:Record<string,string>;onChecked?:(blocked:boolean)=>void}){
 const [results,setResults]=useState<Record<string,{classes?:Classes;error?:string}>>({});
 const key=JSON.stringify(arns);
 useEffect(()=>{let active=true;setResults({});for(const arn of arns)void classesFor(arn).then(classes=>{if(active)setResults(r=>({...r,[arn]:{classes}}));}).catch(e=>{if(active)setResults(r=>({...r,[arn]:{error:e instanceof ApiError?e.detail:'Could not inspect checkpoint classes.'}}));});return()=>{active=false;};},[key]);
 useEffect(()=>{onChecked?.(!expected||!Object.keys(expected).length||!arns.length||arns.some(arn=>{const r=results[arn];return !r?.classes||classDifferences(expected,r.classes).some(d=>d.status!=='Matches'&&d.status!=='Different ID (remapped)');}));},[key,JSON.stringify(expected),results,onChecked]);
 if(!arns.length)return null;
 if(!expected||!Object.keys(expected).length)return <p className="text-xs">Select training classes to check baseline compatibility.</p>;
 return <div className="space-y-2">{arns.map(arn=>{
  const result=results[arn];const label=arn.split('/').slice(-2).join(' / ');
  if(!result)return <p key={arn} role="status" className="text-xs">Checking checkpoint classes: {label}…</p>;
  if(result.error)return <div key={arn} role="alert" className="border-2 border-amber-500 bg-amber-50 p-3 text-xs">Class compatibility not verified for {label}: {result.error}</div>;
  const rows=classDifferences(expected,result.classes!);const differences=rows.filter(r=>r.status!=='Matches');
  if(!differences.length)return <p key={arn} className="text-xs">Classes match: {label}</p>;
  const blocked=differences.some(r=>r.status!=='Different ID (remapped)');
  return <div key={arn} role="alert" className="border-2 border-amber-500 bg-amber-50 p-3 text-ink">
   <p className="font-semibold">{blocked?'Warning: comparison classes do not match':'Class IDs differ — names can be remapped'}</p>
   <p className="my-1 break-all text-xs">{label}. {blocked?'This model cannot be compared with these validation classes. Choose a model with the same class names.':'The evaluator remaps IDs by class name; this does not prevent comparison.'}</p>
   <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr><th className="p-2">Class name</th><th>Dataset ID</th><th>Model ID</th><th>Difference</th></tr></thead><tbody>{rows.map((r,i)=><tr key={i} className={r.status!=='Matches'?'bg-amber-200 font-semibold':'border-t border-hairline'}><td className="p-2">{r.name}</td><td>{r.expected}</td><td>{r.actual}</td><td>{r.status}</td></tr>)}</tbody></table></div>
  </div>;
 })}</div>;
}
