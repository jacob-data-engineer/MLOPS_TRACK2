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
  if(result.error)return <div key={arn} role="alert" className="border-2 border-red-500 bg-red-50 p-2 text-xs">Class compatibility not verified for {label}: {result.error}</div>;
  const rows=classDifferences(expected,result.classes!);const differences=rows.filter(r=>r.status!=='Matches');
  const blocked=differences.some(r=>r.status!=='Different ID (remapped)');
  return <div key={arn} role={blocked?'alert':'status'} className={`border-2 p-2 text-ink ${blocked?'border-red-500 bg-red-50':'border-green-600 bg-green-50'}`}>
   <p className="text-xs font-semibold">{blocked?'Warning: comparison classes do not match':differences.length?'Classes compatible — IDs will be remapped':'Classes match'}</p>
   <p className="my-1 break-all text-xs">{label}. {blocked?'Choose a model with the same class names.':differences.length?'The evaluator remaps IDs by class name.':'All dataset and model classes match.'}</p>
   <div className="max-h-60 overflow-auto"><table className="w-full table-fixed text-left text-xs"><thead className="sticky top-0 bg-surface"><tr><th className="w-[35%] px-2 py-1">Data Set Class ID</th><th className="w-[35%] px-2 py-1">Model Class ID / Class Name</th><th className="px-2 py-1">Difference</th></tr></thead><tbody>{rows.map((r,i)=>{
    const missing=r.status==='Missing from model'||r.status==='Extra model class';
    return <tr key={i} className={`border-t border-black/10 ${missing?'bg-[#fff0f3]':''}`}><td className="break-words px-2 py-1">{r.expected==='Not in dataset'?'Missing':`${r.expected} / ${r.name}`}</td><td className="break-words px-2 py-1">{r.actual==='Missing'?'Missing':`${r.actual} / ${r.name}`}</td><td className="break-words px-2 py-1">{r.status}</td></tr>;
   })}</tbody></table></div>
  </div>;
 })}</div>;
}
