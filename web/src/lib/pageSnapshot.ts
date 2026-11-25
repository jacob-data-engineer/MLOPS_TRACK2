import {captureCardLayout,restoreCardLayout} from "./cardState";
import {useEffect, useRef} from 'react';
import {getTrack2PullControl, saveTrack2PullConfig} from './api';
import {registerPageSave} from './pageSave';

type Entry={read:()=>unknown; restore:(value:any)=>void; ready:boolean};
const entries=new Map<string,Entry>();
let saved:Record<string,unknown>={};
const drafts:Record<string,unknown>={};
let loading:Promise<void>|null=null;
function load(){
 if(!loading)loading=getTrack2PullControl().then(result=>{saved=result.config.page_state??{};}).catch(error=>{loading=null;throw error;});
 return loading;
}
/** Only serializable form choices. Never register jobs, progress or uploaded File objects. */
export function usePageSnapshot<T>(name:string,value:T,restore:(value:T)=>void,enabled=true){
 const key=`${window.location.pathname}/${name}`;
 const latest=useRef({value,restore});latest.current={value,restore};
 useEffect(()=>{
  if(!enabled)return;
  let alive=true;
  const initial=JSON.stringify(latest.current.value);
  const entry:Entry={read:()=>latest.current.value,restore:v=>latest.current.restore(v),ready:false};
  entries.set(key,entry);
  void load().then(()=>{
   if(!alive)return;
   const source={...saved,...drafts};
   if(key in source&&JSON.stringify(latest.current.value)===initial)entry.restore(source[key]);
   entry.ready=true;
  }).catch(()=>{ /* Save retries the read and reports failures. */ });
  return()=>{alive=false;if(entries.get(key)===entry){drafts[key]=entry.read();entries.delete(key);}};
 },[key,enabled]);
}
export function capturePageSnapshot(){
 const prefix=window.location.pathname+'/';
 return {...Object.fromEntries(Object.entries({...saved,...drafts}).filter(([key])=>key.startsWith(prefix))),...Object.fromEntries([...entries].map(([key,entry])=>[key,entry.read()])),[`${prefix}card-layout`]:captureCardLayout()};
}
export function restorePageSnapshot(snapshot:Record<string,unknown>={}){
 const layout=snapshot[`${window.location.pathname}/card-layout`];
 if(layout&&typeof layout==="object")restoreCardLayout(layout as Record<string,boolean>);
 saved={...saved,...snapshot};Object.assign(drafts,snapshot);
 for(const [key,entry] of entries)if(key in snapshot)entry.restore(snapshot[key]);
}
registerPageSave('Page selections',async()=>{
 await load();
 // Read the latest server map so saving this page preserves other pages.
 const live=await getTrack2PullControl();
 const snapshot=capturePageSnapshot();
 await saveTrack2PullConfig({page_state:{...live.config.page_state,...snapshot}});
 saved={...saved,...snapshot};
});
