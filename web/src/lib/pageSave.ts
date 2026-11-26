import {useLayoutEffect,useRef} from 'react';
type Writer={name:string;capture:()=>()=>Promise<unknown>};
const writers=new Map<symbol,Writer>();
let active:Promise<void>|null=null;
export function registerPageSave(name:string,save:()=>Promise<unknown>){
 const id=Symbol(name);writers.set(id,{name,capture:()=>save});
 return()=>{writers.delete(id);};
}
/** Current values are retained without writing on edit or blur. */
export function usePageSave(name:string,save:()=>Promise<unknown>){
 const latest=useRef(save);latest.current=save;
 useLayoutEffect(()=>{
  const id=Symbol(name);writers.set(id,{name,capture:()=>latest.current});
  return()=>{writers.delete(id);};
 },[name]);
}
export function saveEntirePage():Promise<void>{
 if(active)return active;
 // Freeze all callbacks before save notifications can rerender other cards.
 const snapshot=[...writers.values()].map(w=>({name:w.name,save:w.capture()}));
 active=(async()=>{
  for(const writer of snapshot){
   try{await writer.save();}
   catch(e){throw new Error(`${writer.name}: ${e instanceof Error?e.message:String(e)}`);}
  }
 })().finally(()=>{active=null;});
 return active;
}

let namedSave: (()=>Promise<string>)|null=null;
export function useNamedPageSave(save:()=>Promise<string>){
 const latest=useRef(save);latest.current=save;
 useLayoutEffect(()=>{
  const callback=()=>latest.current();namedSave=callback;
  return()=>{if(namedSave===callback)namedSave=null;};
 },[]);
}
export function saveNamedPage():Promise<string>{
 if(!namedSave)return Promise.reject(new Error("Use Save at the top of Data Preparation to save these settings to a named file."));
 return namedSave();
}
