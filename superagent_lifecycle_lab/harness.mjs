// Tutorial adapters. These deliberately replace production policy, workers and storage.
import { Engine, MemoryRepositories, runningAssignment } from './vendor/core/index.ts';
export function trigger(id, overrides = {}) {
  return { id, source: 'chat', threads: [{source:'chat',id:'t1'}],
    actor:{source:'chat',id:'alice'}, human:true, at:'2026-11-05T00:00:00Z',
    text:'Fix the typo in README.', channel:{source:'chat',thread:'t1',address:{message:id}}, meta:{}, ...overrides };
}
export function harness({prepareTicket} = {}) {
  const repo = new MemoryRepositories(), requests = [], messages = [], launches = [], deliveries = new Set();
  let version = 'v1';
  const policy = {
    get version(){return version}, paused:false, evalTimeoutMs:60000,
    gate:t => t.human && t.actor.id === 'alice'
      ? {admissible:true,strength:'explicit',basis:{via:'tutorial-allowlist'}}
      : {admissible:false,reason:'not an authorized human'},
    revalidate:()=>version !== 'revoked', admitsInferred:()=>false,
  };
  const engine = new Engine({repo,policy:()=>policy,
    scope:{scope:()=>({listened:true,observed:false})},
    assessor:{request:c=>requests.push(c),release:()=>{}},
    executor:{start:(t,a)=>launches.push([t,a]),deliver:t=>deliveries.add(t.id),stop:()=>{}},
    outbox:{send:m=>{if(!messages.some(x=>x.id===m.id))messages.push(m)},
      status:id=>messages.some(x=>x.id===id)?'pending':undefined,withdraw:()=>{}},
    now:()=>new Date().toISOString(), ...(prepareTicket?{prepareTicket}:{})});
  function settle(){
    for(;;){
      const next=launches.shift();
      if(next){engine.assignmentStarted(next[0].id,next[1].id);continue}
      const [id]=deliveries;if(id===undefined)return;deliveries.delete(id);
      const t=repo.tickets.get(id), a=runningAssignment(t);
      if(!t.ended && a) for(const input of t.undelivered)engine.inputsDelivered(id,a.id,[input]);
    }
  }
  const pump=()=>{engine.evaluateDue();settle()};
  const decide=(candidate,outcome)=>engine.decide(candidate.triggers[0].id,{...outcome,evalId:candidate.evalId});
  function accept(id='1'){
    engine.receive(trigger(id));pump();
    const d=decide(requests.at(-1),{kind:'accept',title:'Fix README typo',scope:'Correct one spelling error in README.',reason:'Explicit request',acknowledgement:'I will fix the typo.'});
    settle();return d.ticketId;
  }
  const read=id=>engine.inputsRead(id,repo.tickets.get(id).inputs.filter(x=>x.kind==='trigger'&&x.readAt===null).map(x=>x.id));
  return {repo,engine,requests,messages,pump,settle,decide,accept,read,setVersion:v=>{version=v}};
}
