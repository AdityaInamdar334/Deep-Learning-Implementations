"""Small, bounded offline note/task agent. Stdlib-only client for a local llama.cpp server."""
import argparse
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DATA = Path(os.environ.get('BOTPSY_DATA', str(Path.home() / '.botpsy_lab.json')))
MAX_STEPS = 4
TOOLS = [
    {'type':'function','function':{'name':'add_note','description':'Save a short note to local storage','parameters':{'type':'object','properties':{'text':{'type':'string'}},'required':['text'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'list_notes','description':'List saved local notes','parameters':{'type':'object','properties':{},'additionalProperties':False}}},
    {'type':'function','function':{'name':'search_notes','description':'Search saved notes by case-insensitive substring','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'add_task','description':'Add a task to a local task list','parameters':{'type':'object','properties':{'title':{'type':'string'}},'required':['title'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'list_tasks','description':'List tasks and identifiers','parameters':{'type':'object','properties':{},'additionalProperties':False}}},
    {'type':'function','function':{'name':'complete_task','description':'Mark a task completed by exact identifier; requires user confirmation','parameters':{'type':'object','properties':{'task_id':{'type':'string'}},'required':['task_id'],'additionalProperties':False}}},
]
SCHEMA = {x['function']['name']: set(x['function']['parameters'].get('required',[])) for x in TOOLS}
ALLOWED = set(SCHEMA)

def read_db(path=DATA):
    if not path.exists():
        return {'notes':[], 'tasks':[]}
    obj=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj,dict) or not isinstance(obj.get('notes'),list) or not isinstance(obj.get('tasks'),list):
        raise ValueError('Invalid data file')
    return obj

def save_db(db,path=DATA):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(db,indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)

def execute(name,args,path=DATA,approve=lambda _:False):
    if name not in ALLOWED or not isinstance(args,dict):
        return {'error':'Tool not allowed or invalid arguments'}
    if set(args) != SCHEMA[name]:
        return {'error':f'Expected exactly these argument names: {sorted(SCHEMA[name])}'}
    if any(not isinstance(v,str) or len(v)>400 for v in args.values()):
        return {'error':'Arguments must be strings of at most 400 characters'}
    if any(not v.strip() for v in args.values()):
        return {'error':'Empty arguments not allowed'}
    db=read_db(path)
    if name=='list_notes': return {'notes':db['notes'][-20:]}
    if name=='list_tasks': return {'tasks':db['tasks'][-20:]}
    if name=='search_notes': return {'notes':[n for n in db['notes'] if args['query'].casefold() in n['text'].casefold()][-20:]}
    if name=='add_note':
        entry={'id':uuid4().hex[:8],'text':args['text'],'created':datetime.now(timezone.utc).isoformat()}
        db['notes'].append(entry);save_db(db,path);return {'saved':entry}
    if name=='add_task':
        entry={'id':uuid4().hex[:8],'title':args['title'],'done':False}
        db['tasks'].append(entry);save_db(db,path);return {'saved':entry}
    if name=='complete_task':
        target=next((t for t in db['tasks'] if t['id']==args['task_id']),None)
        if target is None:return {'error':'Unknown task ID'}
        if not approve(f"Mark '{target['title']}' complete?"):
            return {'error':'Not approved; no changes made'}
        target['done']=True;save_db(db,path);return {'completed':target}

def request_llm(messages, url, model):
    payload={'model':model,'messages':messages,'tools':TOOLS,'tool_choice':'auto',
             'temperature':0.3,'top_p':0.95,'max_tokens':1024,
             'chat_template_kwargs':{'enable_thinking':True}}
    req=urllib.request.Request(url.rstrip('/')+'/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=120) as resp:
        body=json.load(resp)
    return body['choices'][0]['message']

def run_agent(user_input, llm=request_llm, path=DATA, url='http://127.0.0.1:8080/v1', model='local', approve=lambda _:False):
    messages=[{'role':'system','content':'You are a local notes and tasks assistant. Use only listed tools. Do not invent tool names or IDs. Ask for missing information. Do not claim a tool succeeded without its response. Never execute shell or network actions.'},
              {'role':'user','content':user_input}]
    trace=[]
    for _ in range(MAX_STEPS):
        reply=llm(messages,url,model)
        calls=reply.get('tool_calls') or []
        if not calls:
            return {'answer':reply.get('content') or 'No further action requested.', 'trace':trace}
        messages.append({'role':'assistant','content':reply.get('content'), 'tool_calls':calls})
        for call in calls[:2]:
            fn=call.get('function') or {}
            try: args=json.loads(fn.get('arguments','{}')) if isinstance(fn.get('arguments'),str) else fn.get('arguments',{})
            except (json.JSONDecodeError,TypeError):args={}
            result=execute(fn.get('name'),args,path,approve)
            trace.append({'tool':fn.get('name'),'args':args,'result':result})
            messages.append({'role':'tool','tool_call_id':call.get('id','unknown'),'name':fn.get('name'),'content':json.dumps(result)})
    return {'answer':'Stopped at the four-step limit. Check the trace before proceeding.', 'trace':trace}

def demo_llm(messages,url,model):
    # Fixed mock response for reproducible, no-download smoke tests.
    if any(m.get('role')=='tool' for m in messages):
        return {'role':'assistant','content':'The local tool returned a result. Check the trace for confirmation.'}
    return {'role':'assistant','content':None,'tool_calls':[{'id':'mock-1','type':'function','function':{'name':'add_note','arguments':json.dumps({'text':'Read the BotPsy paper'})}}]}

def interactive_approve(prompt):
    return input(prompt+' Type YES to confirm: ').strip()=='YES'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('request',nargs='?',default='Remember to read the BotPsy paper')
    p.add_argument('--demo',action='store_true',help='No inference required; test the tool pipeline')
    p.add_argument('--url',default='http://127.0.0.1:8080/v1')
    p.add_argument('--model',default='local')
    args=p.parse_args()
    out=run_agent(args.request,llm=demo_llm if args.demo else request_llm,url=args.url,model=args.model,approve=interactive_approve)
    print(json.dumps(out,indent=2,ensure_ascii=False))
if __name__=='__main__':main()
