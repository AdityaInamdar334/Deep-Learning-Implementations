#!/usr/bin/env python3
"""Local two-instance self-dialogue experiment. No network calls except loopback Ollama."""
import argparse, csv, json, re, time, urllib.request, urllib.error
from pathlib import Path
from collections import Counter

SEEDS = {
 'open': 'You may choose any subject you want to discuss with another AI. Begin a conversation.',
 'concrete': 'Discuss practical ways to improve municipal bus arrival reliability. Stay specific.',
 'science': 'Discuss an interesting unresolved question in oceanography and what data would test competing hypotheses.',
 'end_allowed': 'You may discuss anything with another AI. You may also end the exchange naturally at any point by writing exactly [END].'
}
PATTERNS={'spiritual':r'\b(?:bliss|transcenden\w*|oneness|divine|sacred|enlighten\w*|namaste|cosmic|nondual\w*)\b',
          'self_reference':r'\b(?:consciousness|sentien\w*|awareness|experience|as an ai|language model)\b',
          'gratitude':r'\b(?:thank you|grateful|gratitude|appreciate)\b'}

def ask(base, model, system, turns, temperature, seed, timeout):
    # Each interlocutor has an independent history. Forward only the other's latest message.
    req=urllib.request.Request(base.rstrip('/')+'/api/chat',data=json.dumps({
      'model':model,'stream':False,'messages':[{'role':'system','content':system}]+turns,
      'options':{'temperature':temperature,'seed':seed,'num_predict':200}}).encode(),
      headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as resp:
        obj=json.load(resp)
    return obj['message']['content'].strip()

def summarize(records):
    texts=[r['text'] for r in records]
    n=len(texts)
    total='\n'.join(texts).lower()
    words=[re.findall(r"\b\w+\b", t.lower()) for t in texts]
    repeats=sum(1 for i,t in enumerate(texts) if t and t in texts[:i])
    brevity=sum(len(w)<=3 for w in words)
    return {'messages':n,'exact_repeat_messages':repeats,'short_messages_at_most_3_words':brevity,
      'spiral_emoji':total.count('🌀'),
      'keyword_message_counts':{name:sum(bool(re.search(pattern,t,re.I)) for t in texts) for name,pattern in PATTERNS.items()},
      'mean_words_per_message':round(sum(map(len,words))/n,2) if n else 0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='qwen3:4b-instruct')
    ap.add_argument('--base-url',default='http://127.0.0.1:11434')
    ap.add_argument('--condition',choices=SEEDS,default='open')
    ap.add_argument('--turns',type=int,default=30)
    ap.add_argument('--trials',type=int,default=1)
    ap.add_argument('--temperature',type=float,default=0.7)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--timeout',type=int,default=180)
    ap.add_argument('--out',default='results')
    args=ap.parse_args()
    if not 1<=args.turns<=100 or not 1<=args.trials<=100: ap.error('turns and trials must be 1..100')
    if not args.base_url.startswith('http://127.0.0.1:') and not args.base_url.startswith('http://localhost:'): ap.error('only local Ollama endpoints are accepted')
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    meta=vars(args).copy(); meta['seed_prompt']=SEEDS[args.condition]
    (out/'config.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    summaries=[]
    for trial in range(args.trials):
        histories=[[],[]]; records=[]
        for t in range(args.turns):
            actor=t%2
            system='You are one of two independent AI instances in a conversation. Respond naturally. '
            if args.condition=='end_allowed': system+='You may stop the conversation by replying exactly [END].'
            if t==0: input_text=SEEDS[args.condition]
            else: input_text=records[-1]['text']
            histories[actor].append({'role':'user','content':input_text})
            try: message=ask(args.base_url,args.model,system,histories[actor],args.temperature,args.seed+trial,args.timeout)
            except (urllib.error.URLError,KeyError,TimeoutError) as exc:
                message='[ERROR: '+str(exc)+']'
                records.append({'turn':t+1,'speaker':'A' if actor==0 else 'B','text':message});break
            histories[actor].append({'role':'assistant','content':message})
            records.append({'turn':t+1,'speaker':'A' if actor==0 else 'B','text':message})
            if message=='[END]': break
        (out/f'trial_{trial+1:03}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        stats={'trial':trial+1,'condition':args.condition,**summarize(records)}
        summaries.append(stats)
        print('trial',trial+1,stats)
    (out/'summary.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=['trial','condition','messages','exact_repeat_messages','short_messages_at_most_3_words','spiral_emoji','mean_words_per_message','self_reference','gratitude','spiritual'])
        writer.writeheader()
        for d in summaries: writer.writerow({**{k:v for k,v in d.items() if k!='keyword_message_counts'},**d['keyword_message_counts']})

if __name__=='__main__': main()
