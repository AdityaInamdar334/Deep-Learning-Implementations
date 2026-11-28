"""Synthetic support data, baseline, and deterministic dry-run policy."""
import json, random
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score,f1_score
ROOT=Path(__file__).resolve().parent
TEMPLATES={
'billing':['I was charged twice for my {item}','There is a duplicate charge for {item}','My {item} payment appeared twice','Two debits for one {item}','Why did you take my money twice for {item}','My bank shows an extra payment for {item}'],
'returns':['My {item} arrived broken','The {item} is damaged, replace it','I want to send back my {item}','The {item} does not work out of the box','Can I get another {item} instead','The parcel contained a defective {item}'],
'account':['I forgot my password','I cannot sign in to my account','Reset my login please','My credentials are not working','I am locked out of my profile','Help me regain access'],
'shipping':['Where is my {item} delivery','Track my {item} parcel','My {item} has not arrived','The package with my {item} is late','When will the courier bring my {item}','I never received the shipment for {item}']}
INTENTS=list(TEMPLATES); ACTIONS=['authenticate','lookup_order','clarify','offer_replacement','billing_review','account_recovery','track_shipment','escalate']
def next_action(intent,s):
    if s['urgent']: return 'escalate'
    if not s['authenticated']: return 'authenticate'
    if intent=='account': return 'account_recovery'
    if not s['order_found']: return 'lookup_order'
    if intent=='billing': return 'billing_review'
    if intent=='shipping': return 'track_shipment'
    return 'offer_replacement' if s['replacement_eligible'] and s['wants_replacement'] else 'clarify'
def policy(action,s):
    # Dry-run only. Trusted facts must originate in backend services.
    if action in ['authenticate','clarify','escalate']: return True
    if not s.get('authenticated',False): return False
    if action in ['lookup_order','account_recovery']: return True
    if not s.get('order_found',False): return False
    if action=='offer_replacement': return bool(s.get('replacement_eligible') and s.get('wants_replacement'))
    return action in ['billing_review','track_shipment']
def example(state,q,opts,answer,case):
    return {'case_id':case,'state':state,'question':q,'options':[{'id':a,'text':a.replace('_',' ')} for a in opts],'answer':answer}
def generate():
    rng=random.Random(42); splits={k:[] for k in ['train','calibration','test']}
    for intent,templates in TEMPLATES.items():
        for j,template in enumerate(templates):
            split='train' if j<4 else 'calibration' if j==4 else 'test'
            for i in range(48):
                s={'message':template.format(item=rng.choice(['laptop','headphones','tablet','camera'])),'authenticated':rng.choice([True,False]),'order_found':rng.choice([True,False]),'replacement_eligible':rng.choice([True,False]),'wants_replacement':rng.choice([True,False]),'urgent':rng.random()<.15}
                case=f'{intent}-{j}-{i}'; splits[split].append(example({'message':s['message']},'Classify intent',INTENTS,intent,case)); splits[split].append(example(dict(s,intent=intent),'Choose next action',ACTIONS,next_action(intent,s),case)); splits[split].append(example(s,'Escalate urgently?',['yes','no'],'yes' if s['urgent'] else 'no',case))
    for k,v in splits.items(): (ROOT/f'datasets/{k}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in v))
    return splits
def text(e): return json.dumps(e['state'],sort_keys=True)
def baseline(splits):
    report={}
    for q in ['Classify intent','Choose next action','Escalate urgently?']:
        tr=[e for e in splits['train'] if e['question']==q]; te=[e for e in splits['test'] if e['question']==q]
        model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5)),LogisticRegression(max_iter=1000)); model.fit([text(e) for e in tr],[e['answer'] for e in tr]); p=model.predict([text(e) for e in te]); y=[e['answer'] for e in te]; report[q]={'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,average='macro',zero_division=0),'n':len(te)}
    return report
if __name__=='__main__':
    s=generate(); r=baseline(s); (ROOT/'results/support.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
