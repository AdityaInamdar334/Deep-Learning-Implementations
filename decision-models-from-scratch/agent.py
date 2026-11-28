"""Safe, dry-run support agent: learned intent, deterministic policy, fallback."""
import json
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from support import generate,text,next_action,policy

def run():
    splits=generate(); tr=[e for e in splits['train'] if e['question']=='Classify intent']; cal=[e for e in splits['calibration'] if e['question']=='Classify intent']; te=[e for e in splits['test'] if e['question']=='Choose next action']
    model=make_pipeline(TfidfVectorizer(analyzer='char',ngram_range=(2,5)),LogisticRegression(max_iter=1000)); model.fit([text(e) for e in tr],[e['answer'] for e in tr]); cp=model.predict_proba([text(e) for e in cal]); cy=[e['answer'] for e in cal]; threshold=1.01
    for t in sorted(set(cp.max(1))):
        accept=cp.max(1)>=t
        if accept.sum()>=10 and (model.classes_[cp.argmax(1)][accept]==__import__('numpy').array(cy)[accept]).mean()>=.95: threshold=float(t); break
    counts=dict(total=len(te),local=0,deferred=0,blocked=0,correct_local=0)
    for e in te:
        s={k:v for k,v in e['state'].items() if k!='intent'}; p=model.predict_proba([json.dumps({'message':s['message']},sort_keys=True)])[0]
        if p.max()<threshold: counts['deferred']+=1; continue
        proposed=next_action(model.classes_[p.argmax()],s)
        if not policy(proposed,s): counts['blocked']+=1; continue
        counts['local']+=1; counts['correct_local']+=int(proposed==e['answer'])
    counts['threshold']=threshold; counts['local_accuracy']=counts['correct_local']/counts['local'] if counts['local'] else None
    counts['note']='Fallback is a deferral, not an executed or assumed-correct LLM. No external tools are invoked.'
    Path('results/agent.json').write_text(json.dumps(counts,indent=2)); print(json.dumps(counts,indent=2))
if __name__=='__main__': run()
