#!/usr/bin/env python3
"""Repeatable local model/audio sweep; all datasets and models provided by the user."""
import argparse,csv,json,statistics,sys
from pathlib import Path
from transcribe import run

def main():
    p=argparse.ArgumentParser(description='Benchmark multiple locally installed whisper.cpp models on reference WAV clips')
    p.add_argument('--cli',type=Path,required=True)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--manifest',type=Path,required=True,help='JSON array of {id,audio,reference}; relative paths are relative to manifest directory')
    p.add_argument('--out',type=Path,default=Path('results/sweep'))
    p.add_argument('--repeat',type=int,default=3);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--language',default='en')
    a=p.parse_args()
    if a.repeat<1:p.error('repeat must be >=1')
    items=json.loads(a.manifest.read_text(encoding='utf-8'))
    if not isinstance(items,list) or not items:p.error('manifest must be a nonempty JSON array')
    allrows=[]
    for model in a.models:
        for entry in items:
            for key in ('id','audio','reference'):
                if key not in entry:p.error(f'manifest item missing {key}')
            audio=(a.manifest.parent/entry['audio']).resolve()
            ref=(a.manifest.parent/entry['reference']).resolve()
            label=''.join(c if c.isalnum() or c in '-_' else '_' for c in str(entry['id']))
            if not label:p.error('empty manifest id')
            model_label=model.stem
            out=a.out/model_label/label
            for row in run(a.cli.resolve(),model.resolve(),audio,out,language=a.language,threads=a.threads,repeat=a.repeat,reference=ref):
                allrows.append({'model':model_label,'clip':label,**row})
    a.out.mkdir(parents=True,exist_ok=True)
    with (a.out/'all_runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=allrows[0].keys());w.writeheader();w.writerows(allrows)
    groups={}
    for r in allrows:groups.setdefault(r['model'],[]).append(r)
    summaries=[]
    for name,rows in groups.items():
        # Corpus WER sums substitutions/deletions/insertions, then divides by total ref words
        errs=sum(r['substitutions']+r['deletions']+r['insertions'] for r in rows)
        words=sum(r['reference_words'] for r in rows)
        summaries.append({'model':name,'runs':len(rows),'corpus_wer':round(errs/words,6),
                          'median_rtf':round(statistics.median(r['rtf'] for r in rows),4),
                          'median_wall_seconds':round(statistics.median(r['wall_seconds'] for r in rows),4)})
    (a.out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,indent=2))
if __name__=='__main__':main()
