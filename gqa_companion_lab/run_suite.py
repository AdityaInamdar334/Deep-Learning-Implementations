"""Run a reproducible, same-data MHA/GQA/MQA training comparison."""
import argparse,json,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--text',default='demo.txt')
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--block',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu')
    p.add_argument('--out',default='runs/suite')
    a=p.parse_args()
    rows=[]
    for kv in (8,2,1):
        out=Path(a.out)/f'kv{kv}'
        subprocess.run([sys.executable,'train.py','--text',a.text,'--steps',str(a.steps),
            '--batch',str(a.batch),'--block',str(a.block),'--seed',str(a.seed),
            '--data-seed','123','--kv-heads',str(kv),'--device',a.device,'--out',str(out)],check=True)
        rows.append(json.loads((out/'metrics.json').read_text()))
    Path(a.out,'summary.json').write_text(json.dumps(rows,indent=2))
    print('Comparison saved:',Path(a.out,'summary.json'))
if __name__=='__main__':main()
