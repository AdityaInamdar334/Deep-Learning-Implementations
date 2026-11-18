#!/usr/bin/env python3
"""Create dependency-free SVG comparison of actual summary JSON files."""
import argparse, json, html
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('summary',nargs='+');p.add_argument('--out',default='results/comparison.svg');a=p.parse_args()
 runs=[(Path(x).parent.name,json.loads(Path(x).read_text())) for x in a.summary]
 vals=[(n, d['retrieval_recall_at_k'], d.get('grounded_correct_rate')) for n,d in runs]
 h=130+len(vals)*72;shapes=[f'<svg xmlns="http://www.w3.org/2000/svg" width="920" height="{h}" viewBox="0 0 920 {h}">','<rect width="920" height="100%" fill="#fafafa"/>','<text x="24" y="38" font-size="23" font-family="Arial" fill="#222">RAG: retrieval vs grounded answer</text>','<text x="24" y="62" font-size="13" font-family="Arial" fill="#555">Dark: evidence-bearing retrieval Recall@k · Light: grounded answer rate</text>']
 for i,(name,r,g) in enumerate(vals):
  y=94+i*72
  shapes.extend([f'<text x="24" y="{y}" font-size="14" font-family="Arial">{html.escape(name)}</text>', f'<rect x="245" y="{y-16}" width="{int(550*r)}" height="15" fill="#275d87"/>',f'<text x="{252+int(550*r)}" y="{y-3}" font-size="13" font-family="Arial">{r:.0%}</text>'])
  if g is not None:shapes.extend([f'<rect x="245" y="{y+5}" width="{int(550*g)}" height="15" fill="#85a9bd"/>', f'<text x="{252+int(550*g)}" y="{y+18}" font-size="13" font-family="Arial">{g:.0%}</text>'])
 shapes.append('</svg>');out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(shapes));print(out)
if __name__=='__main__':main()
