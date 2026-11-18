"""Generate a self-contained SVG retrieval score chart from evaluation.json files."""
import argparse,json,html
from pathlib import Path

def make(files,out):
    data=[]
    for f in files:
        doc=json.loads(Path(f).read_text());data.append((doc['summary']['mode'],doc['summary']))
    W=880; H=150+len(data)*92
    lines=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
      '<rect width="100%" height="100%" fill="#111827"/>',
      '<text x="30" y="40" font-family="Arial" font-size="25" font-weight="bold" fill="#f9fafb">Local Retrieval Quality</text>',
      '<text x="30" y="64" font-family="Arial" font-size="13" fill="#9ca3af">Recall@k, MRR@k and nDCG@k. Evaluate on the same labeled queries.</text>']
    colors=['#60a5fa','#34d399','#fbbf24']
    for n,(mode,row) in enumerate(data):
        y=96+n*92
        lines.append(f'<text x="30" y="{y+18}" font-family="Arial" font-size="17" fill="#f9fafb">{html.escape(mode.upper())}</text>')
        for j,(label,key) in enumerate([('Recall','recall'),('MRR','mrr'),('nDCG','ndcg')]):
            value=row.get(key,0)
            bx=190+j*225
            lines.extend([f'<text x="{bx}" y="{y}" font-family="Arial" font-size="12" fill="#d1d5db">{label}: {value:.3f}</text>',f'<rect x="{bx}" y="{y+10}" width="160" height="14" rx="5" fill="#374151"/>',f'<rect x="{bx}" y="{y+10}" width="{160*value:.1f}" height="14" rx="5" fill="{colors[j]}"/>'])
    lines.append('</svg>');Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text('\n'.join(lines))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--out',default='results/comparison.svg');args=p.parse_args();make(args.files,args.out);print('wrote',args.out)
