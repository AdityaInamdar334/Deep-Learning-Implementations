#!/usr/bin/env python3
"""Create dependency-free SVG metric charts from measured benchmark CSV."""
import argparse,csv,html,statistics
from collections import defaultdict
from pathlib import Path

def svg_bar(data,title,unit,path):
 width=870; height=120+len(data)*76
 mx=max([v for _,v in data]+[1]);parts=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/><text x="28" y="41" font-size="23" font-family="Arial" fill="#17233a">{html.escape(title)}</text>']
 for i,(name,value) in enumerate(data):
  y=75+i*76; bw=int(520*value/mx)
  parts.append(f'<text x="28" y="{y+20}" font-family="Arial" font-size="15">{html.escape(name)}</text><rect x="180" y="{y}" width="{bw}" height="28" rx="4" fill="#385f9b"/><text x="{192+bw}" y="{y+20}" font-family="Arial" font-size="14">{value:.2f} {html.escape(unit)}</text>')
 parts.append('</svg>');path.write_text(''.join(parts))

def main():
 p=argparse.ArgumentParser();p.add_argument('--csv',default='results/run/runs.csv');p.add_argument('--out',default='results/run/plots');a=p.parse_args();rows=list(csv.DictReader(open(a.csv)));out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 models=sorted(set(x['model'] for x in rows)); metrics={'accuracy':('Exact-score accuracy','%'),'decode_tps':('Median generation throughput','tokens/s'),'prefill_tps':('Median prompt throughput','tokens/s'),'incremental_gpu_mib':('Median incremental GPU memory','MiB'),'wall_s':('Median end-to-end latency','s')}
 for key,(title,unit) in metrics.items():
  data=[]
  for name in models:
   rs=[r for r in rows if r['model']==name]
   if key=='accuracy':val=sum(float(r['score']) for r in rs)/len(rs)*100 if rs else 0
   else:
    nums=[float(r[key]) for r in rs if r.get(key) not in ('','None',None) and not r['error']]
    if not nums:continue
    val=statistics.median(nums)
   data.append((name,val))
  if data:svg_bar(data,title,unit,out/(key+'.svg'))
 print('Charts written to',out)
if __name__=='__main__':main()
