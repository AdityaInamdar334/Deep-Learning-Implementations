#!/usr/bin/env python3
"""Render measured results as portable SVG charts using matplotlib."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS=[('requests_per_s','Completed requests / second','throughput_requests.svg'),
         ('output_tokens_per_s','Aggregate output tokens / second','throughput_tokens.svg'),
         ('latency_p50_s','Median response latency (s)','latency_p50.svg'),
         ('latency_p95_s','P95 response latency (s)','latency_p95.svg'),
         ('ttft_p95_s','P95 first visible text latency (s)','ttft_p95.svg'),
         ('failure_rate','Failure rate','failures.svg'),
         ('gpu_peak_used_mib','Sampled peak device VRAM (MiB)','gpu_peak.svg')]

def plot(input_csv, out):
    rows=list(csv.DictReader(open(input_csv,newline='')))
    target=Path(out);target.mkdir(parents=True,exist_ok=True)
    for key,label,filename in METRICS:
        xs=[]; ys=[]
        for row in rows:
            if row.get(key) not in ('','None',None):
                xs.append(int(row['concurrency']));ys.append(float(row[key]))
        if not xs: continue
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(xs,ys,marker='o');ax.set_xlabel('Simulated concurrent clients');ax.set_ylabel(label)
        ax.set_xticks(xs);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(target/filename)
        plt.close(fig)
    print('Charts saved to',target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',default='results/sweep/summary.csv');p.add_argument('--out',default='results/sweep/charts')
    a=p.parse_args();plot(a.csv,a.out)
