"""Plot request-level TTFT and TPOT percentiles from a run.py JSON trace."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("trace", type=Path)
p.add_argument("--out",type=Path,default=Path("figures/my_run.png"))
args = p.parse_args()
data = json.loads(args.trace.read_text())
fig, axes = plt.subplots(1,2,figsize=(10,4),layout="constrained")
groups = data["groups"]
for ax,key,label,color in zip(axes,["ttft_ms","tpot_ms"],["First token","Later tokens"],["#155b76","#ac5936"]):
    values = [g[key]["p50"] for g in groups]
    high = [g[key]["p95"] for g in groups]
    if any(v is None for v in values):
        ax.text(.5,.5,"Requires two or more output tokens",transform=ax.transAxes,ha="center")
    else:
        ax.bar(range(len(groups)),values,color=color,width=.5,label="p50")
        ax.scatter(range(len(groups)),high,color="#1e2d35",marker="_",s=140,label="p95")
        ax.legend(frameon=False)
    ax.set_xticks(range(len(groups)),[g["length"] for g in groups])
    ax.set_xlabel("Initial prompt length (tokens)")
    ax.set_title(label)
    ax.set_ylabel("Request-level latency (ms)")
    ax.spines[["top","right"]].set_visible(False)
fig.suptitle(f"{data['model']}  |  {data['device'].upper()}  |  {data['runs']} requests per length  |  {data['new']} output tokens")
args.out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.out,dpi=180)
