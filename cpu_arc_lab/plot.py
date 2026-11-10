from pathlib import Path
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from costs import ROWS, D, break_even
values = sorted((break_even(D(s), D(b), D(c)), name) for name,s,b,c,_ in ROWS)
fig, ax = plt.subplots(figsize=(9,4.7),layout="constrained")
for i,(hits,name) in enumerate(values):
    ax.barh(i,hits,color="#236876",height=.56)
    ax.text(hits+.9,i,str(hits),va="center")
ax.set_yticks(range(len(values)),[name for _,name in values])
ax.set_xlim(0,100)
ax.set_xlabel("Subsequent cache hits")
ax.set_title("How many hits repay the builder?",loc="left",fontweight="bold")
ax.spines[["right","top"]].set_visible(False)
fig.savefig(Path(__file__).with_name("breakeven.png"),dpi=170)
