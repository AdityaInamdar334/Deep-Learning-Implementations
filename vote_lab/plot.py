from pathlib import Path
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from votes import majority_correct
n=range(1,66,2)
fig,ax=plt.subplots(figsize=(8.5,4.7),layout="constrained")
for p,c in [(.6,"#11675f"),(.5,"#797d83"),(.4,"#a8442e")]:
    ax.plot(n,[100*majority_correct(p,k) for k in n],label=f"One rollout correct {p:.0%}",color=c,lw=2.6)
ax.set(xlabel="Independent rollouts per prompt (odd counts)",ylabel="Probability vote is correct (%)",ylim=(0,100),xlim=(1,65))
ax.spines[["top","right"]].set_visible(False)
ax.legend(frameon=False,loc="upper left",bbox_to_anchor=(1,1))
ax.set_title("The same vote amplifies both biases",loc="left",weight="bold")
fig.savefig(Path(__file__).with_name("votes.png"),dpi=180)
