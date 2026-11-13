"""Optional: regenerate the article figure beside this script."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from lab import run, oracle


if __name__ == "__main__":
    target, results = run(1000,7,2048)
    fig, (left, right) = plt.subplots(1,2,figsize=(10,4),layout='constrained')
    xs = np.linspace(-3,3,500)
    for a,color in [(.90,'#a64229'),(.01,'#166a72')]:
        left.plot(xs,oracle(xs,a),color=color,label=f"Retention {a:.2f}")
        left.scatter([1],[oracle(1,a)],color=color)
    left.axhline(0,color='gray',lw=.7)
    left.axvline(1,color='gray',lw=.7,ls=':')
    left.set(xlabel="Noisy observation",ylabel="Optimal predicted noise",title="One input, two targets in expectation")
    bins=np.linspace(-3.3,3.3,48)
    for values,label,color in [(target,"Target","#202e3d"),(results["learned"],"Learned DDPM","#b3472e")]:
        right.hist(values,bins=bins,density=True,histtype="step",linewidth=1.7,color=color,label=label)
    right.set(xlabel="Sample value",ylabel="Density",title="Generated samples recover two modes")
    for ax in (left,right):
        ax.spines[["right","top"]].set_visible(False)
        ax.legend(frameon=False)
    fig.savefig(Path(__file__).with_name("experiment.png"),dpi=170)
