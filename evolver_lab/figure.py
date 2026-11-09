"""One square per paired execution. Counts derived from reported marginals and overlap."""
import os
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_cache"
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
groups = [(265, "#57798c", "Passed twice"), (15, "#14734f", "Gained a pass"),
          (1, "#ae3c27", "Lost a pass"), (39, "#c4c9cb", "Failed twice")]
colors = [color for count, color, _ in groups for _ in range(count)]
fig, ax = plt.subplots(figsize=(10,5.4), layout="constrained")
ax.scatter([i % 32 for i in range(320)], [9-i//32 for i in range(320)],
           c=colors, marker="s", s=90, linewidth=0)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("What changed on the second attempt?",loc="left",fontsize=20,pad=18)
ax.legend(handles=[Patch(facecolor=c,label=f"{n}  {title}") for n,c,title in groups],
          loc="upper left", bbox_to_anchor=(0,-.02),ncol=2,frameon=False)
fig.text(.03,.025,"320 paired observations grouped by outcome. One square represents one task in one harness/model configuration.",fontsize=9)
fig.savefig(Path(__file__).with_name("paired-outcomes.png"),dpi=170)
