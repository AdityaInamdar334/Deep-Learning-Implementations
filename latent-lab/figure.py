"""Optional figure. Requires matplotlib."""
from pathlib import Path
import matplotlib.pyplot as plt
from experiment import SCALE, schedule

a_values = schedule()
v = SCALE ** -2
fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
for label, color, curve in [
    ("Training units", "#11685a", a_values),
    ("Scale omitted", "#ab401e", [a*a + a*(1-a)*v for a in a_values]),
    ("Oracle recalibrated", "#4053a0", [a*v/(a*v+1-a) for a in a_values]),
]:
    ax.plot(curve, color=color, label=label, linewidth=2.5)
ax.axvline(500, color="0.5", linestyle=":", linewidth=1)
ax.set(xlabel="Training schedule index", ylabel="Expected squared error in predicted noise",
       title="The cost of feeding a denoiser the wrong units")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.savefig(Path(__file__).with_name("latent-scale.png"), dpi=150)
