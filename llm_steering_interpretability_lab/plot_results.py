"""Optional plot, requires matplotlib and the saved benchmark JSON."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

p = json.loads(Path("results/benchmark.json").read_text())["summary"]
names = ["baseline", "mood_0.5", "mood_1", "mood_2", "mood_4", "random_2", "topic_2"]
xs = range(len(names))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(xs, [p[n]["sad_probability"]["mean"] for n in names],
       yerr=[p[n]["sad_probability"]["std"] for n in names], capsize=3)
ax.set_xticks(list(xs), names, rotation=30, ha="right")
ax.set_ylabel("Mean probability of gloomy label")
ax.set_title("Steering intervention versus control directions, synthetic network")
fig.tight_layout()
Path("figures").mkdir(exist_ok=True)
fig.savefig("figures/01_steering_probabilities.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
name2 = ["mood_0.5", "mood_1", "mood_2", "mood_4"]
ax.plot([.5, 1, 2, 4], [p[n]["mood_kl"]["mean"] for n in name2], marker="o", label="Mood KL")
ax.plot([.5, 1, 2, 4], [p[n]["topic_kl"]["mean"] for n in name2], marker="s", label="Topic KL")
ax.set(xlabel="Steering strength", ylabel="KL divergence against baseline", title="Intervention changes distributions, synthetic model")
ax.legend()
fig.tight_layout()
fig.savefig("figures/02_kl_tradeoff.png", dpi=180)
plt.close(fig)
print("Saved two figures in figures/")
