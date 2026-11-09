"""Reconstruct paired outcomes using only the paper's published aggregates."""
from decimal import Decimal
from math import ceil

pairs = 320
# Per-harness rates in section 4.2 give exact integer marginals.
baseline = int(160 * Decimal("0.85") + 160 * Decimal("0.8125"))
augmented = int(320 * Decimal("0.875"))
both = 265
gains = augmented - both
losses = baseline - both
neither = pairs - both - gains - losses
print(f"Both pass {both}; fail to pass {gains}; pass to fail {losses}; both fail {neither}")
assert (both, gains, losses, neither) == (265, 15, 1, 39)
print(f"Net improvement: {100 * (gains - losses) / pairs:.3f} percentage points")
# Mean tokens per run, in thousands, rounded in section 4.4.
for label, before, after in [("All pairs", 1641.1, 1243.3), ("Both pass", 1278.2, 1053.6)]:
    print(f"{label}: reduction from rounded means = {100*(1-after/before):.2f}%")
# Hypothetical engineering budget, not measured by Evolver.
capture_cost, saving_per_hit, cost_per_miss = 60000, 12000, 1000
for hit_rate in (0.2, 0.5, 0.8):
    saving = hit_rate * saving_per_hit - (1-hit_rate) * cost_per_miss
    print(f"Illustration: {hit_rate:.0%} useful retrievals, {ceil(capture_cost/saving)} attempts to break even")
