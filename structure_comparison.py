"""
Structure comparison: Mann-Whitney U test on image mean texture across treatments.
Uses all 18 images per treatment at magnification 20000.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("results/matrix/combined_summary.csv")

# Treatment display order (control first, then ascending concentration)
TREATMENT_ORDER = [
    "control",
    "0.1 Eth",
    "6.25 ug:ml",
    "12.5 ug:ml",
    "25 ug:ml",
    "50 ug:ml",
]

TREATMENT_LABELS = {
    "control":     "Control",
    "0.1 Eth":     "0.1% EtOH",
    "6.25 ug:ml":  "6.25 µg/ml",
    "12.5 ug:ml":  "12.5 µg/ml",
    "25 ug:ml":    "25 µg/ml",
    "50 ug:ml":    "50 µg/ml",
}

groups = [df[df["treatment"] == t]["mean_texture"].values for t in TREATMENT_ORDER]

# ── Mann-Whitney U pairwise tests ───────────────────────────────────────────────
N_COMPARISONS = len(TREATMENT_ORDER) - 1  # 5 tests vs control

print("── Group descriptive stats ──────────────────────────────────────")
print(f"  {'Treatment':<14} {'n':>3} {'mean':>7} {'median':>7} {'std':>7} {'min':>7} {'max':>7}")
print("-" * 60)
for t, g in zip(TREATMENT_ORDER, groups):
    print(f"  {TREATMENT_LABELS[t]:<14} {len(g):>3} {np.mean(g):>7.2f} {np.median(g):>7.2f} {np.std(g):>7.2f} {g.min():>7.2f} {g.max():>7.2f}")
print()

print("=" * 60)
print("Mann-Whitney U — control vs each treatment  (Bonferroni corrected, n=5)")
print(f"  {'Treatment':<14} {'U':>7} {'p_raw':>9} {'p_bonf':>9} {'sig':>4}")
print("-" * 60)
ctrl_vals = groups[0]
results = {}
for t, g in zip(TREATMENT_ORDER[1:], groups[1:]):
    stat, p = stats.mannwhitneyu(ctrl_vals, g, alternative="two-sided")
    p_bonf = min(p * N_COMPARISONS, 1.0)
    sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
    results[t] = p
    print(f"  {TREATMENT_LABELS[t]:<14} {stat:>7.1f} {p:>9.4f} {p_bonf:>9.4f} {sig:>4}")

print()
print("All pairwise:")
for (t1, g1), (t2, g2) in combinations(zip(TREATMENT_ORDER, groups), 2):
    stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
    print(f"  {TREATMENT_LABELS[t1]:>12s} vs {TREATMENT_LABELS[t2]:<12s}:  U={stat:.1f},  p={p:.4f}")

# ── Plot ────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

blues = ["#4a7ba7"] * len(TREATMENT_ORDER)

bp = ax.boxplot(
    groups,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(color="#333333", linewidth=1.2),
    capprops=dict(color="#333333", linewidth=1.2),
    flierprops=dict(marker="o", markerfacecolor="#555555", markersize=4,
                    markeredgewidth=0.5, alpha=0.7),
)

for patch, color in zip(bp["boxes"], blues):
    patch.set_facecolor(color)
    patch.set_edgecolor("#333333")
    patch.set_linewidth(1.2)

# Axis labels & formatting
ax.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
ax.set_xticklabels([TREATMENT_LABELS[t] for t in TREATMENT_ORDER], fontsize=11)
ax.set_ylabel("Mean Matrix Value (image mean)", fontsize=12)
ax.set_xlabel("Treatment", fontsize=12)
ax.set_title(
    "Biofilm Stracture Comparison Across Treatments",
    fontsize=13, fontweight="bold", pad=12,
)

ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#aaaaaa")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Significance labels based on Bonferroni-corrected U-test p-value
def sig_label(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"

ctrl_mean = np.mean(groups[0])
y_range = max(g.max() for g in groups) - min(g.min() for g in groups)
y_offset = y_range * 0.04

for i, t in enumerate(TREATMENT_ORDER[1:], start=2):
    p = results[t]
    p_bonf = min(p * N_COMPARISONS, 1.0)
    label = sig_label(p_bonf)
    arrow = "" if label == "ns" else ("↑" if np.mean(groups[i - 1]) > ctrl_mean else "↓")
    whisker_top = groups[i - 1].max()
    p_str = "" if label == "ns" else (f"p={p_bonf:.2e}" if p_bonf < 0.001 else f"p={p_bonf:.3f}")
    annotation = f"{label}{arrow}\n{p_str}" if p_str else label
    ax.text(i, whisker_top + y_offset, annotation, ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333333")

plt.tight_layout()
out_path = "results/structure_comparison_boxplot.png"
plt.savefig(out_path, dpi=180, bbox_inches="tight")
print(f"\nPlot saved → {out_path}")
plt.show()
