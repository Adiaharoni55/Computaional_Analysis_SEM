"""
Structure comparison: Mann-Whitney U test on image mean texture across treatments.
Uses all 18 images per treatment at magnification 20000.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import matplotlib.cm as cm


# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("results/matrix/combined_summary.csv")

# Treatment display order (control first, then ascending concentration)
TREATMENT_ORDER = [
    "control",
    "6.25 ug:ml",
    "12.5 ug:ml",
    "25 ug:ml",
    "50 ug:ml",
]

TREATMENT_LABELS = {
    "control":     "Control",
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

medians = [np.median(g) for g in groups]
norm = plt.Normalize(min(medians), max(medians))
cmap = cm.get_cmap('Blues_r')
blues = [cmap(norm(m)) for m in medians]

bp = ax.boxplot(
    groups,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="black", linewidth=1, linestyle="--"),
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
    "Biofilm Structure Comparison Across Treatments",
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

# ── Bacteria coverage boxplot ────────────────────────────────────────────────
cov_groups = [df[df["treatment"] == t]["bacteria_coverage_%"].values for t in TREATMENT_ORDER]

print("\n── Bacteria coverage descriptive stats ──────────────────────────────────")
print(f"  {'Treatment':<14} {'n':>3} {'mean':>7} {'median':>7} {'std':>7} {'min':>7} {'max':>7}")
print("-" * 60)
for t, g in zip(TREATMENT_ORDER, cov_groups):
    print(f"  {TREATMENT_LABELS[t]:<14} {len(g):>3} {np.mean(g):>7.2f} {np.median(g):>7.2f} {np.std(g):>7.2f} {g.min():>7.2f} {g.max():>7.2f}")
print()

print("=" * 60)
print("Mann-Whitney U — control vs each treatment  (Bonferroni corrected, n=5)  [coverage]")
print(f"  {'Treatment':<14} {'U':>7} {'p_raw':>9} {'p_bonf':>9} {'sig':>4}")
print("-" * 60)
ctrl_cov = cov_groups[0]
cov_results = {}
for t, g in zip(TREATMENT_ORDER[1:], cov_groups[1:]):
    stat, p = stats.mannwhitneyu(ctrl_cov, g, alternative="two-sided")
    p_bonf = min(p * N_COMPARISONS, 1.0)
    sig = "***" if p_bonf < 0.001 else "**" if p_bonf < 0.01 else "*" if p_bonf < 0.05 else "ns"
    cov_results[t] = p
    print(f"  {TREATMENT_LABELS[t]:<14} {stat:>7.1f} {p:>9.4f} {p_bonf:>9.4f} {sig:>4}")

fig2, ax2 = plt.subplots(figsize=(10, 6))

bp2 = ax2.boxplot(
    cov_groups,
    patch_artist=True,
    widths=0.55,
    medianprops=dict(color="black", linewidth=1, linestyle="--"),
)

for patch in bp2["boxes"]:
    patch.set_facecolor("#4a7ba7")
    patch.set_edgecolor("#333333")
    patch.set_linewidth(1.2)

# For zero-variance groups the box has zero height — draw an explicit filled bar
for i, g in enumerate(cov_groups, start=1):
    if np.std(g) == 0:
        ax2.bar(i, 2, bottom=np.mean(g) - 1, width=0.55,
                color="#4a7ba7", edgecolor="#333333", linewidth=1.2, zorder=2)

ax2.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
ax2.set_xticklabels([TREATMENT_LABELS[t] for t in TREATMENT_ORDER], fontsize=11)
ax2.set_ylabel("Bacteria Coverage (%)", fontsize=12)
ax2.set_xlabel("Treatment", fontsize=12)
ax2.set_title(
    "Biofilm Coverage Comparison Across Treatments",
    fontsize=13, fontweight="bold", pad=12,
)
ax2.set_ylim(0, 115)   # fixed range so flat lines at 100% are visible

ax2.yaxis.grid(True, linestyle="--", alpha=0.5, color="#aaaaaa")
ax2.set_axisbelow(True)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ctrl_cov_mean = np.mean(cov_groups[0])

for i, t in enumerate(TREATMENT_ORDER[1:], start=2):
    p = cov_results[t]
    p_bonf = min(p * N_COMPARISONS, 1.0)
    label = sig_label(p_bonf)
    arrow = "" if label == "ns" else ("↑" if np.mean(cov_groups[i - 1]) > ctrl_cov_mean else "↓")
    whisker_top = max(cov_groups[i - 1].max(), 100) + 2   # always above 100 line
    p_str = "" if label == "ns" else (f"p={p_bonf:.2e}" if p_bonf < 0.001 else f"p={p_bonf:.3f}")
    annotation = f"{label}{arrow}\n{p_str}" if p_str else label
    ax2.text(i, whisker_top, annotation, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color="#333333")

plt.tight_layout()
cov_out_path = "results/coverage_comparison_boxplot.png"
fig2.savefig(cov_out_path, dpi=180, bbox_inches="tight")
print(f"\nPlot saved → {cov_out_path}")
plt.show()
