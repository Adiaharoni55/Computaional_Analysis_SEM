"""
Structure comparison: Mann-Whitney U test on image mean texture across treatments.
Uses all 18 images per treatment at magnification 20000.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from itertools import combinations
import matplotlib.cm as cm
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with open(CONFIG_PATH) as _f:
    CONFIG = json.load(_f)

_SC_CFG = CONFIG["structure_comparison"]

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(_SC_CFG["summary_csv"])

# Treatment display order (control first, then ascending concentration)
TREATMENT_ORDER = CONFIG["shared"]["treatment_order"]
TREATMENT_LABELS = CONFIG["shared"]["treatment_labels"]

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
BLUE = CONFIG["colors"]["blue"]
LIGHT_BLUE = CONFIG["colors"]["light_blue"]

fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

medians = [np.median(g) for g in groups]
norm = plt.Normalize(min(medians), max(medians))
cmap = cm.get_cmap('Blues_r')
blues = [cmap(norm(m)) for m in medians]

bp = ax.boxplot(
    groups,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(color=BLUE, linewidth=1.8, linestyle="--"),
    capprops=dict(color=BLUE, linewidth=2.2),
    boxprops=dict(linewidth=1.8),
)


for i, (patch, color) in enumerate(zip(bp["boxes"], blues)):
    patch.set_facecolor(color)
    patch.set_edgecolor("#1a3a5c")
    patch.set_linewidth(1.8)
    patch.set_alpha(0.9)

for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(2.0)
    ax.spines[spine].set_color("#222222")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Axis labels & formatting
ax.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
ax.set_xticklabels([TREATMENT_LABELS[t] for t in TREATMENT_ORDER], fontsize=14, fontweight="bold", color="#222222")
ax.tick_params(axis='y', labelsize=13, width=2, length=5, color="#222222")
ax.tick_params(axis='x', width=2, length=5, color="#222222")
ax.set_ylabel("Mean Patch Level STD of Edges", fontsize=15, fontweight="bold", color="#222222", labelpad=10)
ax.set_xlabel("Arachidonic acid (µg/ml)", fontsize=15, fontweight="bold", color="#222222", labelpad=10)
ax.set_title("Biofilm Matrix Quantification", fontsize=16, fontweight="bold", color="#1a1a2e", pad=12)
ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc", linewidth=1.0)
ax.set_axisbelow(True)

# Significance labels based on Bonferroni-corrected U-test p-value
def sig_label(p):
    thresholds = _SC_CFG["significance_thresholds"]
    if p < thresholds["p001"]:
        return "***"
    elif p < thresholds["p01"]:
        return "**"
    elif p < thresholds["p05"]:
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
    color = "#2980b9"
    ax.text(i, whisker_top + y_offset, annotation, ha="center", va="bottom",
            fontsize=14, fontweight="bold", color=color)

plt.tight_layout()
out_path = _SC_CFG["texture_boxplot_out"]
plt.savefig(out_path, dpi=_SC_CFG["figure_dpi"], bbox_inches="tight", facecolor="white")
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

fig2, ax2 = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

bp2 = ax2.boxplot(
    cov_groups,
    patch_artist=True,
    widths=0.55,
    showfliers=False,
    zorder=0,                          # ← push all original boxes to background
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(color=BLUE, linewidth=1.8, linestyle="--"),
    capprops=dict(color=BLUE, linewidth=2.2),
    boxprops=dict(linewidth=1.8),
)

for i, patch in enumerate(bp2["boxes"]):
    patch.set_facecolor(BLUE)
    patch.set_edgecolor(BLUE)
    patch.set_linewidth(1.8)
    patch.set_alpha(0.85)

for i, (patch, g) in enumerate(zip(bp2["boxes"], cov_groups), start=1):
    q1, q3 = np.percentile(g, [25, 75])
    if q1 == q3:                       # ← collapsed box (IQR = 0)
        patch.set_visible(False)
        bar_height = 4
        ax2.bar(i, bar_height, bottom=np.median(g) - bar_height / 2, width=0.55,
                color=BLUE, edgecolor="#1a3a5c", linewidth=1.8, zorder=3, alpha=0.85)
        ax2.plot([i - 0.275, i + 0.275], [np.median(g), np.median(g)],
                 color="white", linewidth=2.5, zorder=4)
    else:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor(BLUE)
        patch.set_linewidth(1.8)
        patch.set_alpha(0.85)


for spine in ["left", "bottom"]:
    ax2.spines[spine].set_linewidth(2.0)
    ax2.spines[spine].set_color("#222222")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

ax2.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
ax2.set_xticklabels([TREATMENT_LABELS[t] for t in TREATMENT_ORDER], fontsize=14, fontweight="bold", color="#222222")
ax2.tick_params(axis='y', labelsize=13, width=2, length=5, color="#222222")
ax2.tick_params(axis='x', width=2, length=5, color="#222222")
ax2.set_ylabel("Bacteria Coverage (%)", fontsize=15, fontweight="bold", color="#222222", labelpad=10)
ax2.set_xlabel("Arachidonic acid (µg/ml)", fontsize=15, fontweight="bold", color="#222222", labelpad=10)
ax2.set_title("Biofilm Coverage Comparison Across Treatments", fontsize=16, fontweight="bold", color="#1a1a2e", pad=12)
ax2.set_ylim(*_SC_CFG["coverage_ylim"])
ax2.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc", linewidth=1.0)
ax2.set_axisbelow(True)

ctrl_cov_mean = np.mean(cov_groups[0])

for i, t in enumerate(TREATMENT_ORDER[1:], start=2):
    p = cov_results[t]
    p_bonf = min(p * N_COMPARISONS, 1.0)
    label = sig_label(p_bonf)
    arrow = "" if label == "ns" else ("↑" if np.mean(cov_groups[i - 1]) > ctrl_cov_mean else "↓")
    whisker_top = max(cov_groups[i - 1].max(), 100) + 2
    p_str = "" if label == "ns" else (f"p={p_bonf:.2e}" if p_bonf < 0.001 else f"p={p_bonf:.3f}")
    annotation = f"{label}{arrow}\n{p_str}" if p_str else label
    color = "#c0392b" if "↑" in annotation else "#2980b9" if "↓" in annotation else "#555555"
    ax2.text(i, whisker_top, annotation, ha="center", va="bottom",
             fontsize=14, fontweight="bold", color=color)

plt.tight_layout()
cov_out_path = _SC_CFG["coverage_boxplot_out"]
fig2.savefig(cov_out_path, dpi=_SC_CFG["figure_dpi"], bbox_inches="tight", facecolor="white")
print(f"\nPlot saved → {cov_out_path}")
plt.show()
