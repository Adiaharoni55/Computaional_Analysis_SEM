"""
Bacteria feature comparison across treatments using Cliff's delta effect size.
Loads per-bacterium feature CSVs and produces 5 boxplot comparisons.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────────
FEATURES_DIR = "results/feature_extraction/features"
OUT_DIR = "results/bacteria_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

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

PLOTS = [
    ("minor_axis", "(B) Minor Axis",  "Minor Axis Length (µm)"),
    ("major_axis", "(A) Major Axis",  "Major Axis Length (µm)"),
    ("aspect_ratio", "(C) Aspect Ratio", "Aspect Ratio (major/minor)"),
    ("area",         "Area",             "Area (µm²)"),
    ("texture",      "Texture",          "Texture (RMS residual)"),
]

# ── Load data ──────────────────────────────────────────────────────────────────
def load_treatment(treatment: str) -> pd.DataFrame:
    pattern = os.path.join(FEATURES_DIR, treatment, "*.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No CSVs found for treatment '{treatment}' in {pattern}")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["treatment"] = treatment
    return df


all_data = pd.concat([load_treatment(t) for t in TREATMENT_ORDER], ignore_index=True)

print(f"Loaded {len(all_data):,} bacteria total.")
for t in TREATMENT_ORDER:
    n = (all_data["treatment"] == t).sum()
    print(f"  {TREATMENT_LABELS[t]:<14}: {n:,} bacteria")
print()


# ── Cliff's delta ──────────────────────────────────────────────────────────────
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: proportion of (x > y) minus proportion of (x < y)."""
    x, y = np.asarray(x), np.asarray(y)
    greater = np.sum(x[:, None] > y[None, :])
    less    = np.sum(x[:, None] < y[None, :])
    return (greater - less) / (len(x) * len(y))


def delta_label(delta: float) -> str:
    ad = abs(delta)
    if ad >= 0.474:
        return "***"
    elif ad >= 0.33:
        return "**"
    elif ad >= 0.147:
        return "*"
    return "ns"


# ── Print summary table ────────────────────────────────────────────────────────
ctrl_df = all_data[all_data["treatment"] == "control"]

for col, title, _ in PLOTS:
    print(f"── {title} — Cliff's δ vs control ──────────────────────────────")
    print(f"  {'Treatment':<14} {'n_ctrl':>6} {'n_trt':>6} {'δ':>8} {'effect':>8} {'min':>10} {'max':>10}")
    print("-" * 70)
    ctrl_vals = ctrl_df[col].dropna().values
    all_feature_vals = all_data[col].dropna().values
    print(f"  {'[overall]':<14} {'':>6} {'':>6} {'':>8} {'':>8} {all_feature_vals.min():>10.3f} {all_feature_vals.max():>10.3f}")
    for t in TREATMENT_ORDER[1:]:
        trt_vals = all_data[all_data["treatment"] == t][col].dropna().values
        d = cliffs_delta(ctrl_vals, trt_vals)
        lbl = delta_label(d)
        print(f"  {TREATMENT_LABELS[t]:<14} {len(ctrl_vals):>6} {len(trt_vals):>6} {d:>8.3f} {lbl:>8} {trt_vals.min():>10.3f} {trt_vals.max():>10.3f}")
    print(f"  {'control':<14} {len(ctrl_vals):>6} {'':>6} {'':>8} {'':>8} {ctrl_vals.min():>10.3f} {ctrl_vals.max():>10.3f}")
    print()


BLUE = "#2c5f8a"
LIGHT_BLUE = "#d0e4f2"

def make_boxplot(col: str, title: str, ylabel: str):
    groups = [
        all_data[all_data["treatment"] == t][col].dropna().values
        for t in TREATMENT_ORDER
    ]
    ctrl_vals = groups[0]

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f9fa")

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

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(BLUE)
        patch.set_edgecolor(BLUE)
        patch.set_linewidth(1.8)
        patch.set_alpha(0.85)

    # Bold axis borders
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(2.0)
        ax.spines[spine].set_color("#222222")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
    ax.set_xticklabels(
        [TREATMENT_LABELS[t] for t in TREATMENT_ORDER],
        fontsize=14, fontweight="bold", color="#222222"
    )
    ax.set_ylabel(ylabel, fontsize=15, fontweight="bold", color="#222222", labelpad=10)
    ax.set_xlabel("Arachidonic acid (µg/ml)", fontsize=15, fontweight="bold", color="#222222", labelpad=10)
    ax.set_title(
        f"{title}",
        fontsize=16, fontweight="bold", color="#1a1a2e", pad=40
    )
    ax.tick_params(axis='y', labelsize=13, width=2, length=5, color="#222222")
    ax.tick_params(axis='x', width=2, length=5, color="#222222")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#cccccc", linewidth=1.0)
    ax.set_axisbelow(True)

    if col in ("minor_axis", "major_axis"):
        ax.set_ylim(0, 4)

    # Significance annotations
    all_vals = np.concatenate(groups)
    y_range = all_vals.max() - all_vals.min()
    y_offset = y_range * 0.04

    for i, (t, g) in enumerate(zip(TREATMENT_ORDER[1:], groups[1:]), start=2):
        d = cliffs_delta(ctrl_vals, g)
        lbl = delta_label(d)
        direction = "" if lbl == "ns" else ("↑" if np.median(g) > np.median(ctrl_vals) else "↓")
        whisker_top = np.percentile(g, 75) + 1.5 * (np.percentile(g, 75) - np.percentile(g, 25))
        whisker_top = min(whisker_top, g.max())
        d_str = f"δ={d:+.3f}"
        annotation = f"{lbl}{direction}\n{d_str}" if lbl != "ns" else lbl
        color = "#2980b9"
        ax.text(i, whisker_top + y_offset, annotation,
                ha="center", va="bottom", fontsize=14,
                fontweight="bold", color=color)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = os.path.join(OUT_DIR, f"{col}_comparison.png")
    plt.savefig(fname, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"── {title} — Median values ──────────────────────────────")
    ctrl_median = np.median(ctrl_vals)
    print(f"  Control median: {ctrl_median:.3f}")
    for t in TREATMENT_ORDER[1:]:
        trt_vals = all_data[all_data["treatment"] == t][col].dropna().values
        trt_median = np.median(trt_vals)
        pct_change = ((trt_median - ctrl_median) / ctrl_median) * 100
        print(f"  {TREATMENT_LABELS[t]:<14}: median={trt_median:.3f}, change={pct_change:+.1f}%")
    print()

    print(f"Saved → {fname}")


# ── Generate all 5 plots ───────────────────────────────────────────────────────
for col, title, ylabel in PLOTS:
    make_boxplot(col, title, ylabel)

print("\nDone.")
