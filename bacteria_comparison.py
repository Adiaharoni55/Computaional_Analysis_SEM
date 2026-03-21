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

PLOTS = [
    ("minor_radius", "Minor Axis",  "Minor Axis Length (µm)"),
    ("major_radius", "Major Axis",  "Major Axis Length (µm)"),
    ("aspect_ratio", "Ratio",       "Aspect Ratio (major/minor)"),
    ("area",         "Area",        "Area (µm²)"),
    ("texture",      "Texture",     "Texture (local std dev)"),
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
    print(f"  {'Treatment':<14} {'n_ctrl':>6} {'n_trt':>6} {'δ':>8} {'effect':>8}")
    print("-" * 50)
    ctrl_vals = ctrl_df[col].dropna().values
    for t in TREATMENT_ORDER[1:]:
        trt_vals = all_data[all_data["treatment"] == t][col].dropna().values
        d = cliffs_delta(ctrl_vals, trt_vals)
        lbl = delta_label(d)
        print(f"  {TREATMENT_LABELS[t]:<14} {len(ctrl_vals):>6} {len(trt_vals):>6} {d:>8.3f} {lbl:>8}")
    print()


# ── Plot helper ────────────────────────────────────────────────────────────────
BLUE = "#4a7ba7"

def make_boxplot(col: str, title: str, ylabel: str):
    groups = [
        all_data[all_data["treatment"] == t][col].dropna().values
        for t in TREATMENT_ORDER
    ]
    ctrl_vals = groups[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    bp = ax.boxplot(
        groups,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops=dict(color="white", linewidth=2),
        whiskerprops=dict(color="#333333", linewidth=1.2),
        capprops=dict(color="#333333", linewidth=1.2),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.2)

    ax.set_xticks(range(1, len(TREATMENT_ORDER) + 1))
    ax.set_xticklabels([TREATMENT_LABELS[t] for t in TREATMENT_ORDER], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Treatment", fontsize=12)
    ax.set_title(f"Bacteria {title} Comparison Across Treatments",
                 fontsize=13, fontweight="bold", pad=40)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#aaaaaa")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Significance annotations (Cliff's δ vs control)
    ctrl_mean = np.mean(ctrl_vals)
    all_vals = np.concatenate(groups)
    y_range = all_vals.max() - all_vals.min()
    y_offset = y_range * 0.04

    for i, (t, g) in enumerate(zip(TREATMENT_ORDER[1:], groups[1:]), start=2):
        d = cliffs_delta(ctrl_vals, g)
        lbl = delta_label(d)
        direction = "" if lbl == "ns" else ("↑" if np.median(g) > np.median(ctrl_vals) else "↓")
        whisker_top = np.percentile(g, 75) + 1.5 * (np.percentile(g, 75) - np.percentile(g, 25))
        whisker_top = min(whisker_top, g.max())
        annotation = f"{lbl}{direction}\nδ={d:.2f}"
        ax.text(i, whisker_top + y_offset, annotation,
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#333333")

    # Legend for effect size
    legend_text = "ns = not significant (|δ|<0.147)  * = small  ** = medium  *** = large (|δ|≥0.474)"
    fig.text(0.5, 0.01, legend_text, ha="center", fontsize=8, color="#555555")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    fname = os.path.join(OUT_DIR, f"{col}_comparison.png")
    plt.savefig(fname, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved → {fname}")


# ── Generate all 5 plots ───────────────────────────────────────────────────────
for col, title, ylabel in PLOTS:
    make_boxplot(col, title, ylabel)

print("\nDone.")
