import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.ndimage import zoom, label, binary_dilation, binary_closing, gaussian_filter
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
with open(CONFIG_PATH) as _f:
    CONFIG = json.load(_f)

_BS_CFG = CONFIG["biofilm_structure"]

INPUT_DIR  = Path(CONFIG["shared"]["input_dir"])
OUTPUT_DIR = Path(_BS_CFG["output_dir"])
TREATMENTS = CONFIG["shared"]["treatments_scan_order"]

CROP_BOTTOM = CONFIG["shared"]["crop_bottom_px"]

# Background detection parameters
_bg_cfg = _BS_CFG["background_detection"]
BG_TILE_SIZE          = _bg_cfg["tile_size"]
BG_STD_THRESHOLD      = _bg_cfg["std_threshold"]
BG_MIN_REGION_SIZE    = _bg_cfg["min_region_size"]
BG_MIN_CONTRAST_RATIO = _bg_cfg["min_contrast_ratio"]
BG_MAX_REGION_MEAN    = _bg_cfg["max_region_mean"]   # tiles in true empty substrate have edge-std ≈ 0;
                                                       # biofilm matrix between bacteria has mean 0.2–0.9

# Texture analysis parameters
_tex_cfg = _BS_CFG["texture_analysis"]
MATRIX_TILE_SIZE     = _tex_cfg["tile_size"]
VMIN, VMAX           = _tex_cfg["vmin"], _tex_cfg["vmax"]
HEATMAP_SMOOTH_SIGMA = _tex_cfg["heatmap_smooth_sigma"]


# ============================================================================
# BACKGROUND DETECTION
# ============================================================================

def get_biofilm_mask(img):
    """Detect biofilm regions using tile-based variance analysis."""
    size = BG_TILE_SIZE
    h, w = img.shape

    scores = np.array([
        [np.std(img[i*size:(i+1)*size, j*size:(j+1)*size])
         for j in range(w // size)]
        for i in range(h // size)
    ])

    low_variance = scores < BG_STD_THRESHOLD
    if np.sum(low_variance) == 0:
        return np.ones(img.shape, dtype=bool)

    labeled_regions, num_regions = label(low_variance)
    background_mask = np.zeros_like(low_variance, dtype=bool)

    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_size = np.sum(region_mask)
        if region_size < BG_MIN_REGION_SIZE:
            continue

        region_mean   = np.mean(scores[region_mask])

        # Reject regions whose tiles carry any real edge content — those are
        # biofilm matrix gaps between bacteria, not empty substrate background.
        if region_mean > BG_MAX_REGION_MEAN:
            continue

        border = binary_dilation(region_mask, iterations=1) & ~region_mask
        contrast_ratio = (
            np.mean(scores[border]) / region_mean
            if np.sum(border) > 0 and region_mean > 0 else 0
        )
        is_sudden = contrast_ratio >= BG_MIN_CONTRAST_RATIO
        if is_sudden:
            background_mask[region_mask] = True

    bg_mask_tiles = binary_closing(background_mask, iterations=1)

    upsampled = zoom(
        bg_mask_tiles.astype(float),
        np.array(img.shape) / (np.array(bg_mask_tiles.shape) * size),
        order=0
    )
    bg_mask = cv2.resize(
        upsampled.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    return ~bg_mask


# ============================================================================
# EDGE DETECTION
# ============================================================================

def compute_edges(img):
    """Smooth image and apply Sobel edge detection."""
    edge_cfg = _BS_CFG["edge_detection"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (edge_cfg["morph_kernel_size"],) * 2)
    opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=edge_cfg["morph_open_iterations"])

    wls = cv2.ximgproc.createFastGlobalSmootherFilter(
        opened, lambda_=edge_cfg["smoother_lambda"], sigma_color=edge_cfg["smoother_sigma_color"]
    )
    smoothed = wls.filter(opened)

    sobel_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=edge_cfg["sobel_ksize"])
    sobel_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=edge_cfg["sobel_ksize"])

    edges = cv2.magnitude(sobel_x, sobel_y)
    edges = np.uint8(np.clip(edges, 0, 255))

    # Remove small isolated edge blobs (debris/noise artifacts)
    _, binary = cv2.threshold(edges, edge_cfg["binary_threshold"], 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = edge_cfg["min_debris_area_px"]  # bacteria edge loops are much larger than debris spots
    clean_mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    edges = cv2.bitwise_and(edges, edges, mask=clean_mask)
    return edges


# ============================================================================
# TEXTURE ANALYSIS
# ============================================================================

def analyze_tiles(edge_img, biofilm_mask, tile_size=MATRIX_TILE_SIZE,
                  stride=None, build_heatmap=True):
    """Tile-based texture analysis on edge image."""
    if stride is None:
        stride = tile_size // 2

    h, w = edge_img.shape
    pad  = tile_size // 2

    img_padded  = np.pad(edge_img,     pad, mode='reflect')
    mask_padded = np.pad(biofilm_mask, pad, mode='constant', constant_values=False)
    h_p, w_p = img_padded.shape

    tile_values = []

    if build_heatmap:
        score_acc  = np.zeros((h_p, w_p), dtype=np.float32)
        weight_acc = np.zeros((h_p, w_p), dtype=np.float32)
        center = tile_size // 2
        y, x   = np.ogrid[0:tile_size, 0:tile_size]
        weight_kernel = np.exp(
            -((y - center) ** 2 + (x - center) ** 2) / (2 * (tile_size / 3) ** 2)
        )

    for yi in range(0, h_p - tile_size + 1, stride):
        for xi in range(0, w_p - tile_size + 1, stride):
            tile_mask = mask_padded[yi:yi + tile_size, xi:xi + tile_size]
            if np.sum(tile_mask) / (tile_size * tile_size) <= _BS_CFG["texture_analysis"]["min_tile_coverage_frac"]:
                continue

            biofilm_pixels = img_padded[yi:yi + tile_size, xi:xi + tile_size][tile_mask]
            if len(biofilm_pixels) == 0:
                continue

            std = np.std(biofilm_pixels)
            tile_values.append(std)

            if build_heatmap:
                wstd = weight_kernel * std
                wstd[~tile_mask] = 0
                wc = weight_kernel.copy()
                wc[~tile_mask] = 0
                score_acc[yi:yi + tile_size,  xi:xi + tile_size] += wstd
                weight_acc[yi:yi + tile_size, xi:xi + tile_size] += wc

    tile_values = np.array(tile_values)

    if not build_heatmap:
        return tile_values

    scores = np.full((h_p, w_p), np.nan, dtype=np.float32)
    valid  = weight_acc > 0.01
    scores[valid] = score_acc[valid] / weight_acc[valid]
    scores = scores[pad:pad + h, pad:pad + w]
    scores[~biofilm_mask] = np.nan

    return scores, tile_values


def smooth_heatmap(scores, sigma=HEATMAP_SMOOTH_SIGMA):
    """Apply Gaussian smoothing while preserving NaN boundaries."""
    mask = ~np.isnan(scores)
    if not np.any(mask):
        return scores

    scores_filled = scores.copy()
    scores_filled[~mask] = np.nanmean(scores)
    smoothed = gaussian_filter(scores_filled.astype(float), sigma=sigma)
    smoothed[~mask] = np.nan
    return smoothed


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(original, edges, scores_raw, scores_smoothed,
                         kernel_values, title=""):
    """Create 4-panel visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Edge Detection (Sobel on Smoothed)', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(edges, cmap='gray', alpha=0.5)
    masked   = np.ma.masked_where(np.isnan(scores_smoothed), scores_smoothed)
    valid    = scores_raw[~np.isnan(scores_raw)]
    mean_val = np.mean(valid) if len(valid) > 0 else np.nan
    hm = axes[1, 0].imshow(
        masked, cmap='RdYlBu', alpha=0.4,
        norm=Normalize(VMIN, VMAX), interpolation='bilinear'
    )
    axes[1, 0].set_title(f'Texture Heatmap (Mean: {mean_val:.1f})', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(hm, ax=axes[1, 0], fraction=0.046, pad=0.04, label='STD Value')

    if len(kernel_values) > 0:
        axes[1, 1].hist(kernel_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(np.mean(kernel_values),   color='red',    linestyle='--',
                           linewidth=2, label=f'Mean: {np.mean(kernel_values):.1f}')
        axes[1, 1].axvline(np.median(kernel_values), color='orange', linestyle='--',
                           linewidth=2, label=f'Median: {np.median(kernel_values):.1f}')
        axes[1, 1].set_xlabel('Kernel STD Value', fontsize=11)
        axes[1, 1].set_ylabel('Count', fontsize=11)
        axes[1, 1].set_title(f'Distribution (n={len(kernel_values)})', fontsize=12, fontweight='bold')
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].set_xlim(0, max(VMAX, np.max(kernel_values) * 1.1))
    else:
        axes[1, 1].text(0.5, 0.5, 'No valid kernels', ha='center', va='center', fontsize=14)
        axes[1, 1].set_title('Distribution', fontsize=12, fontweight='bold')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN
# ============================================================================

def process_all():
    paths = [
        p
        for s in TREATMENTS
        for p in (INPUT_DIR / s / "20000").glob("*.tif")
        if (INPUT_DIR / s / "20000").exists()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results        = []
    treatment_data = {s: [] for s in TREATMENTS}

    print(f"Processing {len(paths)} images...\n")

    for i, path in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {path.name}")

        devnull = os.open(os.devnull, os.O_WRONLY)
        stderr_fd = os.dup(2)
        os.dup2(devnull, 2)
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        os.dup2(stderr_fd, 2)
        os.close(devnull)
        os.close(stderr_fd)
        if img is None:
            continue
        original = img[:-CROP_BOTTOM, :]

        edges        = compute_edges(original)
        biofilm_mask = get_biofilm_mask(edges)

        scores_raw, kernel_values = analyze_tiles(edges, biofilm_mask)
        scores_smoothed = smooth_heatmap(scores_raw)
        tile_values = analyze_tiles(
            edges, biofilm_mask, stride=MATRIX_TILE_SIZE, build_heatmap=False
        )

        # path = ./data/{treatment}/20000/{file}.tif
        treatment_name = path.parent.parent.name  # e.g. "control", "6.25 ug/ml"
        magnification  = path.parent.name         # "20000"

        t_clean = treatment_name.replace(' ', '_').replace(':', '')
        out_dir = OUTPUT_DIR / t_clean / magnification
        out_dir.mkdir(parents=True, exist_ok=True)

        fig = create_visualization(
            original, edges, scores_raw, scores_smoothed, kernel_values,
            f"{treatment_name} - {magnification} - {path.stem}"
        )
        fig.savefig(out_dir / f"{path.stem}.png", dpi=_BS_CFG["figure_dpi"], bbox_inches='tight')
        plt.close(fig)

        valid_tiles = tile_values[tile_values > 0] if len(tile_values) > 0 else np.array([])
        biofilm_pct = np.sum(biofilm_mask) / biofilm_mask.size * 100
        bg_pct      = 100 - biofilm_pct

        if treatment_name in treatment_data:
            treatment_data[treatment_name].append({
                'image_name':          path.name,
                'bacteria_coverage_%': round(biofilm_pct, 2),
                'image_mean':          np.mean(tile_values) if len(tile_values) > 0 else np.nan,
                'tile_values':         tile_values,
            })

        results.append({
            'treatment':               treatment_name,
            'magnification':           magnification,
            'filename':                path.name,
            'bacteria_coverage_%':     round(biofilm_pct, 2),
            'background_%':            round(bg_pct, 2),
            'mean_texture':            f"{np.mean(valid_tiles):.2f}" if len(valid_tiles) > 0 else "N/A",
            'n_tiles_overlapping':     len(kernel_values),
            'n_tiles_non_overlapping': len(tile_values),
        })

    if results:
        pd.DataFrame(results).to_csv(OUTPUT_DIR / 'combined_summary.csv', index=False)
        print(f"\nSaved: {OUTPUT_DIR}/combined_summary.csv")

    for treatment, data_list in treatment_data.items():
        if not data_list:
            continue
        rows = [{
            'image_name':          item['image_name'],
            'bacteria_coverage_%': item['bacteria_coverage_%'],
            'image_mean':          f"{item['image_mean']:.2f}" if not np.isnan(item['image_mean']) else 'N/A',
            'tile_values':         ';'.join([f'{v:.2f}' for v in item['tile_values']])
        } for item in data_list]

        t_clean = treatment.replace(' ', '_').replace(':', '').replace('.', '')
        pd.DataFrame(rows).to_csv(OUTPUT_DIR / f'{t_clean}_tile_data.csv', index=False)
        print(f"Saved: {OUTPUT_DIR}/{t_clean}_tile_data.csv")

    print(f"\nDone! Processed {len(results)} images")
    return results


if __name__ == "__main__":
    process_all()
