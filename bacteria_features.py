import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from cellpose import models
from skimage.measure import regionprops
import cv2
from scipy.spatial import cKDTree


MODEL_TYPE   = "cyto3"
FLOW_THRESH  = 0.4       
CELLPROB_THRESH = -2.0   
DIAMETER = None      

INPUT_DIR  = Path('./data')
OUTPUT_DIR = Path('./results/feature_extraction')
TREATMENTS = ['6.25 ug:ml', '12.5 ug:ml', '25 ug:ml', '50 ug:ml', 'control']


def run_segmentation(img):
 
    model = models.Cellpose(model_type=MODEL_TYPE, gpu=False)
    masks, _, _, diams = model.eval(
        img,
        diameter=DIAMETER,
        channels=[0, 0],
        flow_threshold=FLOW_THRESH,
        cellprob_threshold=CELLPROB_THRESH,
    )
    print(f"Estimated diameter: {diams:.1f} px")
 
    print(f"Detected {masks.max()} bacteria")
    return masks
 

def compute_ellipse_fit_score(contour, mask_shape):
    """
    Fit an ellipse to the contour and return IoU between
    the actual mask and the ideal fitted ellipse mask.
    Score = 1.0 means perfect ellipse; lower = more irregular / occluded.
    """
    if len(contour) < 5:
        return 0.0

    ellipse = cv2.fitEllipse(contour)
    ellipse_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.ellipse(ellipse_mask, ellipse, 1, -1)

    actual_mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.drawContours(actual_mask, [contour], -1, 1, -1)

    intersection = np.logical_and(actual_mask, ellipse_mask).sum()
    union = np.logical_or(actual_mask, ellipse_mask).sum()
    return float(intersection / union) if union > 0 else 0.0


def is_touching_border(contour, mask_shape, margin=1):
    """Return True if any contour point is within `margin` pixels of the image edge."""
    h, w = mask_shape
    pts = contour.reshape(-1, 2)  # shape (N, 2): col, row
    return bool(
        np.any(pts[:, 0] <= margin) or       # left edge
        np.any(pts[:, 1] <= margin) or        # top edge
        np.any(pts[:, 0] >= w - 1 - margin) or  # right edge
        np.any(pts[:, 1] >= h - 1 - margin)     # bottom edge
    )


def extract_bacteria_features(mask: np.ndarray, min_contour_pts: int = 5,
                               min_area_px: int = 1000) -> list[dict]:
    bacteria_list = []
    dropped = []
    mask_shape = mask.shape

    SCALE_BAR_UM = 5.0
    SCALE_BAR_PX = 185
    ratio = SCALE_BAR_UM / SCALE_BAR_PX

    for prop in regionprops(mask):
        individual_mask = (mask == prop.label).astype(np.uint8)
        contours, _ = cv2.findContours(individual_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        

        # ── filters ──────────────────────────────────────────────
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < min_contour_pts:
            dropped.append((prop.label, prop.centroid, f'contour too small ({len(contour)} pts < {min_contour_pts})'))
            continue
        if prop.area < min_area_px:
            dropped.append((prop.label, prop.centroid, f'area too small ({prop.area} px < {min_area_px})'))
            continue
        # ─────────────────────────────────────────────────────────

        major_px = prop.major_axis_length / 2
        minor_px = prop.minor_axis_length / 2
        
        perimeter = cv2.arcLength(contour, True)

        bacteria_list.append({
            'bacteria_id':         prop.label,
            'contour':             contour,
            'center':              (int(prop.centroid[1]), int(prop.centroid[0])),
            'major_radius_pixels': major_px,
            'minor_radius_pixels': minor_px,
            'major_radius':        major_px * ratio,
            'minor_radius':        minor_px * ratio,
            'area_pixels':         prop.area,
            'area':                prop.area * ratio ** 2,
            'aspect_ratio':        major_px / minor_px if minor_px > 0 else float('inf'),
            'ellipse_fit_score':   compute_ellipse_fit_score(contour, mask_shape),
            'touches_border':      is_touching_border(contour, mask_shape),
        })

    # ── report dropped ────────────────────────────────────────────
    if dropped:
        print(f"Dropped {len(dropped)} regions:")
        for label, centroid, reason in dropped:
            print(f"  label={label:3d}  center=({int(centroid[1])}, {int(centroid[0])})  reason: {reason}")

    return bacteria_list


# ── Filter Bacteria ────────────────────────────────────────────────────────────

def filter_bacteria_by_shape(
    bacteria_list: list[dict],
    min_ellipse_fit: float = 0.8,
    min_area: int = 1000,
) -> tuple[list[dict], list[dict], dict]:
    """
    Filter out bacteria that are:
      - not well-described by an ellipse (occluded by neighbors), OR
      - cut off by the image border
    """
    kept = []
    for b in bacteria_list:
        if b['area_pixels'] > min_area and b['ellipse_fit_score'] >= min_ellipse_fit and not b['touches_border']:
            kept.append(b)

    return kept


def visualize_filter(image, bacteria_list_filtered, merged_bacteria_list, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(image, cmap='gray')

    kept_ids = set(id(b) for b in bacteria_list_filtered)
    kept_plotted = deleted_plotted = False

    for idx, bacteria in enumerate(merged_bacteria_list):
        contour = bacteria['contour'].squeeze()
        if contour.ndim == 1:
            contour = contour[np.newaxis, :]

        is_kept = id(bacteria) in kept_ids
        color = 'g' if is_kept else 'r'
        lw = 1.5 if is_kept else 1.2

        label = None
        if is_kept and not kept_plotted:
            label = f'Kept'
            kept_plotted = True
        elif not is_kept and not deleted_plotted:
            label = f'Removed'
            deleted_plotted = True

        axes[1].plot(contour[:, 0], contour[:, 1], color=color, linewidth=lw, label=label)
        axes[1].text(*bacteria['center'], str(idx), color='yellow', fontsize=6,
                    ha='center', va='center', fontweight='bold')

    n_total = len(merged_bacteria_list)
    n_kept = len(bacteria_list_filtered)
    axes[1].set_title(
        f'Shape Filter: {n_kept}/{n_total} kept, {n_total - n_kept} removed (occluded)',
        fontsize=12
    )
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].axis('off')

    plt.savefig(output_path)
    plt.close()

def add_texture(bacteria_list, image):
    """
    Calculate surface texture (roughness) for all bacteria using polynomial surface fitting.
    Adds 'texture' key to each bacteria dict.
    """
    # Normalize image once
    if image.dtype == np.uint8:
        img = image.astype(np.float64) / 255.0
    else:
        img = image.astype(np.float64)
        if img.max() > 1:
            img = img / img.max()
    
    for b in bacteria_list:
        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [b['contour']], -1, 1, thickness=-1)
        
        # Erode to avoid edges
        kernel = np.ones((7, 7), np.uint8)
        inner_mask = cv2.erode(mask, kernel, iterations=2)
        
        # Get bounding box
        ys, xs = np.where(mask == 1)
        if len(ys) == 0:
            b['texture'] = 0.0
            continue
        
        pad = 5
        y1, y2 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
        x1, x2 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
        
        patch = img[y1:y2, x1:x2]
        mask_p = inner_mask[y1:y2, x1:x2]

        valid = mask_p > 0
        n_valid = np.sum(valid)
        if n_valid < 50:
            b['texture'] = 0.0
            continue

        # Edge-preserving smoothing: removes pixel-level noise but preserves
        # real membrane ridges/bumps (spatially coherent structures)
        patch_smooth = cv2.bilateralFilter(
            patch.astype(np.float32), d=5, sigmaColor=0.08, sigmaSpace=9
        ).astype(np.float64)

        # Get coordinates and values
        yy, xx = np.where(valid)
        values = patch_smooth[valid]
        
        h, w = patch.shape
        xx_norm = (xx - w / 2) / (w / 2)
        yy_norm = (yy - h / 2) / (h / 2)
        
        # Fit 2nd order polynomial: z = a + bx + cy + dx² + ey² + fxy
        A = np.column_stack([
            np.ones(n_valid), xx_norm, yy_norm,
            xx_norm**2, yy_norm**2, xx_norm * yy_norm
        ])
        
        coeffs, _, _, _ = np.linalg.lstsq(A, values, rcond=None)
        fitted = A @ coeffs
        residual = values - fitted
        
        b['texture'] = np.sqrt(np.mean(residual**2))
    
    return bacteria_list


def process_image(image_path):
    original_image = np.array(Image.open(image_path).convert("L"))

    height, width = original_image.shape

    image = original_image[:height - 70, :]

    masks = run_segmentation(image)

    bacteria_list = extract_bacteria_features(masks)

    # # path = ./data/{treatment}/20000/{sample}.tif
    treatment = image_path.parent.parent.name   # e.g. "control", "6.25 ug/ml"
    sample    = image_path.stem                 # e.g. "Sample 7_07"

    bacteria_list_filtered = filter_bacteria_by_shape(
        bacteria_list,
        min_ellipse_fit=0.8,
    )  

    out_path_filter = OUTPUT_DIR / 'filter' / treatment
    out_path_filter.mkdir(parents=True, exist_ok=True)

    output_path = out_path_filter / f"{sample}.png"

    visualize_filter(image, bacteria_list_filtered, bacteria_list, output_path)

    bacteria_list_texture = add_texture(bacteria_list_filtered, image)

    return bacteria_list_texture


def main():

    paths = [
        p
        for s in TREATMENTS
        for p in (INPUT_DIR / s / "20000").glob("*.tif")
        if (INPUT_DIR / s / "20000").exists()
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in paths:
        treatment = path.parent.parent.name
        sample    = path.stem
        print(f"Working on {treatment} / {sample} ...")
        bacterial_list = process_image(path)

        out_path_csv = OUTPUT_DIR / 'features' / treatment
        out_path_csv.mkdir(parents=True, exist_ok=True)
        csv_path = out_path_csv / f"{sample}.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'bacteria_id', 'center', 'major_radius', 'minor_radius',
                'area', 'aspect_ratio', 'texture',
            ])
            writer.writeheader()
            for b in bacterial_list:
                writer.writerow({
                    'bacteria_id':  b['bacteria_id'],
                    'center':       b['center'],
                    'major_radius': b['major_radius'],
                    'minor_radius': b['minor_radius'],
                    'area':         b['area'],
                    'aspect_ratio': b['aspect_ratio'],
                    'texture':      b['texture'],
                })
    

if __name__ == "__main__":
    main()