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
TREATMENTS = ['0.1 Eth', '6.25 ug:ml', '12.5 ug:ml', '25 ug:ml', '50 ug:ml', 'control']


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
            'circularity':         (4 * np.pi * prop.area / perimeter ** 2) if perimeter > 0 else 0.0,
            'ellipse_fit_score':   compute_ellipse_fit_score(contour, mask_shape),
            'touches_border':      is_touching_border(contour, mask_shape),
        })

    # ── report dropped ────────────────────────────────────────────
    if dropped:
        print(f"Dropped {len(dropped)} regions:")
        for label, centroid, reason in dropped:
            print(f"  label={label:3d}  center=({int(centroid[1])}, {int(centroid[0])})  reason: {reason}")

    return bacteria_list


# ── Ellipse fit quality ────────────────────────────────────────────────────────

def merged_ellipse_fit_score(b1: dict, b2: dict) -> float:
    p1 = b1['contour'].reshape(-1, 2)
    p2 = b2['contour'].reshape(-1, 2)
    merged_pts = np.vstack([p1, p2]).astype(np.float32)

    if len(merged_pts) < 5:
        return 0.0

    ellipse = cv2.fitEllipse(merged_pts)
    (cx, cy), (ma, mb), angle = ellipse

    pad = 10
    x_min = int(np.floor(merged_pts[:, 0].min())) - pad
    y_min = int(np.floor(merged_pts[:, 1].min())) - pad
    x_max = int(np.ceil(merged_pts[:, 0].max())) + pad
    y_max = int(np.ceil(merged_pts[:, 1].max())) + pad
    canvas_w = x_max - x_min + 1
    canvas_h = y_max - y_min + 1

    merged_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for b in (b1, b2):
        c_shifted = b['contour'].reshape(-1, 1, 2).copy()
        c_shifted[:, :, 0] -= x_min
        c_shifted[:, :, 1] -= y_min
        cv2.drawContours(merged_mask, [c_shifted.astype(np.int32)], -1, 1, thickness=cv2.FILLED)

    ellipse_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    shifted_ellipse = ((cx - x_min, cy - y_min), (ma, mb), angle)
    cv2.ellipse(ellipse_mask, shifted_ellipse, 1, thickness=cv2.FILLED)

    intersection = np.logical_and(merged_mask, ellipse_mask).sum()
    union = np.logical_or(merged_mask, ellipse_mask).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


# ── Shared boundary ────────────────────────────────────────────────────────────

def get_shared_boundary(b1: dict, b2: dict, proximity: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    p1 = b1['contour'].reshape(-1, 2).astype(float)
    p2 = b2['contour'].reshape(-1, 2).astype(float)
    d1, _ = cKDTree(p2).query(p1)
    d2, _ = cKDTree(p1).query(p2)
    return p1[d1 < proximity], p2[d2 < proximity]


def get_shared_line_tips(contour: np.ndarray, shared_mask: np.ndarray) -> list[int]:
    """
    Returns the 2 tip indices of the shared boundary zone:
      - first shared point entering the zone
      - last shared point leaving the zone
    """
    n = len(shared_mask)
    tips = []
    for i in range(n):
        prev = (i - 1) % n
        if shared_mask[i] and not shared_mask[prev]:
            tips.append(i)
        elif not shared_mask[i] and shared_mask[prev]:
            tips.append(prev)
    return tips


# ── Outer tangent from tip ─────────────────────────────────────────────────────

def outer_tangent_from_tip(contour: np.ndarray, tip_idx: int,
                            shared_mask: np.ndarray,
                            arc_length: float = 10.0) -> np.ndarray | None:
    """
    Origin = tip_idx coordinate (the junction point itself).
    Walk along the contour from tip_idx in both directions, skipping shared
    points, accumulating arc_length of outer contour.
    Return the unit vector from the tip toward the end of the longer outer arc.

    This keeps the origin pinned to the junction regardless of contour sparsity.
    """
    n = len(contour)
    tip = contour[tip_idx]

    def walk(step: int):
        outer_pts = []
        arc = 0.0
        prev = tip.copy()
        for k in range(1, n):
            idx_k = (tip_idx + step * k) % n
            pt = contour[idx_k]
            seg = np.linalg.norm(pt - prev)
            prev = pt
            if shared_mask[idx_k]:
                # still accumulate arc to not get stuck, but don't record direction
                arc += seg
                if arc >= arc_length:
                    break
                continue
            arc += seg
            outer_pts.append(pt)
            if arc >= arc_length:
                break
        return outer_pts

    pts_fwd = walk(+1)
    pts_bwd = walk(-1)

    # Pick direction with more outer points collected
    pts = pts_fwd if len(pts_fwd) >= len(pts_bwd) else pts_bwd
    if not pts:
        return None

    vec = pts[-1] - tip
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        return None
    return vec / norm


# ── Junction angle check ───────────────────────────────────────────────────────

def get_junction_angles(b1: dict, b2: dict, proximity: float = 3.0,
                        arc_length: float = 15.0) -> tuple[float, float] | None:
    """
    For each of the 2 shared-line tips:
      - Origin = the tip coordinate itself (always at the junction).
      - Direction = walk arc_length along the outer contour from the tip,
        return unit vector from tip to the end of the walk.
      - Angle = between b1's and b2's direction vectors at this junction.
    ~180° means a straight line through the junction → likely a split cell.
    """
    p1 = b1['contour'].reshape(-1, 2).astype(float)
    p2 = b2['contour'].reshape(-1, 2).astype(float)

    mask1 = cKDTree(p2).query(p1)[0] < proximity
    mask2 = cKDTree(p1).query(p2)[0] < proximity

    if mask1.sum() < 3 or mask2.sum() < 3:
        return None

    tips1 = get_shared_line_tips(p1, mask1)
    tips2 = get_shared_line_tips(p2, mask2)
    if len(tips1) < 2 or len(tips2) < 2:
        return None

    angles = []
    for tip_idx1 in tips1[:2]:
        tip_coord = p1[tip_idx1]

        # For p2: use the tip on p2 closest to this junction coordinate
        tip_dists_p2 = [np.linalg.norm(p2[t] - tip_coord) for t in tips2[:2]]
        tip_idx2 = tips2[np.argmin(tip_dists_p2)]

        t1 = outer_tangent_from_tip(p1, tip_idx1, mask1, arc_length)
        t2 = outer_tangent_from_tip(p2, tip_idx2, mask2, arc_length)

        if t1 is None or t2 is None:
            return None

        cos_a = np.clip(np.dot(t1, t2), -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos_a))))

    return float(angles[0]), float(angles[1])


# ── Intensity similarity across shared line ────────────────────────────────────

def shared_line_intensity_diff(image: np.ndarray, shared_pts: np.ndarray,
                                b1: dict, b2: dict, sample_width: int = 3) -> float:
    if len(shared_pts) < 2:
        return 1.0

    h, w = image.shape[:2]
    max_val = 255.0 if image.dtype == np.uint8 else float(image.max()) or 1.0
    c1, c2 = np.array(b1['center'], float), np.array(b2['center'], float)
    s1, s2 = [], []

    for pt in shared_pts.astype(float):
        for center, bucket in ((c1, s1), (c2, s2)):
            d = center - pt
            d /= np.linalg.norm(d) + 1e-6
            for step in range(1, sample_width + 1):
                col, row = int(round((pt + d * step)[0])), int(round((pt + d * step)[1]))
                if 0 <= row < h and 0 <= col < w:
                    bucket.append(float(image[row, col]))

    if not s1 or not s2:
        return 1.0
    return abs(np.mean(s1) - np.mean(s2)) / max_val


# ── Candidate pair generation ─────────────────

def _candidate_pairs(bacteria_list: list[dict], proximity: float) -> list[tuple[int, int]]:
    centers = np.array([b['center'] for b in bacteria_list], dtype=float)
    max_r = max(b['major_radius_pixels'] for b in bacteria_list)
    search_r = 2 * max_r + proximity
    tree = cKDTree(centers)

    pairs = []
    for i, b in enumerate(bacteria_list):
        candidates = tree.query_ball_point(b['center'], search_r)
        p_i = b['contour'].reshape(-1, 2).astype(float)
        for j in candidates:
            if j <= i:
                continue
            p_j = bacteria_list[j]['contour'].reshape(-1, 2).astype(float)
            if float(cKDTree(p_j).query(p_i)[0].min()) < proximity:
                pairs.append((i, j))
    return pairs


# ── Main: find split pairs ─────────────────────────────────────────────────────

def find_split_pairs(bacteria_list: list[dict], image: np.ndarray,
                     max_intensity_diff: float = 0.1,
                     min_junction_angle: float = 130.0,
                     min_ellipse_fit: float = 0.8,
                     proximity: float = 3.0) -> list[tuple[int, int]]:
    split_pairs, paired = [], set()
    candidates = _candidate_pairs(bacteria_list, proximity)

    for i, j in candidates:
        if i in paired or j in paired:
            continue
        b1, b2 = bacteria_list[i], bacteria_list[j]

        shared_b1, shared_b2 = get_shared_boundary(b1, b2, proximity)
        shared_pts = shared_b1 if len(shared_b1) >= len(shared_b2) else shared_b2
        if shared_line_intensity_diff(image, shared_pts, b1, b2) > max_intensity_diff:
            continue

        angles = get_junction_angles(b1, b2, proximity)
        if angles is None or min(angles) < min_junction_angle:
            continue

        if merged_ellipse_fit_score(b1, b2) < min_ellipse_fit:
            continue

        split_pairs.append((i, j))
        paired |= {i, j}

    return split_pairs


def visualize_post_process(bacteria_list, split_pairs, image, output_path):
    # Visualise split pairs — each pair gets a unique colour, unpaired = grey
    pair_colors = plt.cm.tab10(np.linspace(0, 1, max(len(split_pairs), 1)))
    is_split = {i: False for i in range(len(bacteria_list))}
    bacteria_color = {i: 'silver' for i in range(len(bacteria_list))}

    for idx, (i, j) in enumerate(split_pairs):
        bacteria_color[i] = pair_colors[idx % len(pair_colors)]
        bacteria_color[j] = pair_colors[idx % len(pair_colors)]
        is_split[i] = True
        is_split[j] = True

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(24, 8))

    # ax0: original image
    ax0.imshow(image, cmap='gray')
    ax0.set_title('Original Image', fontsize=13)
    ax0.axis('off')

    # ax1: split pair detection
    ax1.imshow(image, cmap='gray')

    for i, b in enumerate(bacteria_list):
        contour = b['contour'].squeeze()
        color = bacteria_color[i]
        lw = 2.5 if is_split[i] else 1.0
        ax1.plot(contour[:, 0], contour[:, 1], color=color, linewidth=lw)
        cx, cy = b['center']
        ax1.text(cx, cy, str(i), color='white', fontsize=7, ha='center', va='center',
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', fc=color, ec='none', alpha=0.7))

    for idx, (i, j) in enumerate(split_pairs):
        c1 = bacteria_list[i]['center']
        c2 = bacteria_list[j]['center']
        color = pair_colors[idx % len(pair_colors)]
        ax1.plot([c1[0], c2[0]], [c1[1], c2[1]], color=color, linewidth=1.5, linestyle='--')

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='silver', lw=1.5, label='Normal'),
        Line2D([0], [0], color=pair_colors[0], lw=2.5, label='Split pair (same colour = same cell)'),
        Line2D([0], [0], color='gray', lw=1.5, linestyle='--', label='Pair connection'),
    ]
    ax1.legend(handles=handles, loc='upper right', fontsize=9)
    ax1.set_title(f'Split Bacteria Detection — {len(split_pairs)} pairs found', fontsize=13, fontweight='bold')
    ax1.axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def merge_split_pairs(bacteria_list: list[dict], split_pairs: list[tuple[int, int]],
                      image: np.ndarray, proximity: float = 3.0) -> list[dict]:
    """
    Returns a new bacteria list where each split pair is merged into one entry.
    Unpaired bacteria are passed through unchanged.
    All features are recomputed for merged bacteria.
    """
    # Track which indices get merged
    merged_into = {}  # idx → partner idx
    for i, j in split_pairs:
        merged_into[i] = j
        merged_into[j] = i

    SCALE_BAR_UM = 2.0
    SCALE_BAR_PX = 188
    ratio = SCALE_BAR_UM / SCALE_BAR_PX

    new_list = []
    visited = set()

    for i, b in enumerate(bacteria_list):
        if i in visited:
            continue

        if i not in merged_into:
            # Not part of a split pair — pass through as-is
            new_list.append(b)
            continue

        # Merge this pair
        j = merged_into[i]
        visited.add(i)
        visited.add(j)
        b2 = bacteria_list[j]

        # Build merged mask and extract merged contour
        p1 = b['contour'].reshape(-1, 2).astype(float)
        p2 = b2['contour'].reshape(-1, 2).astype(float)

        # Outer points only (remove shared boundary)
        d1, _ = cKDTree(p2).query(p1)
        d2, _ = cKDTree(p1).query(p2)
        outer1 = p1[d1 >= proximity]
        outer2 = p2[d2 >= proximity]

        if len(outer1) < 3 or len(outer2) < 3:
            # Fallback: just keep the larger one
            new_list.append(b if b['area_pixels'] >= b2['area_pixels'] else b2)
            continue

        merged_pts = np.vstack([outer1, outer2])
        centroid = merged_pts.mean(axis=0)
        order = np.argsort(np.arctan2(merged_pts[:, 1] - centroid[1],
                                       merged_pts[:, 0] - centroid[0]))
        merged_contour = merged_pts[order].astype(np.int32).reshape(-1, 1, 2)

        # Recompute features on merged contour
        perimeter = cv2.arcLength(merged_contour, True)
        area_px   = cv2.contourArea(merged_contour)

        if len(merged_contour) >= 5:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(merged_contour)
            major_px = max(MA, ma) / 2
            minor_px = min(MA, ma) / 2
            aspect   = major_px / minor_px if minor_px > 0 else float('inf')
        else:
            major_px = minor_px = aspect = 0.0

        circularity = (4 * np.pi * area_px / perimeter ** 2) if perimeter > 0 else 0.0

        # Ellipse fit score (IoU)
        mask_shape = image.shape[:2]
        ellipse_mask  = np.zeros(mask_shape, np.uint8)
        actual_mask   = np.zeros(mask_shape, np.uint8)
        if len(merged_contour) >= 5:
            cv2.ellipse(ellipse_mask, cv2.fitEllipse(merged_contour), 1, -1)
        cv2.drawContours(actual_mask, [merged_contour], -1, 1, -1)
        inter = np.logical_and(actual_mask, ellipse_mask).sum()
        union = np.logical_or(actual_mask,  ellipse_mask).sum()
        ellipse_fit = float(inter / union) if union > 0 else 0.0

        touches = is_touching_border(merged_contour, mask_shape)

        # Center of mass
        M = cv2.moments(merged_contour)
        if M['m00'] > 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            center = (int(centroid[0]), int(centroid[1]))

        new_list.append({
            'bacteria_id':         b['bacteria_id'],   # keep lower id
            'contour':             merged_contour,
            'center':              center,
            'major_radius_pixels': major_px,
            'minor_radius_pixels': minor_px,
            'major_radius':        major_px * ratio,
            'minor_radius':        minor_px * ratio,
            'area_pixels':         area_px,
            'area':                area_px * ratio ** 2,
            'aspect_ratio':        aspect,
            'circularity':         circularity,
            'ellipse_fit_score':   ellipse_fit,
            'touches_border':      touches,
        })

    return new_list


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
    kept, removed = [], []
    for b in bacteria_list:
        if b['area_pixels'] > min_area and b['ellipse_fit_score'] >= min_ellipse_fit and not b['touches_border']:
            kept.append(b)
        else:
            removed.append(b)

    stats = {
        'total': len(bacteria_list),
        'kept': len(kept),
        'removed': len(removed),
        'removed_occluded': sum(1 for b in removed if not b['touches_border']),
        'removed_border': sum(1 for b in removed if b['touches_border']),
    }
    return kept, removed, stats


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
            efs = bacteria['ellipse_fit_score']
            circ = bacteria['circularity']
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
            patch.astype(np.float32), d=9, sigmaColor=0.08, sigmaSpace=9
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

    split_pairs = find_split_pairs(bacteria_list, image)

    # path = ./data/{treatment}/20000/{sample}.tif
    treatment = image_path.parent.parent.name   # e.g. "control", "0.1 Eth"
    sample    = image_path.stem                 # e.g. "Sample 7_07"

    out_path_pp = OUTPUT_DIR / 'post process' / treatment
    out_path_pp.mkdir(parents=True, exist_ok=True)

    output_path = out_path_pp / f"{sample}.png"

    visualize_post_process(bacteria_list, split_pairs, image, output_path)

    merged_bacteria_list = merge_split_pairs(bacteria_list, split_pairs, image)

    bacteria_list_filtered, removed_bacteria, filter_stats = filter_bacteria_by_shape(
        merged_bacteria_list,
        min_ellipse_fit=0.8,
    )  

    out_path_filter = OUTPUT_DIR / 'filter' / treatment
    out_path_filter.mkdir(parents=True, exist_ok=True)

    output_path = out_path_filter / f"{sample}.png"

    visualize_filter(image, bacteria_list_filtered, merged_bacteria_list, output_path)

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