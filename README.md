# SEM Bacteria Image Processing and Segmentation

## Overview

This project focuses on the detection and analysis of bacterial structures in Scanning Electron Microscopy (SEM) images. It contains two main scripts, each targeting a different aspect of the analysis:

* **`bacteria_features.py`** — Segments individual bacteria using Cellpose and extracts per-cell morphological features.
* **`biofilm_stracture.py`** — Analyzes the biofilm matrix structure using edge detection and tile-based texture analysis, without relying on cell segmentation.

---

## Script 1: `bacteria_features.py` — Bacterial Segmentation & Feature Extraction

This script runs a full per-bacterium analysis pipeline on SEM images:

1. **Segmentation** — Detects individual bacteria using the Cellpose `cyto3` model.
2. **Feature extraction** — Computes morphological features for each detected cell:
   * Major and minor radius (in µm)
   * Area (in µm²)
   * Aspect ratio
   * Circularity
   * Ellipse fit score (IoU between actual shape and fitted ellipse)
3. **Split-pair detection** — Identifies pairs of bacteria that were incorrectly split by the segmenter (sharing a boundary, forming a straight junction, and fitting a single ellipse when merged), and merges them into one cell.
4. **Shape filtering** — Removes bacteria that are cut off by the image border or heavily occluded by neighbors (low ellipse fit score).
5. **Texture measurement** — Computes a noise-corrected coefficient of variation (CV) for each bacterium's internal surface texture using the inner pixel region.
6. **Output** — Saves a CSV per image under `results/feature_extraction/features/{treatment}/` with columns: `bacteria_id`, `center`, `major_radius`, `minor_radius`, `area`, `aspect_ratio`, `texture`.

---

## Script 2: `biofilm_stracture.py` — Biofilm Matrix Structure Analysis

This script characterizes the spatial texture of the biofilm matrix across the image — independent of individual cell segmentation:

1. **Background detection** — Uses tile-based variance analysis to distinguish true empty substrate (background) from biofilm-covered regions. Only biofilm-covered areas are analyzed.
2. **Edge detection** — Applies morphological opening, weighted least-squares smoothing, and Sobel filtering to extract structural edges. Small noise artifacts are removed by connected-component filtering.
3. **Tile-based texture analysis** — Slides a 100×100 px window across the biofilm region and computes the standard deviation of edge intensities per tile, producing a texture score map.
4. **Heatmap visualization** — Generates a 4-panel figure per image: original, edge map, smoothed texture heatmap overlaid on edges, and a histogram of tile texture scores.
5. **Output** — Saves PNG visualizations per image and CSV summaries including bacteria coverage (%), mean texture, and per-tile values under `results/matrix/`.

---

## Segmentation Using Cellpose

Segmentation (used in `bacteria_features.py`) is performed using the pretrained **Cellpose** model.

### Model Selection

* **Model:** `cyto3`
* The *cyto3* model is Cellpose's most general and robust pretrained model, trained on a highly diverse dataset of cell types, shapes, and microscopy modalities.
* It is well-suited for bacterial detection, as bacteria are typically small, dense, and exhibit rod- or coccus-like shapes that do not perfectly match specialized models.

---

## Parameter Selection

The following parameters were chosen to optimize segmentation performance for SEM bacterial images:

* **FLOW_THRESHOLD (0.4)**
  Controls the consistency required for predicted flow vectors to define an object.
  A relatively permissive value allows detection of structures with less-defined or noisy boundaries.

* **CELLPROB_THRESHOLD (-2.0)**
  Defines the minimum confidence score for a pixel to be classified as part of a cell.
  A lower threshold enables detection of dim or low-contrast bacterial structures that might otherwise be missed.

* **diameter = None**
  Allows the model to automatically estimate object size.
  This is useful when working with varying magnifications, where predefined diameter values may lead to incorrect scaling.

---

## Environment & Dependencies

* **Python:** 3.10.20

### Dependencies

This project uses Python 3.10.20 and relies mainly on:

- Cellpose
- NumPy
- Matplotlib
- Pillow
- scikit-image
- OpenCV
- SciPy
- Pandas
- Seaborn

For the full environment and exact versions, see `requirements.txt`.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Goal

To build a robust computational pipeline that combines AI-based segmentation with image processing for accurate and interpretable analysis of bacterial morphology in SEM images.

---

## Notes

This project was developed as part of an M.Sc. research project.
