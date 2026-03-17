# SEM Bacteria Image Processing and Segmentation

## Overview

This project focuses on the detection and analysis of bacterial structures in Scanning Electron Microscopy (SEM) images using a combination of classical image processing techniques and pretrained AI models. The approach enables segmentation and quantitative characterization of morphological features such as structure, texture, and area.

---

## Segmentation Using Cellpose

Segmentation is performed using the pretrained **Cellpose** model.

### Model Selection

* **Model:** `cyto3`
* The *cyto3* model is Cellpose’s most general and robust pretrained model, trained on a highly diverse dataset of cell types, shapes, and microscopy modalities.
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

## Image Processing and Analysis

Following segmentation, classical image processing techniques are applied to refine results and extract meaningful features:

* Noise reduction and mask refinement
* Boundary correction and object separation
* Extraction of morphological features:

  * Area and size distribution
  * Structural characteristics
  * Texture and surface patterns
  * Spatial organization

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
