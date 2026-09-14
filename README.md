# Deep Learning Exoplanet Vetting from Time-Series Photometry

> An end-to-end deep learning pipeline for vetting exoplanet candidates from Kepler time-series photometry using an AstroNet-style dual-input CNN, uncertainty estimation, explainability, and calibration analysis.

---

## Overview

Identifying genuine exoplanet candidates from astronomical observations is a challenging classification problem. Transit-like signals in stellar light curves can be caused by planets as well as several types of false positives.

This project explores deep learning for automated exoplanet vetting using Kepler photometric data.

The project implements and compares:

- A baseline single-input 1D CNN
- An AstroNet-style dual-input CNN using global and local light-curve views
- Monte Carlo Dropout for predictive uncertainty estimation
- 1D Grad-CAM for model explainability
- Calibration analysis using a reliability curve and Brier score

The goal is not only to classify candidates, but also to investigate **how confident the model is and which regions of the light curve influence its predictions**.

---

## Key Results

The AstroNet-style dual-input CNN outperformed the global-view baseline across all major evaluation metrics.

| Metric | Baseline 1D CNN | AstroNet-style CNN |
|---|---:|---:|
| Accuracy | 91.61% | **94.41%** |
| Precision | 74.46% | **81.49%** |
| Recall | 96.39% | **97.78%** |
| F1 Score | 84.02% | **88.89%** |
| ROC-AUC | 97.81% | **98.65%** |

### Improvement

- Accuracy: **+2.80 percentage points**
- Precision: **+7.03 percentage points**
- Recall: **+1.39 percentage points**
- F1 Score: **+4.87 percentage points**
- ROC-AUC: **+0.84 percentage points**

The AstroNet-style model reduced false positives from **119 to 80** and false negatives from **13 to 8** on the test set.

---

## Project Pipeline

```text
Kepler Time-Series Photometry
              │
              ▼
      Global + Local Views
              │
              ▼
   AstroNet-style Dual CNN
              │
              ▼
     Planet Probability
              │
        ┌─────┴─────┐
        ▼           ▼
 MC Dropout      Grad-CAM
 Uncertainty    Explainability
        │           │
        └─────┬─────┘
              ▼
       Calibration Analysis
