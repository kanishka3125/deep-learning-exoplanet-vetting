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

---

## Dataset

The project uses the Kepler exoplanet candidate-vetting dataset distributed with the AstroNet project.

The dataset contains precomputed phase-folded representations:

- **Global view:** 2001 data points
- **Local view:** 201 data points

The original labels are:

| Original Label | Meaning | Binary Label |
|---|---|---:|
| PC | Planet Candidate | 1 |
| AFP | Astrophysical False Positive | 0 |
| NTP | Non-Transit Phenomenon | 0 |

### Dataset Split

| Split | Samples |
|---|---:|
| Training | 12,589 |
| Validation | 1,574 |
| Test | 1,574 |

Training set composition:

- Planet Candidates (PC): 2,885
- Astrophysical False Positives (AFP): 7,643
- Non-Transit Phenomena (NTP): 2,061

> **Important:** The project uses the already-generated phase-folded global and local representations provided by the AstroNet dataset. Raw Kepler FITS light curves were not independently phase-folded in the project notebook.

---

## Models

### 1. Baseline 1D CNN

The baseline model uses only the global light-curve representation.

Architecture:

```text
Global View (2001 × 1)
        ↓
Conv1D (16)
        ↓
MaxPooling
        ↓
Conv1D (32)
        ↓
MaxPooling
        ↓
Conv1D (64)
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense (64)
        ↓
Sigmoid
        ↓
Planet Probability

Total parameters: 1,012,545

2. AstroNet-Style Dual-Input CNN

The main model processes both global and local light-curve views.

                 ┌── Global View (2001) ── CNN ──┐
                 │                                 │
Input ───────────┤                                 ├── Concatenate
                 │                                 │
                 └── Local View (201) ─── CNN ────┘
                                                   ↓
                                             Dense (512)
                                                   ↓
                                             Dropout (0.5)
                                                   ↓
                                             Sigmoid
                                                   ↓
                                          Planet Probability

Total parameters: 8,135,361

The architecture is described as AstroNet-style because it adopts the global/local dual-view concept. It is a simplified adaptation and is not an exact reproduction of the original AstroNet implementation.

Training

The models were trained using TensorFlow/Keras with the following configuration:

Parameter	Value
Optimizer	Adam
Learning Rate	0.001
Loss	Binary Cross-Entropy
Batch Size	64
Epochs	10
Dropout	0.5
Metrics	Accuracy, Precision, Recall, AUC
Hardware	Google Colab GPU

Class weights were used to account for class imbalance:

Class 0: 0.6487
Class 1: 2.1818
Evaluation

The final AstroNet-style model was evaluated on a held-out test set.

Test Performance
Metric	Result
Accuracy	94.41%
Precision	81.49%
Recall	97.78%
F1 Score	88.89%
ROC-AUC	98.65%
Confusion Matrix
                 Predicted
               Non-Planet  Planet
Actual
Non-Planet       1134       80
Planet              8      352

The model produced:

True Negatives: 1134
False Positives: 80
False Negatives: 8
True Positives: 352
Uncertainty Estimation

Monte Carlo Dropout was used to estimate predictive uncertainty.

Each test example was evaluated 50 times with dropout enabled. The mean prediction and standard deviation across these stochastic forward passes were used as measures of predictive probability and uncertainty.

Results:

Mean uncertainty: 0.02048
Median uncertainty: 0.00375
Maximum uncertainty: 0.15271

The most uncertain test example was Test Index 592:

Property	Value
True Label	0
Mean Probability	0.45896
Uncertainty	0.15271
Prediction	0
Confidence	0.04104
Explainability

1D Grad-CAM was used to identify regions of the light curves that contributed strongly to the model's prediction.

For Test Index 592, the Grad-CAM analysis showed strong activation around the central transit-like feature in the local representation.

Grad-CAM provides an attribution-based explanation and should not be interpreted as proof of the physical cause of the signal.

Calibration

The probabilistic predictions were evaluated using a reliability curve and Brier score.

Brier Score: 0.0441

The calibration analysis showed some deviation from ideal calibration, with evidence of overconfidence in parts of the intermediate-to-high probability range.

Case Studies
Test Index 592 — Most Uncertain Example

The model assigned a mean probability of approximately 0.4590 with an uncertainty of 0.1527. The prediction was close to the 0.5 decision threshold, making it a useful example of an ambiguous candidate.

Test Index 1343 — Difficult Positive

The true label was 1, with a mean Monte Carlo probability of approximately 0.6879 and uncertainty of 0.1497.

Test Index 587 — False Positive

The true label was 0, but the model assigned a mean probability of approximately 0.7307, resulting in a false-positive prediction. This demonstrates the importance of uncertainty estimation and additional candidate vetting.

Project Structure
deep-learning-exoplanet-vetting/
│
├── notebooks/
│   └── Deep_Learning_Exoplanet_Vetting_AstroNet.ipynb
│
├── figures/
│   ├── baseline_roc_curve.png
│   ├── baseline_confusion_matrix.png
│   ├── astronet_roc_curve.png
│   ├── astronet_confusion_matrix.png
│   ├── global_and_local_light_curves.png
│   ├── uncertainty_distribution.png
│   ├── confidence_vs_uncertainty.png
│   ├── most_uncertain_global_view.png
│   ├── most_uncertain_local_view.png
│   ├── mc_dropout_predictions.png
│   ├── gradcam_local_test_592.png
│   └── astronet_calibration_curve.png
│
├── results/
│   ├── model_comparison.csv
│   └── case_studies.csv
│
├── .gitignore
├── requirements.txt
└── README.md

Trained .keras models, TFRecord datasets, and large intermediate files are excluded from the repository because of their size.

Technologies
Python
TensorFlow / Keras
NumPy
Pandas
Scikit-learn
Matplotlib
Google Colab
Kepler/AstroNet TFRecord data
Reproducibility

To reproduce the experiment:

Open the notebook in notebooks/.
Set up a TensorFlow environment.
Obtain the required AstroNet Kepler TFRecord dataset.
Update the dataset path in the notebook.
Run the notebook sequentially.
Train the baseline and AstroNet-style models.
Evaluate the models on the held-out test set.
Run MC Dropout uncertainty estimation.
Generate Grad-CAM explanations.
Perform calibration analysis.

The repository intentionally does not include the original dataset or trained model files.

Limitations
The experiment uses the available Kepler/AstroNet dataset and may not generalize directly to other missions or stellar populations.
The dataset contains class imbalance.
MC Dropout provides an approximate measure of predictive uncertainty rather than a complete Bayesian posterior.
Grad-CAM provides model attribution rather than causal explanation.
The model classifies candidates based on learned patterns and does not independently confirm exoplanets.
Performance on TESS or other surveys requires additional validation.
Future Work

Possible extensions include:

Applying the pipeline to TESS data
Training on larger and more diverse datasets
Exploring deep ensembles or Bayesian neural networks for improved uncertainty estimation
Investigating additional explainability methods
Extending the task to multiclass classification
Automating candidate period detection
Developing human-in-the-loop candidate vetting systems
References
Shallue, C. J., & Vanderburg, A. (2018). Identifying Exoplanets with Deep Learning: A Five-Planet Resonant Chain around Kepler-80 and an Eighth Planet around Kepler-90. The Astronomical Journal, 155(2), 94.
Koch, D. G., et al. (2010). Kepler Mission Design, Realized Photometric Performance, and Early Science. The Astrophysical Journal Letters, 713(2), L79–L86.
Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning. Proceedings of ICML, 48, 1050–1059.
Selvaraju, R. R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization. Proceedings of ICCV, 618–626.
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. Proceedings of ICML, 70, 1321–1330
