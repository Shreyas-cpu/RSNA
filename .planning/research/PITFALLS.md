# Pitfalls Research

**Domain:** Medical Imaging / Deep Learning (RSNA Knee Abnormality Detection)  
**Researched:** 2026-10-06  
**Confidence:** HIGH  

## Critical Pitfalls

### Pitfall 1: Data Leakage Across MRI Slices/Series in Cross-Validation

**What goes wrong:**
Validation scores appear falsely optimistic (e.g. 0.85+ CV), but the Public Leaderboard drops to ~0.50.

**Why it happens:**
Multiple MRI series or slices from the same patient/exam are split across training and validation folds. The model memorizes patient-specific anatomical quirks rather than learning generalized pathological biomarkers.

**How to avoid:**
Use `StratifiedGroupKFold` grouped strictly by `StudyInstanceUID` (or Patient ID if provided in DICOM tags). Slices and series from any given study must exist entirely in train or entirely in validation.

**Warning signs:**
Huge discrepancy between local validation AUC and Kaggle public leaderboard score (> 0.05 gap).

**Phase to address:**
Phase 1 (Validation Setup & Data Ingestion).

---

### Pitfall 2: Slow DICOM I/O Triggering Kaggle Notebook Timeout (9h limit)

**What goes wrong:**
Inference notebook runs out of time on Kaggle's hidden test set, resulting in "Notebook Timeout" and no submission score.

**Why it happens:**
The hidden test set contains thousands of DICOM files. Standard `pydicom.dcmread` parses headers and decodes pixel data in single-threaded pure Python, taking 100–300ms per slice, which explodes to >9 hours for full evaluation.

**How to avoid:**
Use `dicomsdl` or `pylibjpeg` compiled C++ decompressors, parse only necessary tags, read slices with `concurrent.futures.ThreadPoolExecutor`, and only process selected informative slices/slabs rather than every empty slice.

**Warning signs:**
Processing a sample study takes > 5 seconds in the inference loop.

**Phase to address:**
Phase 1 (Preprocessing Engine) and Phase 6 (Inference Kernel).

---

### Pitfall 3: Severe Class Imbalance Distorting Macro ROC AUC

**What goes wrong:**
The model performs well on frequent targets (`Effusion`, `Medial Meniscus`) but gets an AUC of 0.50 (random guessing) on rare targets (`Fracture`, `Contusion`), dragging down the unweighted macro average.

**Why it happens:**
Standard binary cross-entropy treats all samples equally. Positive examples for rare pathologies represent < 5% of cases, so gradients are dominated by negative samples.

**How to avoid:**
Use positive-class weighting in `BCEWithLogitsLoss(pos_weight=...)`, focal loss, or asymmetric loss. Calibrate prediction probabilities post-hoc per class.

**Warning signs:**
Looking only at average loss or micro-AUC instead of inspecting individual per-class AUC scores.

**Phase to address:**
Phase 3 (Model Architecture & Loss Formulation).

---

### Pitfall 4: NaNs or Out-of-Bounds Probabilities in `submission.csv`

**What goes wrong:**
Kaggle rejects the submission with "Submission Scoring Error" or score drops to 0.0.

**Why it happens:**
Missing DICOM series, corrupt files, or numerical instability in sigmoid/logits producing `NaN` or values outside $[0.0, 1.0]$.

**How to avoid:**
Implement an automated post-inference sanitation step:
```python
sub[TARGET_COLS] = sub[TARGET_COLS].fillna(prior_class_means).clip(1e-5, 1 - 1e-5)
```
Add strict assertion checks verifying row count, columns, and value ranges before writing `submission.csv`.

**Warning signs:**
Missing series in edge-case test studies.

**Phase to address:**
Phase 6 (Inference Kernel & Verification).

---

### Pitfall 5: Submitting Binarized Predictions Instead of Continuous Probabilities

**What goes wrong:**
Public score plummets below 0.52 because participants threshold probabilities at 0.5.

**Why it happens:**
Misunderstanding ROC AUC as an accuracy/F1 metric. ROC AUC evaluates the ranking across all threshold cutoffs. A binarized prediction provides only a single operating point on the ROC curve.

**How to avoid:**
Always output continuous sigmoid confidence scores $[0.0, 1.0]$. Never apply `> 0.5` thresholding to submission files.

**Warning signs:**
`submission.csv` containing only integers 0 and 1.

**Phase to address:**
Phase 6 (Submission Pipeline).

---
*Pitfalls research for: RSNA Knee Abnormality Detection Pipeline*  
*Researched: 2026-10-06*
