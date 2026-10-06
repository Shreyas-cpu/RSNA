# Feature Research

**Domain:** Medical Imaging / Deep Learning (RSNA Knee Abnormality Detection)  
**Researched:** 2026-10-06  
**Confidence:** HIGH  

## Feature Landscape

### Table Stakes (Users Expect These)

Features mandatory for a working, valid Kaggle competition solution.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **DICOM Loader & Normalization** | Raw DICOM pixel arrays must be scaled by RescaleSlope/Intercept and windowed to knee tissue contrast. | MEDIUM | Must handle diverse pixel representations (MONOCHROME1 vs MONOCHROME2) and bit depths. |
| **Plane Identification (Sagittal, Coronal, Axial)** | Different knee pathologies are visible only on specific planes (e.g. ACL on Sagittal, collateral ligaments on Coronal). | LOW | Extracted from `ImageOrientationPatient` or DICOM series metadata. |
| **12-Class Multi-Label Classifier** | Must output probabilities for all 12 target pathologies for every test study. | MEDIUM | Model head with 12 sigmoid outputs. |
| **Stratified Group Cross-Validation** | Must reliably predict public/private LB performance without data leakage across series. | LOW | Grouped by `StudyInstanceUID`. |
| **Kaggle Offline Inference Kernel** | Submissions require offline execution producing `submission.csv` matching test set format. | MEDIUM | Must complete in < 9h with internet disabled. |

### Differentiators (Competitive Advantage)

Features that push the model from baseline (0.584) to medal-tier (>0.65+).

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **Clinical Report Silver-Label Extraction** | Unlabeled training studies with free-text reports can be automatically parsed via LLM into soft/silver labels, expanding training volume by 3-5x. | MEDIUM | Prompt LLM to output calibrated probabilities/labels for the 12 targets from raw text. |
| **2.5D Slab Multi-Slice Attention (MIL)** | Slices within an MRI volume have varying diagnostic relevance. Multi-Instance Learning (MIL) attention dynamically weights critical slices (e.g., tear location). | MEDIUM | Outperforms simple mean/max slice pooling significantly. |
| **Multi-Plane Feature Fusion** | Merges feature representations across Sagittal, Coronal, and Axial series per study into a unified diagnostic vector before final classification. | HIGH | Essential for compound conditions like osteoarthritis and multi-ligament injuries. |
| **High-Throughput C++ DICOM Decompression** | Uses `dicomsdl` / `libjpeg-turbo` with multithreading to slash test inference time. | LOW | Qualifies for the **$17,000 Model Efficiency Track**. |
| **Temperature Calibration per Target** | Optimizes sigmoid temperature/threshold scaling per class to maximize macro AUC ROC under class imbalance. | LOW | Prevents dominant negative classes from washing out minority targets. |

### Anti-Features (Commonly Requested, Often Problematic)

Features that sound attractive but degrade performance or burn unnecessary compute.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| **Heavy Full 3D ResNet/UNet Architecture** | Knee MRI is volumetric, so developers reflexively jump to 3D convolutions. | Massive VRAM usage, slow training, prone to severe overfitting on noisy background slices, impossible to iterate quickly on 8GB VRAM / T4. | 2.5D multi-slice backbones with attention pooling across the slice dimension. |
| **Threshold Binarization (0/1) for Submission** | Some confuse classification with detection and submit rounded binary labels. | Destroys rank ordering required for ROC AUC calculation, collapsing leaderboard score. | Keep outputs as unrounded continuous probabilities in $[0.0, 1.0]$. |
| **Training End-to-End LLM + Vision from Scratch** | Multimodal competition implies training a unified Vision-Language Model. | Prohibitively expensive, requires hundreds of GPU hours, and performs worse than decoupling weak label extraction and vision training. | Decouple: extract silver labels with LLM offline, train 2.5D vision backbone on combined dataset. |

## Feature Dependencies

```
[DICOM Parsing & Plane Classifier]
    └──requires──> [Windowing & 2.5D Slab Extraction]
                       └──requires──> [2.5D Vision Backbone Feature Extractor]
                                          └──requires──> [Multi-Slice MIL Attention Pooling]
                                                             └──requires──> [12-Class Prediction Head]
                                                                                └──requires──> [Offline Kaggle Kernel Export]

[Report Weak-Label Extractor] ──enhances──> [Training Dataset Volume & Accuracy]
[dicomsdl High-Speed I/O]    ──enhances──> [Efficiency Track Runtime (<15 min)]
```

### Dependency Notes

- **2.5D Vision Backbone requires Windowing & Slab Extraction**: Preprocessed inputs must be normalized to standard dimensions (e.g. 256x256 or 384x384) with consistent HU/intensity windows.
- **MIL Attention Pooling requires Slice-Level Feature Extraction**: Aggregates variable-length slice sequences into a fixed-length study-level embedding.
- **Weak-Label Extractor enhances Model Robustness**: Augmenting the limited ground-truth set with silver labels boosts generalization across unseen clinical sites.

---
*Feature research for: RSNA Knee Abnormality Detection Pipeline*  
*Researched: 2026-10-06*
