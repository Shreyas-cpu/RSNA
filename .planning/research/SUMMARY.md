# Project Research Summary

**Project:** RSNA Knee Abnormality Detection Pipeline  
**Domain:** Medical Deep Learning / Multimodal Knee MRI & Clinical NLP  
**Researched:** 2026-10-06  
**Confidence:** HIGH  

## Executive Summary

The RSNA Knee Abnormality Detection competition challenges participants to detect 12 joint abnormalities from multi-series knee MRI scans paired with unstructured radiology reports. The team's current development baseline (`rsna-knee-submission-baseline`) stands at **Version 9 with a verified Public Score of 0.584** (Macro AUC ROC). To systematically beat this baseline and compete for both the **$60,000 Accuracy Track** and the **$17,000 Model Efficiency Track**, the pipeline must transition from simplistic feature extraction to a structured 2.5D multi-plane attention architecture augmented with weak supervision.

Our research indicates that attempting full 3D dense volumetric CNNs is an architectural trap: it causes severe VRAM bottlenecks on 8GB/16GB GPUs, inflates training times, and overfits to background slices. Instead, the winning paradigm in medical Kaggle challenges combines **2.5D slice encoders (ConvNeXt-V2 / EfficientNet-V2 / DINOv2)** with **Multiple Instance Learning (MIL) Attention Pooling** across anatomical planes (Sagittal, Coronal, Axial).

Critically, because thousands of training exams feature unannotated narrative impressions, deploying an offline LLM weak-label extractor can expand the supervised training volume by 3–5x. Pairing this with high-speed C++ DICOM decompressors (`dicomsdl` / `libjpeg-turbo`) guarantees sub-15 minute test inference on Kaggle, securing high marks on the Efficiency Leaderboard while elevating competitive AUC toward >0.65+.

## Key Findings

### Recommended Stack

- **Deep Learning Core**: PyTorch 2.2+ with automatic mixed precision (AMP fp16/bf16).
- **Vision Backbones**: `timm` (ConvNeXt-V2, EfficientNet-V2, DINOv2).
- **Fast DICOM Engine**: `dicomsdl` with fallback to `pydicom` and `pylibjpeg` for high-throughput slice decompression.
- **Data & Augmentation**: `Albumentations` for spatial/contrast transformations, `polars`/`pandas` for metadata.
- **Validation**: `scikit-learn` `StratifiedGroupKFold` grouped by `StudyInstanceUID`.

### Expected Features

**Must have (table stakes):**
- Fast multi-threaded DICOM reader handling standard and compressed transfer syntaxes.
- Plane classification and slice sorting (Sagittal, Coronal, Axial).
- Central slab / multi-slice extraction with tissue-specific windowing.
- 12-class multi-label sigmoid prediction head.
- Leakage-free 5-fold Stratified Group Cross-Validation.
- Self-contained, offline-compatible Kaggle inference kernel generator.

**Should have (competitive differentiators):**
- LLM weak supervision: extracting structured 12-class pseudo-labels from free-text reports in `train.csv`.
- Gated Attention / MIL Pooling across slice sequences to dynamically focus on lesion slices.
- Multi-plane feature concatenation for compound multi-ligament / OA pathologies.
- Positivity-weighted loss (`BCEWithLogitsLoss(pos_weight=...)`) to prevent minority class collapse.
- Model Efficiency Track optimizations: sub-millisecond DICOM parsing and lean backbone inference.

### Architecture Approach

The architecture is structured as a decoupled, multi-stage modular pipeline:
1. **Data Ingestion & Preprocessing**: Reads raw DICOMs, extracts anatomical planes, windows pixel intensities, and samples 2.5D slabs.
2. **Weak Supervision Generator**: Parses narrative radiology reports into structured silver labels.
3. **Multi-Plane 2.5D Vision Model**: Encodes slices through a pretrained backbone, applies MIL attention pooling per plane, and concatenates embeddings into a study-level classifier.
4. **Validation Engine**: Tracks macro-averaged ROC AUC across 5 folds without data leakage.
5. **Kaggle Kernel Exporter**: Packages model checkpoints, wheels, and inference logic into an offline-executable submission notebook.

### Critical Pitfalls

1. **Slice/Series Leakage in CV**: Slices from the same study must never cross train and validation folds. Grouping strictly by `StudyInstanceUID` is mandatory.
2. **Kaggle Timeout on Inference**: Slow Python DICOM reading can cause >9h timeouts. `dicomsdl` + thread pool execution solves this.
3. **Minority Class Collapse**: Rare targets (`Fracture`, `Contusion`) drag down the unweighted macro AUC unless positive class weighting or temperature calibration is applied.
4. **Invalid Submissions**: Missing values or probability thresholding (binarization) ruin the ROC AUC score. Outputs must be continuous probabilities with NaN fallback.

## Phase Structure Recommendations

Based on these findings, we recommend decomposing the project into focused execution phases:
- **Phase 1: Validation Harness & High-Speed DICOM Ingestion Engine** (`dicomsdl`, plane grouping, slab extraction, Group K-Fold split).
- **Phase 2: Baseline Model Reproduction & Verification** (End-to-end pipeline reproducing and verifying the ~0.584 baseline locally).
- **Phase 3: Clinical Report Weak-Label Extractor** (LLM parsing of narrative reports to expand training dataset).
- **Phase 4: Multi-Plane 2.5D Attention Model** (ConvNeXt / EfficientNet-V2 + MIL Attention Pooling per plane).
- **Phase 5: Loss Optimization & Calibration** (Class-balanced BCE, temperature scaling targeting Macro AUC).
- **Phase 6: Ensembling, Kaggle Kernel Packaging & Efficiency Optimization** (Offline submission generator, TTA, sub-15 min inference validation).

---
*Synthesized for: RSNA Knee Abnormality Detection Pipeline*  
*Researched: 2026-10-06*
