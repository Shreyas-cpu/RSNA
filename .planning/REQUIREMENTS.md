# Requirements: RSNA Knee Abnormality Detection Pipeline

**Defined:** 2026-10-06  
**Core Value:** Beat the 0.584 Public AUC baseline with a robust, reproducible 2.5D multi-plane attention architecture and generate submission-ready Kaggle kernels under strict runtime limits.  

## v1 Requirements

### Data Ingestion & High-Speed DICOM Engine (INGEST)

- [ ] **INGEST-01**: C++ accelerated DICOM reader (`dicomsdl` / `libjpeg-turbo` with `pydicom` fallback) supporting multi-threaded slice decompression.
- [ ] **INGEST-02**: Automatic anatomical plane categorization (Sagittal, Coronal, Axial) from DICOM headers / series descriptions.
- [ ] **INGEST-03**: Intensity rescaling, tissue-specific knee windowing, and spatial aspect ratio normalization.
- [ ] **INGEST-04**: 2.5D slab sampler (extracting 3-channel contiguous slice slabs) with local disk/memory cache for zero-bottleneck training.

### Radiology Report Weak Supervision (WEAK)

- [ ] **WEAK-01**: Clinical NLP report parser to extract structured 12-class pseudo-probabilities from unannotated radiology text in `train.csv`.
- [ ] **WEAK-02**: Generation and verification of `silver_train.csv` to expand supervised training dataset volume.

### 2.5D Multi-Slice Vision Architecture (MODEL)

- [ ] **MODEL-01**: Pretrained 2D vision backbone integration (ConvNeXt-V2 / EfficientNet-V2 / DINOv2 via `timm`) extracting dense slice representations.
- [ ] **MODEL-02**: Multiple Instance Learning (MIL) / Gated Attention pooling module to dynamically weight and aggregate variable-length slice sequences per plane.
- [ ] **MODEL-03**: Multi-plane fusion block concatenating Sagittal, Coronal, and Axial embeddings into a unified study-level representation.
- [ ] **MODEL-04**: Multi-label classification head outputting continuous calibrated probabilities for all 12 target pathologies.

### Training Dynamics & Loss Formulation (TRAIN)

- [ ] **TRAIN-01**: Class-imbalance-weighted BCE loss (`pos_weight`) to ensure minority pathologies (`Fracture`, `Contusion`) receive adequate gradient signal.
- [ ] **TRAIN-02**: Mixed-precision training pipeline (PyTorch AMP fp16/bf16) with gradient accumulation, AdamW optimizer, and Cosine Annealing learning rate schedule.
- [ ] **TRAIN-03**: Semi-supervised / weak-label joint training schedule blending ground-truth labels with LLM silver labels.

### Validation & Metric Rigor (EVAL)

- [ ] **EVAL-01**: Leakage-free 5-Fold Stratified Group K-Fold splitting scheme grouped strictly by `StudyInstanceUID`.
- [ ] **EVAL-02**: Exact Macro-averaged ROC AUC calculation across all 12 classes with per-class score logging to prevent silent regression.
- [ ] **EVAL-03**: Verification that local CV score correlates with public leaderboard progression (> 0.584 benchmark).

### Kaggle Kernel Exporter & Efficiency Packaging (SUBMIT)

- [ ] **SUBMIT-01**: Automated builder script creating a self-contained, offline-compatible Kaggle submission notebook (`submission.py` / `.ipynb`).
- [ ] **SUBMIT-02**: Bundling of offline wheels (`dicomsdl`, `timm`) and model weight checkpoints for direct mounting in Kaggle Datasets.
- [ ] **SUBMIT-03**: Pre-submission sanitation suite (verifying non-null continuous probabilities, correct row count, and valid test format).
- [ ] **SUBMIT-04**: End-to-end benchmark demonstrating test set inference runtime is well within Kaggle's 9-hour limit and competitive for the $17,000 Efficiency Track (< 30 minutes).

## v2 Requirements

- **V2-01**: Multimodal cross-attention transformer aligning slice features directly with radiology report embeddings at inference time.
- **V2-02**: Test-Time Augmentation (TTA) with multi-scale flip/crop voting.
- **V2-03**: Semi-supervised pseudo-label refinement through iterative self-training rounds.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full 3D dense volumetric CNNs | Too compute/memory-heavy for fast iterations on 8GB VRAM / T4 GPUs, high risk of overfitting to background voxels. |
| Probability thresholding (binarization 0/1) | Destroys rank ordering required for ROC AUC calculation, collapsing score. |
| Training end-to-end LLM + Vision from scratch | Prohibitive compute requirements; decoupled weak supervision delivers better performance per watt. |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INGEST-01 | Phase 1 | Complete |
| INGEST-02 | Phase 1 | Complete |
| INGEST-03 | Phase 1 | Complete |
| INGEST-04 | Phase 1 | Complete |
| EVAL-01   | Phase 2 | Complete |
| EVAL-02   | Phase 2 | Complete |
| MODEL-01  | Phase 3 | Complete |
| MODEL-02  | Phase 3 | Complete |
| MODEL-04  | Phase 3 | Complete |
| TRAIN-01  | Phase 4 | Pending |
| TRAIN-02  | Phase 4 | Pending |
| EVAL-03   | Phase 4 | Pending |
| WEAK-01   | Phase 5 | Pending |
| WEAK-02   | Phase 5 | Pending |
| MODEL-03  | Phase 6 | Pending |
| TRAIN-03  | Phase 6 | Pending |
| SUBMIT-01 | Phase 7 | Pending |
| SUBMIT-02 | Phase 7 | Pending |
| SUBMIT-03 | Phase 7 | Pending |
| SUBMIT-04 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 19
- Unmapped: 0 ✓

---
*Requirements defined: 2026-10-06*  
*Last updated: 2026-10-06 after initialization*
