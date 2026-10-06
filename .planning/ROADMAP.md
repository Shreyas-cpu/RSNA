# Roadmap: RSNA Knee Abnormality Detection Pipeline

## Overview

This roadmap lays out an iterative, vertical MVP journey from raw multi-series DICOM scans and unannotated radiology reports to a competitive, medal-contending submission that systematically surpasses the team's Version 9 baseline (0.584 Public AUC) while qualifying for the $17,000 Model Efficiency Track.

## Phases

- [x] **Phase 1: Fast DICOM Ingestion & Slab Caching Engine** - High-speed C++ DICOM decoding, plane categorization, windowing, and 2.5D slab caching.
- [x] **Phase 2: Leakage-Free Stratified Group Validation Harness** - 5-fold Stratified Group K-Fold split and macro-averaged ROC AUC metric tracker.
- [x] **Phase 3: 2.5D Multi-Slice Vision Model & Attention Pooling** - timm vision backbone with Gated Attention / MIL pooling and 12-class prediction head.
- [ ] **Phase 4: Class-Balanced Training & Baseline V9 Reproduction** - Mixed-precision training with positive-class weighting reproducing and beating the 0.584 baseline locally.
- [ ] **Phase 5: Clinical Radiology Report Weak Supervision Engine** - LLM extraction of structured 12-class pseudo-probabilities from free-text reports in `train.csv`.
- [ ] **Phase 6: Multi-Plane Feature Fusion & Joint Training** - Unified model fusing Sagittal, Coronal, and Axial representations trained on blended ground-truth and silver labels.
- [ ] **Phase 7: Offline Kaggle Kernel Exporter & Efficiency Packaging** - Self-contained Kaggle submission notebook with bundled wheels, non-null sanity suite, and sub-15 minute execution.

## Phase Details

### Phase 1: Fast DICOM Ingestion & Slab Caching Engine

**Goal**: Build a high-throughput C++ accelerated DICOM reader and slab preprocessing engine that parses knee MRI series into normalized 2.5D slabs.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: INGEST-01, INGEST-02, INGEST-03, INGEST-04
**Success Criteria** (what must be TRUE):

  1. DICOM reader parses individual slices in < 2ms using `dicomsdl` / `libjpeg-turbo` with automatic fallback to `pydicom`.
  2. Series are reliably identified into Sagittal, Coronal, or Axial planes based on orientation vectors and metadata tags.
  3. Preprocessor produces normalized 3-channel 2.5D slabs (e.g. 256x256/384x384) with correct tissue contrast windowing.
  4. Slabs can be sampled and cached to local disk/memory for zero-bottleneck PyTorch DataLoader streaming.

**Plans**: 2 plans

Plans:
**Wave 1**

- [x] 01-01: Implement high-throughput `dicom_reader.py` with C++ decompression and orientation plane tagging.

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02: Implement slab windowing, spatial normalization, and PyTorch dataset cache streamer.

---

### Phase 2: Leakage-Free Stratified Group Validation Harness

**Goal**: Establish a mathematically sound, reproducible 5-Fold Stratified Group K-Fold split and exact Macro-averaged ROC AUC evaluation engine.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: EVAL-01, EVAL-02
**Success Criteria** (what must be TRUE):

  1. Validation folds are strictly grouped by `StudyInstanceUID` with zero patient/series cross-fold leakage.
  2. Target distributions across all 12 classes are evenly balanced across the 5 folds.
  3. Evaluation module computes exact Kaggle macro-averaged ROC AUC $\frac{1}{12}\sum \text{AUC}_i$ with granular per-class breakdown logging.

**Plans**: 1 plan

Plans:

- [x] 02-01: Build Stratified Group K-Fold split generator and standalone Macro AUC ROC evaluation harness.

---

### Phase 3: 2.5D Multi-Slice Vision Model & Attention Pooling

**Goal**: Implement the core deep learning architecture combining pretrained 2D slice backbones with Multiple Instance Learning (MIL) attention pooling.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: MODEL-01, MODEL-02, MODEL-04
**Success Criteria** (what must be TRUE):

  1. timm vision backbones (ConvNeXt-V2, EfficientNet-V2, DINOv2) successfully extract spatial feature vectors from 2.5D slabs.
  2. Gated Attention / MIL pooling dynamically aggregates variable numbers of slices per series into a fixed-dimensional study vector.
  3. Multi-label classification head outputs 12 continuous probabilities bounded in $[0.0, 1.0]$.
  4. Forward and backward passes run cleanly under PyTorch AMP without CUDA OOM on 8GB VRAM.

**Plans**: 2 plans

Plans:

- [x] 03-01: Build modular 2D slice feature extractor supporting ConvNeXt-V2 and EfficientNet-V2 backbones.
- [x] 03-02: Implement Gated Attention / MIL sequence aggregator and 12-class prediction head with unit tests.

---

### Phase 4: Class-Balanced Training & Baseline V9 Reproduction

**Goal**: Train the end-to-end 2.5D model using positive-class-weighted BCE loss, verify convergence, and exceed the team's 0.584 baseline in local CV.
**Mode:** mvp
**Depends on**: Phase 3
**Requirements**: TRAIN-01, TRAIN-02, EVAL-03
**Success Criteria** (what must be TRUE):

  1. Training loop executes with PyTorch AMP fp16, gradient accumulation, and Cosine Annealing scheduler.
  2. Class-imbalance weighting (`pos_weight`) prevents minority targets (`Fracture`, `Contusion`) from collapsing to zero gradient.
  3. Local validation Macro AUC matches or exceeds the 0.584 baseline benchmark (> 0.60 CV).

**Plans**: 2 plans

Plans:

- [ ] 04-01: Implement training loop with AMP, positive-weighted loss, optimizer, and checkpoint manager.
- [ ] 04-02: Run 5-fold training experiment, compute OOF validation Macro AUC, and verify baseline reproduction.

---

### Phase 5: Clinical Radiology Report Weak Supervision Engine

**Goal**: Build an NLP report extractor to generate structured 12-class pseudo-labels from narrative radiology impressions in `train.csv`.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: WEAK-01, WEAK-02
**Success Criteria** (what must be TRUE):

  1. LLM parser accurately maps complex unstructured radiology narratives to calibrated probabilities for the 12 targets.
  2. Extracted pseudo-labels on ground-truth verified studies achieve > 90% alignment/accuracy against human expert labels.
  3. Generated `silver_train.csv` expands available training cases by 3–5x.

**Plans**: 2 plans

Plans:

- [ ] 05-01: Implement clinical report prompt parser and LLM extraction script.
- [ ] 05-02: Validate extracted labels against labeled ground-truth subset and generate full `silver_train.csv`.

---

### Phase 6: Multi-Plane Feature Fusion & Joint Training

**Goal**: Fuse representations across Sagittal, Coronal, and Axial series and train jointly on gold ground-truth and LLM silver labels.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: MODEL-03, TRAIN-03
**Success Criteria** (what must be TRUE):

  1. Model integrates slice sequences from all three planes simultaneously per study.
  2. Joint training schedule seamlessly weights gold ground-truth and silver weak labels.
  3. Multi-plane fusion pushes local 5-fold CV score to > 0.65+ Macro AUC.

**Plans**: 2 plans

Plans:

- [ ] 06-01: Build multi-plane sequence fusion module concatenating Sagittal, Coronal, and Axial representations.
- [ ] 06-02: Execute joint training with silver label curriculum and validate CV improvements.

---

### Phase 7: Offline Kaggle Kernel Exporter & Efficiency Packaging

**Goal**: Package the trained pipeline into a self-contained, offline-compatible Kaggle submission notebook optimized for accuracy and the Efficiency Track.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: SUBMIT-01, SUBMIT-02, SUBMIT-03, SUBMIT-04
**Success Criteria** (what must be TRUE):

  1. Self-contained notebook runs completely offline using local dataset mounts and pre-bundled wheels.
  2. Generates `submission.csv` strictly compliant with Kaggle formatting (12 probability columns, non-null, valid IDs).
  3. Full test set inference executes in < 30 minutes, meeting Efficiency Track criteria.
  4. Kernel pushes directly to Kaggle via Kaggle API / CLI and executes successfully.

**Plans**: 2 plans

Plans:

- [ ] 07-01: Build offline bundle packager (wheels, model checkpoints, and self-contained inference script).
- [ ] 07-02: Implement post-inference verification suite and benchmark inference throughput on test data.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Fast DICOM Ingestion & Slab Caching Engine | 0/2 | Not started | - |
| 2. Leakage-Free Stratified Group Validation Harness | 0/1 | Not started | - |
| 3. 2.5D Multi-Slice Vision Model & Attention Pooling | 0/2 | Not started | - |
| 4. Class-Balanced Training & Baseline V9 Reproduction | 0/2 | Not started | - |
| 5. Clinical Radiology Report Weak Supervision Engine | 0/2 | Not started | - |
| 6. Multi-Plane Feature Fusion & Joint Training | 0/2 | Not started | - |
| 7. Offline Kaggle Kernel Exporter & Efficiency Packaging | 0/2 | Not started | - |

---
*Roadmap defined: 2026-10-06*  
*Last updated: 2026-10-06 after initialization*
