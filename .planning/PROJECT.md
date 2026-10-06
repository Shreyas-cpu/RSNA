# RSNA Knee Abnormality Detection Pipeline

## What This Is

An end-to-end, competitive machine learning pipeline for the Kaggle RSNA Knee Abnormality Detection challenge. It ingests multimodal knee MRI exams (DICOM series across Sagittal, Coronal, and Axial planes) paired with clinical radiology reports, detects 12 target joint abnormalities, and produces high-speed, calibrated predictions that surpass the team's current Version 9 baseline (0.584 Public AUC) while optimizing for the Model Efficiency track.

## Core Value

Beat the 0.584 Public AUC baseline with a robust, reproducible 2.5D multi-plane attention architecture and generate submission-ready Kaggle kernels under strict runtime limits.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] **Fast DICOM Preprocessing Engine**: High-throughput parsing via `dicomsdl`/`pydicom`, orientation/plane categorization (Sagittal, Coronal, Axial), intensity clipping/windowing, and multi-slice slab extraction.
- [ ] **Clinical Report Weak-Label Extractor**: Automated LLM parser to extract structured 12-class pseudo-labels from unannotated free-text reports in `train.csv` to scale the effective training dataset.
- [ ] **2.5D Multi-Slice Attention Model**: Pretrained vision backbone (ConvNeXt / EfficientNet-V2 / DINOv2) extracting per-slice representation followed by Multiple Instance Learning (MIL) / attention pooling per plane and study-level classification head.
- [ ] **Loss & Calibration Strategy**: Class-imbalance-weighted multi-label BCE loss with temperature scaling and threshold calibration targeting Macro ROC AUC.
- [ ] **Stratified Group Validation**: 5-Fold Stratified Group K-Fold split grouped by `StudyInstanceUID` / patient ID preventing data leakage and matching competition evaluation.
- [ ] **Kaggle Kernel Exporter & Inference Pipeline**: Automated bundling of offline dependencies, model checkpoints, and inference script meeting Kaggle's offline constraints and optimized for the $17,000 Model Efficiency Track.

### Out of Scope

- Full 3D dense volumetric convolutions — too compute-heavy, memory-intensive for rapid iterations on 8GB VRAM / T4 GPUs, and prone to overfitting.
- End-to-end vision-language transformer training from scratch — prohibitive compute budget; weak label extraction + vision fine-tuning achieves better return on investment.

## Context

- **Competition**: Kaggle RSNA Knee Abnormality Detection ($77,000 prize pool).
- **Evaluation**: Macro-averaged Area Under ROC Curve (Macro AUC ROC) across 12 targets.
- **Current Baseline**: Team notebook `rsna-knee-submission-baseline` achieved a verified Public Score of **0.584** on Version 9.
- **Hardware Environment**: Local development on RTX 4060 GPU with mixed precision (AMP fp16/bf16); Kaggle execution environment on NVIDIA T4/P100 (9h max limit, offline).
- **Target Conditions (12)**: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
- **Reference Guide**: Full competition and notebook lifecycle documentation documented in [about_RSNA.md](file:///home/promethious/projects/RSNA/about_RSNA.md).

## Constraints

- **Submission Format**: Submissions must run strictly as offline Kaggle Notebooks outputting `submission.csv` with 12 continuous probabilities.
- **Efficiency Constraint**: Total inference runtime on Kaggle hidden test set must be sub-hour (ideally < 15 minutes) to compete for the $17,000 Efficiency Prize.
- **GPU Memory**: Local training and Kaggle inference must fit comfortably within standard VRAM limits (8GB - 16GB).
- **Evaluation Metric**: Unweighted macro average across 12 targets penalizes poor minority-class calibration (`Fracture`, `Contusion`).

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 2.5D Multi-Slice + Attention Pooling over full 3D CNN | Faster training iteration, higher image resolution per slice, lower VRAM consumption, superior performance on medical leaderboards | — Pending |
| Multi-plane separation (Sagittal, Coronal, Axial) | Different abnormalities (ACL vs Meniscus vs PF OA) require distinct anatomical orientations | — Pending |
| Weak supervision from clinical reports | Thousands of training exams only have narrative text; LLM extraction dramatically expands supervised volume | — Pending |
| Dual-track submission optimization (Accuracy + Efficiency) | High-speed DICOM I/O (`dicomsdl`) and lean backbones qualify for both the $60k accuracy pool and the $17k efficiency track | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-06 after initialization*
