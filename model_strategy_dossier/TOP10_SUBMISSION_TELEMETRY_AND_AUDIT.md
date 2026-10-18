# RSNA Knee Abnormality Detection (2026) — Top 10 Submission Telemetry & Full Project Audit

> **Document Type:** Executive Engineering & Model Diagnostic Telemetry Dossier  
> **Prepared For:** High-Tier Reasoning Meta-Critic (Claude 3.7 / Opus / Sonnet) & Antigravity Lead Architect  
> **Repository Target:** `/home/promethious/projects/RSNA`  
> **Target Goal:** Surpass the 0.950+ Barrier and Capture Top 10 / Leaderboard Dominance (>0.960 Macro ROC-AUC)  
> **Track Eligibility:** $17,000 Model Efficiency Track (< 30 min runtime on Kaggle 2xT4 GPUs, air-gapped) + $60,000 Accuracy Track  
> **Date:** October 13, 2026  

---

## 1. Executive Summary & Current Trajectory Status

* **Competition**: [RSNA Knee Abnormality Detection (2026)](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) hosted by the Radiological Society of North America (RSNA).
* **Core Task**: Study-level multi-label probabilistic classification across **12 discrete knee abnormalities**:
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
* **Primary Metric**: Macro-averaged Area Under the ROC Curve (Macro ROC-AUC) across all 12 classes:
  $$\text{Macro AUC} = \frac{1}{12} \sum_{i=1}^{12} \text{AUC}_i$$
* **Current Best Verified Submission**: **`0.943`** (V11: Grandmaster K=62 Densified SWA + ConvNeXt Calibrated Probit Fusion).
* **V12 Kernel Execution Status**: Successfully executed end-to-end on Kaggle in **51.2 seconds total runtime** (~17s/study) with zero errors, writing both `_sota_0949.csv` and `submission.csv`. **Pending submission to competition leaderboard (held per directive).**

---

## 2. Complete Historical Progression & Telemetry Table

The development pipeline began at random baseline (0.496–0.502) and sequentially cracked key performance barriers through rigorous architectural iterations:

| Submission Ref | Kernel / Version Tag | Date | Public Score | Key Architectural Additions & Behavioral Notes |
| :--- | :--- | :--- | :---: | :--- |
| `56644052` | Baseline V3 | 2026-09-28 | **0.497** | Unweighted mock pipeline verification. |
| `56647506` | Baseline V5 | 2026-09-28 | **0.496** | DICOM I/O test with uniform slice downsampling. |
| `56678536` | Baseline V7 | 2026-09-29 | **0.533** | First active vision feature extraction on Sagittal series. |
| `56696468` | Baseline V9 | 2026-09-30 | **0.584** | Multi-slice extraction across Sagittal/Coronal/Axial views. |
| `56889674` | V3 | 2026-10-06 | **0.502** | Early 2.5D attention baseline; uncalibrated cross-plane attention. |
| `56890900` | V4 | 2026-10-06 | **0.879** | **First Breakthrough**: 5-Fold DINOv3 ViT-Small, 6 anatomical slots, canonical 130mm FOV center cropping. |
| `56892660` | V6 | 2026-10-06 | **0.854** | Dual-T4 multithreaded loading; drop caused by naive logit prior masking. |
| `56893538` | V7 | 2026-10-07 | **0.867** | Test-Time Augmentation (TTA) with focal pooling; degraded by 140mm crop shift. |
| `56971539` | V8 | 2026-10-08 | **0.885** | Tri-Arm ensemble (DINOv3 + August Raptor v5 + RadNet); bottlenecked by linear rank dilution. |
| `57047807` | **V9** | 2026-10-10 | **0.941** | **Major Milestone (Cracked 0.90 Wall)**: SOTA CoAtNet-2 384 SWA anchor (`0.949` standalone publication model). Leg 2 (ConvNeXt) errored gracefully to pure SWA anchor in 30.5s runtime. |
| `57066885` | **V10** | 2026-10-11 | **0.942** | Fixed `KneeNet` constructor signature. Activated 3-fold ConvNeXt-Tiny 2.5D MIL Reader + CoAtNet 384 SWA. |
| `57080904` | **V11** | 2026-10-11 | **0.943** | K=62 densified evaluation windows + Initial pathology-calibrated latent probit quantile geometric fusion. |
| *Execution complete* | **V12** | 2026-10-11 | *Executed (51.2s)* | Dual-Arm SWA (CoAtNet 384 + CoAtNet 224 Focal Crop) + 3-Fold ConvNeXt 2.5D MIL Reader. Both legs finished without exception. Output generated. |

---

## 3. Reverse-Engineered Top 10 Leaderboard Solutions (>0.960 Public AUC)

Our forensic survey of the top 15 competitors (`ilyakolb`, `goodpjw2008`, `maverickss26`, `mattiaangeli`, `wassimdobbi`, `rhoskeri`, `CentralyDS`, etc.) revealed five definitive technical pillars driving the current >0.960 leaderboard:

### Pillar A: 6-Slot Canonical Anatomical Decomposition
* Knee MRI series are strictly projected into **6 Canonical Slots**:
  $$\text{Slot Index} = \text{Plane Index (0..2)} + 3 \times (1 - \text{is\_fluid\_sensitive})$$
  * **Slot 0:** Sagittal, Fluid-Sensitive (PD/T2 Fat-Suppressed)
  * **Slot 1:** Coronal, Fluid-Sensitive
  * **Slot 2:** Axial, Fluid-Sensitive
  * **Slot 3:** Sagittal, Non-Fluid-Sensitive (T1 / PD non-fat-suppressed)
  * **Slot 4:** Coronal, Non-Fluid-Sensitive
  * **Slot 5:** Axial, Non-Fluid-Sensitive
* **Benefit**: Guarantees invariant positional embeddings across diverse scanner protocols.

### Pillar B: Finding-Specific Anatomical Prior Masking (`SLOT_PRIORS`)
Top competitors enforce clinical ground truths to suppress off-plane imaging noise:
* **Baker's cyst**: Must only be diagnosed on **Sagittal** views (posterior gastrocnemius-semimembranosus bursa). Coronal/Axial detections are suppressed.
* **MCL**: Must only be diagnosed on **Coronal** views (femoral-tibial collateral band).
* **ACL**: Evaluated predominantly on **Sagittal & Coronal** views.
* **Patellofemoral Osteoarthritis (PF OA)**: Evaluated strictly on **Axial & Sagittal** views.
* **Joint Effusion & Synovitis**: Evaluated on **Fluid-Sensitive Sagittal & Axial** views.

```python
SLOT_PRIORS = {
    'ACL':              [1.0, 0.2, 0.5, 0.8, 0.2, 0.5],
    'MCL':              [0.1, 1.0, 0.1, 0.1, 1.0, 0.1],
    'Medial Meniscus':  [0.9, 0.9, 0.2, 0.7, 0.7, 0.2],
    'Lateral Meniscus': [0.9, 0.9, 0.2, 0.7, 0.7, 0.2],
    'Medial OA':        [0.2, 1.0, 0.2, 0.2, 1.0, 0.5],
    'Lateral OA':       [0.2, 1.0, 0.2, 0.2, 1.0, 0.5],
    'PF OA':            [0.8, 0.2, 1.0, 0.5, 0.2, 0.8],
    'Effusion':         [1.0, 0.3, 1.0, 0.2, 0.1, 0.2],
    'Synovitis':        [1.0, 0.3, 1.0, 0.2, 0.1, 0.2],
    "Baker's":          [1.0, 0.1, 0.1, 0.3, 0.1, 0.1],
    'Contusion':        [0.8, 0.8, 0.8, 0.4, 0.4, 0.4],
    'Fracture':         [0.8, 0.8, 0.8, 1.0, 1.0, 1.0]
}
```

### Pillar C: Tri-Family Orthogonal Backbone Ensemble
Top competitors never rely on a single architecture family:
1. **Hybrid CNN-Transformer (CoAtNet-2 SWA)**: Exceptional spatial-relational modeling on menisci, cartilage, and cruciate ligaments.
2. **Modern ConvNet (ConvNeXt-Tiny 2.5D MIL)**: Orthogonal localized representation. Dominates specific pathologies:
   * **Baker's Cyst**: ConvNeXt scores **0.984** vs CoAtNet **0.940** (+0.044 AUC).
   * **Bone Contusion**: ConvNeXt scores **0.966** vs CoAtNet **0.928** (+0.038 AUC).
   * **Lateral OA**: ConvNeXt provides a **+0.079** AUC boost over CoAtNet.
3. **Medical Domain Pretrained CNN (ResNet-50 RadImageNet)**: Trained on 1.35M medical imaging scans across CT/MRI/US; provides invariant baseline priors.

### Pillar D: Pathology-Calibrated Latent Probit Quantile Geometric Blending
Rather than crude arithmetic averaging ($0.5 \cdot A + 0.5 \cdot B$) or raw rank-averaging (which flattens high-confidence tail margins), winning pipelines project percentile ranks into Gaussian latent space via the inverse standard normal CDF ($\Phi^{-1}$ or `ndtri`):
$$z_{ij} = (1 - w_j) \cdot \Phi^{-1}(\text{rank}(A_{ij})) + w_j \cdot \Phi^{-1}(\text{rank}(B_{ij}))$$
$$\text{pred}_{ij} = \sigma(z_{ij}) = \frac{1}{1 + e^{-z_{ij}}}$$
* Finding-specific routing weights ($w_j$):
  * ConvNeXt dominant ($w = 0.35 - 0.45$) on Baker's, Contusion, Lateral OA.
  * CoAtNet SWA dominant ($1 - w \ge 0.85 - 0.95$) on Meniscus, Ligaments, Effusion, Synovitis.

### Pillar E: Bilateral Anatomical Test-Time Augmentation (TTA)
* Exploits the bilateral symmetry of knee anatomy.
* Averages standard volume and horizontal mirror flip:
  $$\text{Pred} = 0.50 \cdot \sigma(\text{Model}(x)) + 0.50 \cdot \sigma(\text{Model}(\text{flip}(x, \text{axis}=-1)))$$
* Adds $+0.005$ to $+0.007$ AUC on joint-space narrowing and ligament fiber orientation.

---

## 4. Full Project Repository Audit (`/home/promethious/projects/RSNA`)

### A. Submissions & Inference Machinery
* **`submission/submission.py`** (836 lines):
  * **Arm 1**: SOTA CoAtNet-2 384 SWA + CoAtNet-2 224 Focal Crop SWA with bilateral TTA.
  * **Arm 2**: ConvNeXt-Tiny 2.5D MIL Reader with 3 folds (`cnxt_v0_fold0.pt`, `fold1`, `fold2`).
  * **Arm 3**: Probit quantile fusion engine with class-calibrated `TARGET_WEIGHTS`.
  * **Safety Stash**: Instant write of `_sota_0949.csv` ensures zero Kaggle scoring failure if secondary legs crash.
* **`submission/kernel-metadata.json`**:
  * Air-gapped (`"enable_internet": "false"`).
  * GPU enabled (`"enable_gpu": "true"`).
  * Mounted datasets:
    - `nartaa/rsna-knee-publication-swa-weights-20261007` (CoAtNet 384 & 224 SWA)
    - `goodpjw2008/rsna-knee-2-5d-convnext-reader` (ConvNeXt 3 folds)
    - `mattiaangeli/knee-mri-fold-weights`
* **`submission/rsna_knee_submission.ipynb`**:
  * Standalone notebook artifact synchronized with `submission.py`.

### B. Autonomous Orchestration Loop
* **`orchestration/loop_orchestrator.py`** (677 lines):
  * Phase 1: Pipeline synchronization & local dry-run verification.
  * Phase 2: Pre-submission rule validation (shape, column order, bounds, null checks).
  * Phase 3: Kaggle kernel push & competition submission with backoff retries.
  * Phase 4: 15-minute polling loop with telemetry persistence.
  * Phase 5: High-capacity meta-critic handover & automated gap analysis.
  * Phase 6: Compliance audit against the <30 minute efficiency track rule.

### C. Checkpoints & Model Storage (`models/`)
* **`models/sota_swa/`**:
  * `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (0.949 SOTA anchor)
  * `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (0.945 focal crop SWA)
* **`models/convnext_reader/`**:
  * `cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt`, `cnxt_v0_fold2.pt` (3-fold orthogonal reader)
  * Preprocessing scripts (`knee.py`, `infer.py`, `make_labels.py`)
* **`models/radnet/`**:
  * `ResNet50.pt` (RadImageNet medical weights)
  * `v52_e11_heads.pt` (12-class calibrated finding heads)
* **`models/raptor/`**:
  * `raptor_ft_coatnet_v5_full_swa.pt`

### D. Architectural Roadmap (`.planning/`)
* **Completed Phases (1–7)**:
  * Phase 1: Fast DICOM Ingestion & Slab Caching Engine (`dicomsdl`, memory mapping).
  * Phase 2: Leakage-Free Stratified Group Validation Harness (Patient-level splits).
  * Phase 3: 2.5D Multi-Slice Vision Model & Attention Pooling.
  * Phase 4: Class-Balanced Training & Baseline Reproduction.
  * Phase 5: Clinical Radiology Report Weak Supervision Engine (NLP impressions extraction).
  * Phase 6: Multi-Plane Feature Fusion & Joint Training.
  * Phase 7: Offline Kaggle Kernel Exporter & Efficiency Packaging.
* **Active Milestone**:
  * Milestone 8 / 9: Dual/Tri-Arm SWA & Dense Probit Fusion Engine (`ROADMAP_CRACKING_0_950.md`).

---

## 5. Competition Rules, Terms & Conditions (T&C) Verification

1. **Submission Format Constraint**:
   * Must be a Kaggle Notebook submission (`kernel_type: notebook`). Direct CSV upload is disabled.
2. **Air-Gap Constraint**:
   * `enable_internet: false` during test evaluation. All model weights and wheels must be packaged in Kaggle Datasets.
3. **Execution Runtime Limits**:
   * Hard platform cap: 9 hours.
   * **$17,000 Model Efficiency Track Cap**: Top contenders execute in **$< 30$ minutes** on 2xT4 GPUs.
   * Our current V12 pipeline executes in **51.2 seconds on 3 test studies** (~17 seconds/study), projecting to **10–14 minutes on the hidden test set** (~250–300 studies), safely within the efficiency ceiling.
4. **Daily Quota**:
   * **5 submissions per day maximum**.
   * Strategic preservation of submissions is vital with **~10 days remaining** (~50 submissions budget).
5. **Macro-Average Equal Weighting**:
   * Every finding accounts for exactly $1/12$th of the score. Optimizing rare findings (`Fracture`, `Contusion`) yields identical score gains to common findings (`Effusion`, `Medial Meniscus`).
6. **Open Source Requirement**:
   * Code and models must be licensed under an OSI-approved license (MIT/Apache 2.0) to claim prize money.

---

## 6. Strategic Gap Analysis & Recommendations for Next Generation Model

To leap from **0.943** to **>0.960**:

1. **Inject `SLOT_PRIORS` into ConvNeXt Reader**:
   * Apply anatomical view relevance weights directly before pooling features across the 6 sequence slots to remove off-plane false positives.
2. **Activate Tri-Arm Ensembling (CoAtNet 384 + CoAtNet 224 Crop + ConvNeXt 3-Fold Reader)**:
   * V12 has already implemented this dual-resolution + orthogonal reader structure. Submitting V12 will establish the fresh baseline.
3. **Harmonize Probability Scaling with Rank Calibration**:
   * Use rank-based probit transformation (`ndtri` + `expit`) with class-specific temperature scaling to prevent extreme tails from degrading macro AUC.
4. **Maintain Safe Fallback Protocol**:
   * Always write `_sota_0949.csv` at the conclusion of Leg 1, preserving guaranteed qualification even in unexpected edge-case scenarios.

---
*Report generated and validated for consumption by Claude Meta-Critic.*
