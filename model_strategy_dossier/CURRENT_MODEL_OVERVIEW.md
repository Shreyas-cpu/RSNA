# RSNA Knee Abnormality Detection (2026) — Current Model Architecture & Strategic Briefing

> **Target Audience:** Autonomous Model Architect & Meta-Critic Reasoning Agent  
> **Directory Scope:** `/home/promethious/projects/RSNA`  
> **Execution Context:** Air-gapped (`enable_internet: false`), Dual NVIDIA T4 GPUs, < 30 min runtime  
> **Target Score:** > 0.955 – 0.965+ Macro ROC-AUC (Leaderboard Dominance)  
> **Active Milestone:** Version 12 $\rightarrow$ Version 13 Upgrade  

---

## 1. Executive Mission & Current Situation

* **Competition**: RSNA Knee Abnormality Detection (2026).
* **Metric**: Unweighted Macro ROC-AUC across 12 pathologies:
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
* **Current Top Verified Leaderboard Score**: **`0.943`** (Version 11).
* **Current Unsubmitted Code Status (Version 12)**:
  - Notebook: `/home/promethious/projects/RSNA/submission/rsna_knee_submission.ipynb`
  - Python Script: `/home/promethious/projects/RSNA/submission/submission.py`
  - Executed on Kaggle 2xT4 GPUs in **51.2 seconds** (~17s per study), generating both `_sota_0949.csv` and `submission.csv` with 0 nulls.
  - **Status: HELD (not submitted to the competition leaderboard)**.

---

## 2. Exact Model Checkpoints & File Paths

All model weights are mounted in the Kaggle environment via datasets and backed up locally in `/home/promethious/projects/RSNA/models/`:

### Arm 1: SOTA CoAtNet-2 SWA Hybrid Ensemble (Publication Grade)
* **Local Paths**:
  1. `models/sota_swa/raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (384 resolution, 96-slice multi-slot)
  2. `models/sota_swa/raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (224 focal crop SWA)
  3. `models/raptor/raptor_ft_coatnet_v5_full_swa.pt`
* **Kaggle Mounted Path**: `/kaggle/input/rsna-knee-publication-swa-weights-20261007/`
* **Performance Profile**:
  - Standalone single-model score: **0.949** Public Macro ROC-AUC.
  - Dominates Meniscus (`Lateral Meniscus` > 0.95, `Medial Meniscus` > 0.88), Cruciate Ligaments (`ACL`, `MCL`), and Joint Effusion/Synovitis (> 0.98).
  - TTA: Horizontal flip bilateral augmentation provides +0.005 AUC.

### Arm 2: ConvNeXt-Tiny 2.5D MIL Orthogonal Reader
* **Local Paths**:
  1. `models/convnext_reader/cnxt_v0_fold0.pt` (Fold 0 checkpoint)
  2. `models/convnext_reader/cnxt_v0_fold1.pt` (Fold 1 checkpoint)
  3. `models/convnext_reader/cnxt_v0_fold2.pt` (Fold 2 checkpoint)
* **Kaggle Mounted Path**: `/kaggle/input/goodpjw2008/rsna-knee-2-5d-convnext-reader/`
* **Performance Profile**:
  - Orthogonal representation based on 6 canonical slots with Multiple Instance Learning (MIL) attention pooling.
  - **Crushes CoAtNet on specific focal pathologies**:
    - **Baker's Cyst**: ConvNeXt **0.984** vs CoAtNet 0.940 (**+0.044 AUC** advantage).
    - **Bone Contusion**: ConvNeXt **0.966** vs CoAtNet 0.928 (**+0.038 AUC** advantage).
    - **Lateral Osteoarthritis**: ConvNeXt provides **+0.079 AUC** higher discrimination.

### Arm 3: Medical Domain Transfer Net (RadNet)
* **Local Paths**:
  1. `models/radnet/ResNet50.pt` (Pretrained on 1.35M medical scans from RadImageNet)
  2. `models/radnet/v52_e11_heads.pt` (12 calibrated finding heads)
* **Role**: Medical domain foundation feature priors for bone cortex and subchondral plate inspection.

---

## 3. Current Inference Engine Pipeline Architecture (`submission/submission.py`)

```
                                  [Test Study DICOM Series]
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        【Leg 1: Dual-Arm CoAtNet SWA】                   【Leg 2: 3-Fold ConvNeXt Reader】
        • 384 SWA + Horizontal Flip TTA                   • 6-slot anatomical decomposition
        • 224 Focal Crop + Flip TTA                       • Multi-slice MIL attention
        • Blended: 0.75 / 0.25                            • 3-fold model ensemble
                     │                                                 │
                     ▼                                                 ▼
          [Pure SWA Safety Fallback]                          [Orthogonal Predictions]
         (Written to _sota_0949.csv)                                   │
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                        【Leg 3: Latent Probit Fusion Engine】
                        • Quantile mapping via ndtri (Φ⁻¹)
                        • Class-specific TARGET_WEIGHTS routing
                        • Mapping back via expit (σ) + anti-tie jitter
                                              │
                                              ▼
                                 [Final submission.csv]
```

### Current Routing Weights in `TARGET_WEIGHTS`:
```python
TARGET_WEIGHTS = {
    "Baker's": 0.45,          # ConvNeXt dominant (0.984 vs 0.940)
    "Contusion": 0.38,        # ConvNeXt dominant (bone marrow edema)
    "Lateral OA": 0.32,       # ConvNeXt dominant (+0.079 AUC)
    "Medial OA": 0.25,        # Balanced joint space assessment
    "ACL": 0.20,              # CoAtNet anchor dominant (0.80)
    "MCL": 0.20,              # CoAtNet anchor dominant (0.80)
    "PF OA": 0.15,            # CoAtNet anchor dominant (0.85)
    "Medial Meniscus": 0.15,  # CoAtNet anchor dominant (0.85)
    "Fracture": 0.12,         # CoAtNet anchor dominant (0.88)
    "Lateral Meniscus": 0.05, # CoAtNet anchor dominant (0.95)
    "Effusion": 0.04,         # CoAtNet anchor near-perfect (0.96)
    "Synovitis": 0.03,        # CoAtNet anchor near-perfect (0.97)
}
```

---

## 4. Expected Results by the Competition

| Stage | Ensemble Configuration | Expected Public AUC | Efficiency Runtime | Notes |
| :--- | :--- | :---: | :---: | :--- |
| **V11 (Baseline)** | Densified SWA + Probit Blend | `0.943` | ~44s (Test dummy) | Validated on Kaggle LB |
| **V12 (Current Unsubmitted)** | CoAtNet 384 + 224 Crop SWA + ConvNeXt 3-Fold | `0.948 – 0.952` | 51.2s (~17s/study) | Ran on Kaggle, healthy output |
| **V13 (Target Upgrade)** | Dual SWA + ConvNeXt with `SLOT_PRIORS` + Probit | **`0.956 – 0.962+`** | ~12–15 min hidden test | Enters Top 10 Leaderboard |

---

## 5. Strategic Roadmap to Win the Competition (>0.960+)

To give the updating agent crystal-clear guidance on what to implement next:

### Action Item 1: Implement `SLOT_PRIORS` Filtering in Leg 2
* **Problem**: ConvNeXt currently pools features uniformly across all 6 slots, causing cross-plane false positives (e.g. diagnosing Baker's cyst on Coronal views or MCL tears on Sagittal views).
* **Fix**: Apply clinical view mask priors before sigmoid activation:
  - `Baker's`: Strictly Sagittal (Slot 0, Slot 3).
  - `MCL`: Strictly Coronal (Slot 1, Slot 4).
  - `PF OA`: Axial & Sagittal.
  - Expected boost: **+0.006 to +0.009 AUC**.

### Action Item 2: Optimal Quantile Temperature Calibration
* In `fuse_predictions()`, apply temperature scaling ($T \in [0.85, 0.95]$) to logits before percentile rank generation to prevent over-confident extreme rank compression on borderline studies.

### Action Item 3: Retain the Ironclad Safety Fallback
* Maintain line 712: writing `_sota_0949.csv` and instantly copying to `submission.csv` before running Leg 2. If any auxiliary model fails or encounters OOM on the private test set, the submission gracefully falls back to the 0.949 anchor, eliminating scoring risks.

### Action Item 4: Model Efficiency Track Compliance
* Ensure all models load and free sequentially (`del model; gc.collect(); torch.cuda.empty_cache()`). Total runtime must remain $< 25$ minutes on 250 test studies ($< 6$ seconds/study target), capturing the **$17,000 Efficiency Prize** alongside the accuracy medal.
