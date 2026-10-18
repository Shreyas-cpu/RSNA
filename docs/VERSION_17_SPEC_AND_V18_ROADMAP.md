# RSNA Knee Abnormality Detection: Version 17 Spec & Version 18 Master Roadmap

This document serves as the permanent architectural, clinical, and algorithmic reference for **Submission Version 17** and the blueprint for the **Version 18 (0.965+ Cracking)** release.

---

## 1. Version 17: Implementation Specification & Audit

### 1.1 Objective & Strategy
Version 17 was deployed to restore and elevate our verified state-of-the-art baseline (**0.943 Public Macro ROC-AUC** from Versions 12 and 15), while ruthlessly eliminating the regression cause identified in Version 16.

### 1.2 Architectural Breakdown

```
+----------------------------------------------------------------------------------------------------+
|                                   RSNA KNEE PIPELINE - VERSION 17                                  |
+----------------------------------------------------------------------------------------------------+
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         │                                                 │
                         ▼                                                 ▼
            [LEG 1: Dual-Arm SWA CoAtNet]                    [LEG 2: ConvNeXt 2.5D MIL Reader]
           • SWA 384 (75% anchor weight)                     • 6 Canonical Anatomical Slots
           • SWA 224 Focal Crop (25% weight)                 • 3-Fold Ensemble (F0, F1, F2)
           • Bilateral Horizontal TTA                        • Universal Bilateral Mirroring TTA
           • Writes _sota_0949.csv (Safety net)              • Empty-mask exception guard
                         │                                                 │
                         └────────────────────────┬────────────────────────┘
                                                  │
                                                  ▼
                              [LEG 3: RadNet ResNet-50 Arm - BYPASSED]
                              • w_rad = 0.0 (Preserves calibrated probability)
                                                  │
                                                  ▼
                           [LEG 4: Probit Quantile Geometric Calibration]
                           • Inverse Normal CDF Projection: z_j = Φ⁻¹(q_j)
                           • Temperature Scaling: z_calib = z_j / T_j (T_j ∈ [0.78, 1.10])
                           • Logistic Sigmoid Re-mapping: P_j = σ(z_calib)
                                                  │
                                                  ▼
                             [LEG 5: Bi-Directional Pivot-Shift Coupling]
                             • ACL tear -> Lateral Meniscus (+0.03) & Contusion (+0.04)
                             • Lateral Plateau Contusion -> ACL tear (+0.035) (Reverse cascade)
                                                  │
                                                  ▼
                                       [FINAL SUBMISSION.CSV]
```

### 1.3 Key Mathematical Formulas & Parameters

1. **Dual-Anchor CoAtNet SWA Blend (Leg 1)**:
   $$P_{\text{anchor}} = 0.75 \cdot P_{\text{coatnet\_384}} + 0.25 \cdot P_{\text{coatnet\_224}}$$

2. **Universal Bilateral Test-Time Augmentation (Leg 2)**:
   For any volume slice tensor $x \in \mathbb{R}^{B \times C \times H \times W}$:
   $$x_{\text{flip}} = \text{flip}(x, \text{dim}=-1)$$
   $$P_{\text{cnxt}} = 0.5 \cdot \text{Model}(x) + 0.5 \cdot \text{Model}(x_{\text{flip}})$$

3. **Gaussian Latent Space (Probit Quantile Fusion)**:
   Percentile rank $q_{i,j} \in (0, 1)$ for study $i$ and pathology $j$:
   $$z_{i,j} = \Phi^{-1}(\text{clip}(q_{i,j}, \epsilon, 1 - \epsilon)) = \sqrt{2} \cdot \text{erfinv}(2 \cdot q_{i,j} - 1)$$
   $$P_{\text{fused}, i, j} = \sigma\left(\frac{z_{i,j}}{T_j}\right) = \frac{1}{1 + \exp\left(-\frac{z_{i,j}}{T_j}\right)}$$

4. **Calibrated Temperature Vector ($T_j$)**:
   ```python
   CALIBRATED_TEMPERATURES = {
       "ACL": 0.88,               # Sharpened (High confidence findings)
       "MCL": 0.94,
       "Medial Meniscus": 0.85,
       "Lateral Meniscus": 0.84,
       "Medial OA": 0.90,
       "Lateral OA": 0.82,        # Extreme sharpness
       "PF OA": 0.92,
       "Effusion": 0.78,          # Fluid sensitivity dominant
       "Synovitis": 0.95,
       "Baker's": 0.80,           # Focal cystic lesion
       "Contusion": 1.08,         # Smoothed (tail noise reduction)
       "Fracture": 1.10           # Extreme tail smoothing
   }
   ```

5. **Bi-Directional Pivot-Shift Trauma Matrix**:
   ```python
   CO_OCCURRENCE_BOOSTS = {
       "ACL": {
           "Contusion": 0.040,         # Forward trauma impact
           "Lateral Meniscus": 0.030   # Classic O'Donoghue/pivot triad
       },
       "Contusion": {
           "ACL": 0.035                # Reverse pathognomonic pivot coupling
       },
       "Medial Meniscus": {
           "Medial OA": 0.025
       }
   }
   ```

### 1.4 Verification & Telemetry
- **Local Dry Run**: Shape `(3, 13)`, Nulls: 0, Total Runtime: 12.6s (4.20s/study).
- **Kaggle Kernel Backend**: Completed cleanly in 59.7s on Kaggle GPU.
- **Efficiency Track Compliance**: Strictly <30 minutes, 100% air-gapped (`enable_internet: false`).
- **Submission Ref**: `57346056` (Kaggle evaluation queued).

---

## 2. Version 18: Master Roadmap to Break 0.965+

To break past the 0.950 barrier into the top-10 tier (0.965+ Macro ROC-AUC), Version 18 implements the following 5 high-leverage pillars:

### Pillar 1: Dual-T4 Process Sharding (2x Latency Acceleration)
- **Problem**: Kaggle allocates 2x NVIDIA Tesla T4 GPUs (`CUDA devices: 2`), but Version 17 runs single-threaded on `cuda:0`, leaving GPU 1 completely idle.
- **Solution**: Shard the test studies 50/50 across `cuda:0` and `cuda:1` via `torch.multiprocessing`:
  ```python
  import torch.multiprocessing as mp
  
  def process_shard(device_id, study_chunk, output_queue):
      torch.cuda.set_device(device_id)
      # Run inference pipeline on study_chunk
      output_queue.put(predictions)
  ```
- **ROI**: Inference drops from ~20s/study to ~10s/study, enabling us to add heavier ViT backbones without risking the 30-minute Efficiency Track deadline.

### Pillar 2: Quad-Arm DINOv2 / DINOv3 ViT Integration (Arm 3 Replacement)
- **Problem**: CNNs (ConvNeXt) and Hybrids (CoAtNet) have local inductive bias, missing global diffuse synovial thickening across the knee joint capsule.
- **Solution**: Replace the bypassed RadNet with a self-supervised Vision Transformer (DINOv2 ViT-Small).
- **Architecture**:
  - Arm 1: SWA CoAtNet 384 + 224 (Hybrid) [Weight: 0.50]
  - Arm 2: ConvNeXt-Tiny 2.5D MIL Reader (Modern CNN) [Weight: 0.30]
  - Arm 3: DINOv2 ViT-Small (Pure Self-Supervised Vision Transformer) [Weight: 0.20]
- **Expected Gain**: +0.012 – +0.018 Public AUC from genuine tri-family diversity.

### Pillar 3: Bayesian Optuna Calibration
- **Action**: Optimize the 12-class blend weights $w_j$ and temperature scaling factors $T_j$ over out-of-fold validation sets using Bayesian search:
  $$\max_{w, T} \text{Macro-AUC}(w, T)$$
- **Expected Gain**: +0.003 – +0.005 Public AUC.

### Pillar 4: Asymmetric Loss (ASL) for Training Refinement
- **Action**: Mitigate extreme label sparsity (<5% positive for Fractures/MCL) by dynamically down-weighting easy negatives during offline training:
  $$L_{\text{ASL}} = - y (1 - p)^{\gamma_+} \log(p) - (1 - y) (p_m)^{\gamma_-} \log(1 - p_m)$$
  where $p_m = \max(p - m, 0)$ with $\gamma_- = 4, \gamma_+ = 1, m = 0.05$.

### Pillar 5: Offline Local Verification Protocol
- All tests, shape checks, schema matching (13 columns), and fallback tests must run 100% locally on this Arch Linux workstation before triggering Kaggle CLI submission.

---

## 3. Git Branch Tracking Guide

To switch between or inspect any iteration:
- `main`: Clean repository root and core modules (`src/`, `docs/`, `orchestration/`).
- `v15-probit-anchor`: Verified 0.943 best-baseline (CoAtNet + ConvNeXt + Probit Quantiles).
- `v16-tri-arm`: Tri-Arm experiment with RadNet ResNet-50 audit.
- `v17-bilateral-tta`: Version 17 (Bilateral TTA + Pivot-Shift Coupling + Stripped RadNet).
- `v18-dual-t4-dinov3`: Active development branch for Dual-GPU sharding and DINOv2 integration.
