# 🏅 RSNA Knee Abnormality Detection (2026) — WINNING PLANNER

> **Mission Objective**: Capture Top 10 / Prize Territory (>0.966 Macro ROC-AUC) on the Global Leaderboard while maintaining full compliance with the $17,000 Model Efficiency Track (<30 min runtime on 2xT4 GPUs, air-gapped).  
> **Current Best Score**: `0.943` (V11) $\to$ `V13` (Under Kaggle evaluation)  
> **Target Score**: **`0.966 – 0.970+`** (Podium / Top 3)  
> **Date**: October 14, 2026  

---

## 1. Competitive Battlefield Audit (Top 10 Benchmark)

From the live leaderboard scan, the benchmark targets to beat are:

| Rank | Team | Score | Key Suspected Technique |
|:---:|:---|:---:|:---|
| **#1** | `CentralyDS` | **0.966** | Tri-family ensemble + pseudo-labeling on silver data |
| **#2** | `Sergey Bryansky` | **0.966** | Multi-plane 2.5D ConvNeXt + dense slice attention |
| **#3** | `joinT fusion` | **0.965** | Latent space rank geometric ensembling |
| **#5** | `Ilya Kolb` | **0.965** | SWA CoAtNet + clinical view priors |
| **#10** | `🦵YOKKOISHO🦵` | **0.963** | Top 10 cutoff barrier |
| **Our Team** | `Promethious` | **0.943 $\to$ V13** | Dual SWA + ConvNeXt + SLOT_PRIORS + Probit |

---

## 2. Our Active Baseline Architecture (Version 13)

```
                            [Test Study DICOM Series]
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
【Leg 1: Dual-Arm SWA Ensemble】                           【Leg 2: 3-Fold ConvNeXt Reader】
• CoAtNet-2 384 SWA (0.75 weight)                         • 6 Canonical Anatomical Slots
• CoAtNet-2 224 Focal Crop (0.25 weight)                  • 2.5D Triplet Slice Inputs
• Bilateral Horizontal-Flip TTA                           • Transformer MIL Attention Pooling
• Standalone Score: 0.949 Public AUC                                 │
           │                                                         ▼
           │                                              【Apply SLOT_PRIORS Mask】
           │                                              • Baker's Cyst ➔ Sagittal only
           │                                              • MCL ➔ Coronal only
           │                                              • PF OA ➔ Axial & Sagittal
           │                                              • Effusion/Synovitis ➔ Fluid-sensitive
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                  【Leg 3: Latent Probit Quantile Fusion】
                  • Quantile Projection via Φ⁻¹ (ndtri)
                  • Temperature Calibration (T = 0.90)
                  • Pathology Finding-Specific Weights (w)
                  • 40% Linear + 60% Probit Ensemble
                                        │
                                        ▼
                             [Final submission.csv]
```

---

## 3. The 4-Phase Progression to 0.966+

```
Phase 10 (Current)          Phase 11 (V14)               Phase 12 (V15)                Phase 13 (V16)
┌──────────────────┐       ┌────────────────────┐       ┌─────────────────────┐       ┌─────────────────────┐
│ V13 Telemetry    │       │ RadImageNet Tri-Arm│       │ Per-Class Temp Array│       │ Quad-Arm Ensemble   │
│ Ingestion &      │  ──►  │ Foundation Backbone│  ──►  │ & Dual-Axis TTA     │  ──►  │ & Final Freeze      │
│ Gap Analysis     │       │ (ResNet-50 1.35M)  │       │ Calibration         │       │ (LB 0.966 - 0.970+) │
│ (Target: ~0.958) │       │ (Target: ~0.962)   │       │ (Target: ~0.965)    │       │                     │
└──────────────────┘       └────────────────────┘       └─────────────────────┘       └─────────────────────┘
```

---

## 4. Candidate Ideas & Weapons Under Consideration

Here are the prime candidates known to deliver marginal gains in medical MRI competitions:

### A. Pre-processing & Extraction
- [ ] **Adaptive Histogram / CLAHE on Joint Space**: Enhances contrast on low-contrast cartilage and subchondral bone interfaces.
- [ ] **Anatomical Joint-Space Bounding Box Centering**: Detecting the femoral-tibial junction and cropping around the articular surface rather than center-cropping the full slice.
- [ ] **Slice Spacing Normalization**: Resampling volumes to a uniform 3.0mm slice thickness across different scanner protocols.

### B. Architectural Ensembling (Orthogonal Diversity)
- [ ] **RadImageNet ResNet-50 Activation**: Checkpoints already stored locally in `models/radnet/ResNet50.pt` + `v52_e11_heads.pt`. Provides medical domain-specific features pretrained on 1.35M CT/MRI scans.
- [ ] **DINOv3 ViT-Small Feature Distillation**: Self-supervised vision transformer features capturing structural geometry without human label bias.
- [ ] **3D Multi-Slice Attention Averaging**: Stacking 5 slices instead of 3 slices in the 2.5D ConvNeXt reader for deeper z-axis awareness.

### C. Post-Processing & Calibration
- [ ] **Per-Pathology Temperature Scaling Vector ($T_j$)**:
  - Crisp boundary classes (`Effusion`, `Lateral Meniscus`): $T \approx 0.75 - 0.85$
  - Rare/Ambiguous classes (`Fracture`, `Contusion`): $T \approx 1.05 - 1.15$
- [ ] **Rank Normalization with Power Transform**: Applying Box-Cox or power transform to percentile ranks before latent Gaussian mapping.
- [ ] **Cross-Finding Co-Occurrence Priors**:
  - e.g., If `ACL` tear is high-probability, boost `Contusion` (pivot-shift bone marrow edema) and `Lateral Meniscus`.
  - e.g., If `Medial OA` is high-probability, boost `Medial Meniscus` degeneration.

---

## 5. User's Secret Strategic Weapons (Ready to Ingest)

> 💡 **User Input Area**: The space below is ready to ingest your ideas, hypotheses, and custom strategies to reach the Top 3.

---

## 6. Efficiency Track Guardrails ($17,000 Prize Checklist)
- [x] Air-gapped (`enable_internet: false`)
- [x] Peak GPU VRAM $\le 4.5$ GB per GPU (achieved via sequential model load/unload)
- [x] Total runtime $\le 16$ min on 250 test studies (limit: 30 min)
- [x] Output strictly named `submission.csv` with zero NaNs
- [x] Fallback anchor (`_sota_0949.csv`) written before auxiliary model execution
