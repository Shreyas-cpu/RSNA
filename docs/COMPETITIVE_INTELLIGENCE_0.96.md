# RSNA Knee Abnormality Detection (2026) — Top 15 Solutions & Competitive Intelligence (>0.960 LB)

> **Goal:** Reverse-engineer top leaderboard solutions (>0.960 Public AUC) to implement winning techniques into our pipeline while maintaining qualification for the **$17,000 Model Efficiency Track**.
> **Date:** October 7, 2026  
> **Target Score:** 0.879 $\rightarrow$ **0.955 – 0.965+**

---

## 1. Top 15 Leaderboard Standings & Identified Contributors

| Rank | Team / Competitor Name | Handle / Identity | Public AUC | Known Public Stacks & Contributions |
| :---: | :--- | :--- | :---: | :--- |
| **1** | **Ilya Kolb** | `ilyakolb` | **0.964** | Biomedical engineering pipeline, multi-head attention pooling, clinical report distillation. |
| **2** | **Pa3฿aJluHа** | Community Ensemble | **0.963** | Raptor + CoAtNet + RadImageNet calibrated blend. |
| **3** | **rhoskeri** | Rahul Hoskeri (`rhoskeri`) | **0.963** | UC Merced EECS; multi-view spatial attention & clinical text fusion. |
| **4** | **CentralyDS** | Data Science Team | **0.963** | DINOv3 + CoAtNet SWA checkpoints with finding-specific routing. |
| **5** | **Wassim Dobbi** | `wassimdobbi` | **0.961** | Medical vision transformer + 6-slot contrast decomposition. |
| **6** | **joinT fusion** | Consortium | **0.961** | Cross-attention fusion over Sagittal/Coronal/Axial fat-sat and non-fat-sat series. |
| **7** | **NMarkS** | Research Group | **0.961** | ConvNeXt-Tiny + DINOv2 ensemble with rank-based calibration. |
| **8** | **🦵YOKKOISHO🦵** | Competitor | **0.960** | Multi-plane 2.5D slab extraction with 336px FOV center cropping. |
| **9** | **TomKroumov** | `tom1115` | **0.960** | ResNet50-RadImageNet + DINOv3 ViT-Small. |
| **10** | **dalab** | Academic Lab | **0.960** | Multi-sequence MRI transformer with clinical prior masking. |
| **11** | **RB & Tucker & Tony**| Trio | **0.960** | Diverse backbone ensembling + TTA. |
| **12** | **daimaru** | Competitor | **0.959** | 6-slot anatomical extraction with slot-prior weighting. |
| **13** | **Zhengru Li** | Competitor | **0.959** | RadImageNet transfer learning + text weak supervision. |
| **14** | **Attention Is All You Knee 🦵** | Competitor | **0.959** | Multi-head cross-attention across pulse sequences. |
| **15** | **knee-deep** | Competitor | **0.959** | Depth-zone MIL pooling + calibrated sigmoid scaling. |

---

## 2. Key Published Repositories & Kernels Analyzed

Through Kaggle API and code reverse-engineering, we retrieved and analyzed the following landmark community solutions:

1. **`goodpjw2008/rsna-knee-stack-2-5d-convnext-mil-lb-0-944`** (LB 0.944):
   - **Stack Architecture:** ConvNeXt-Tiny 2D slice encoder $\rightarrow$ 2-layer Study Transformer $\rightarrow$ 12 abnormality-specific attention heads.
   - **Slot Schema:** 6 slots ($3 \text{ planes} \times 2 \text{ fat-sat conditions}$).
   - **Empirical Breakthrough:** Verified that image model alone achieves 0.930, report text pseudo-labels achieve 0.896 on gold data, and rank-blending them reaches 0.944.

2. **`maverickss26/rsna-knee-0942-restructured`** (LB 0.942):
   - **Multi-Stage Fusion:** DINOv3 ViT-S base $\rightarrow$ A5 attention folds ($w=0.52$) $\rightarrow$ RadImageNet ResNet50 ($\alpha=0.55$) $\rightarrow$ Raptor 94-window dense sampling $\rightarrow$ CoAtNet SWA checkpoints.
   - **Finding-Specific Routing:** Assigns distinct model weights per target column:
     - `Lateral Meniscus`: 100% CoAtNet arm
     - `Medial Meniscus`: 80% CoAtNet arm
     - `ACL`, `Lateral OA`, `Fracture`: 75% CoAtNet arm
     - Other findings: 60% CoAtNet / 40% DINO + RadImageNet.

3. **`mattiaangeli/bend-the-knee-to-dinov3-ensembled`**:
   - **Clinical Anatomical Prior Table (`SLOT_PRIOR_TABLE`):**
     Hard-codes clinical anatomical rules so heads only attend to diagnostic views:
     - `Baker's cyst`: strictly Sagittal (Slot 0)
     - `MCL`: Coronal (Slots 1, 4)
     - `PF OA`: Axial & Sagittal (Slots 0, 2, 5)
     - `Effusion / Synovitis`: Sagittal & Axial (Slots 0, 2)
   - **Multilingual Report Negation Parser:** Extracts clinical impressions across English, Spanish, French, Dutch, and German with lexical scoping.

4. **`jiweiliu/rsna-knee-fast-2xt4-inference`**:
   - **Dual T4 Parallel Sharding:** Spawns two independent worker processes across `cuda:0` and `cuda:1`, sharding test cases 50/50. Cuts inference latency by exactly 50% without altering outputs.

---

## 3. The 5 Core Pillars Powering >0.960 Scores

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PILLAR 1: 6-SLOT DECOMPOSITION                     │
│   Plane Index (0..2) + 3 * (1 - is_fluid_sensitive) -> 6 Canonical Slots │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│                  PILLAR 2: TRI-FAMILY BACKBONE DIVERSITY                 │
│   1. Vision Transformer (DINOv3 ViT-Small)                              │
│   2. Medical Domain CNN (ResNet-50 RadImageNet, 1.3M medical scans)     │
│   3. Modern ConvNet / Hybrid (ConvNeXt-Tiny / CoAtNet)                  │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│             PILLAR 3: FINDING-SPECIFIC CLINICAL ROUTING                 │
│   Apply anatomical relevance priors (e.g. MCL strictly on Coronal,       │
│   Baker's on Sagittal) to eliminate cross-plane false positives.       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│             PILLAR 4: MULTILINGUAL NLP IMPRESSION BLENDING              │
│   Fuse image predictions with clinical report text scores               │
│   (our 4,407 silver impressions provide 0.896 base AUC alone).          │
└────────────────────────────────────┬────────────────────────────────────┘
                                     │
┌────────────────────────────────────┴────────────────────────────────────┐
│             PILLAR 5: DUAL T4 MULTI-PROCESS SHARDING                    │
│   Process test cohort across GPU 0 & GPU 1 in parallel. Runtime <12 min. │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Concrete Implementation Blueprint for Version 5 (Target: 0.955 – 0.965+)

To surpass 0.960 while maintaining Model Efficiency Track qualification (<30 min runtime):

### Phase 1: Incorporate `SLOT_PRIOR_TABLE` Masking
Add anatomical prior weights to each classification head in `KneeNet6Slot`:
```python
SLOT_PRIORS = {
    'ACL': [1.0, 0.2, 0.5, 0.8, 0.2, 0.5],
    'MCL': [0.1, 1.0, 0.1, 0.1, 1.0, 0.1],
    'Medial Meniscus': [0.9, 0.9, 0.2, 0.7, 0.7, 0.2],
    'Lateral Meniscus': [0.9, 0.9, 0.2, 0.7, 0.7, 0.2],
    'Medial OA': [0.2, 1.0, 0.2, 0.2, 1.0, 0.5],
    'Lateral OA': [0.2, 1.0, 0.2, 0.2, 1.0, 0.5],
    'PF OA': [0.8, 0.2, 1.0, 0.5, 0.2, 0.8],
    'Effusion': [1.0, 0.3, 1.0, 0.2, 0.1, 0.2],
    'Synovitis': [1.0, 0.3, 1.0, 0.2, 0.1, 0.2],
    "Baker's": [1.0, 0.1, 0.1, 0.3, 0.1, 0.1],
    'Contusion': [0.8, 0.8, 0.8, 0.4, 0.4, 0.4],
    'Fracture': [0.8, 0.8, 0.8, 1.0, 1.0, 1.0]
}
```

### Phase 2: Finding-Specific Head Blending
Combine RadImageNet, DINOv3, and CoAtNet arms using finding-specific routing ratios rather than global weights:
- `Lateral Meniscus`: 100% CoAtNet / ConvNeXt
- `Medial Meniscus`: 80% CoAtNet / 20% DINO
- `ACL`: 75% CoAtNet / 25% RadImageNet
- `Effusion / Synovitis`: 60% DINOv3 / 40% RadImageNet

### Phase 3: Rank-Based Quantile Normalization
Ensure monotonic rank-average ensemble:
```python
def rank_blend(preds_list, weights):
    blended = np.zeros_like(preds_list[0])
    for col in range(blended.shape[1]):
        col_ranks = np.zeros(blended.shape[0])
        for p, w in zip(preds_list, weights):
            col_ranks += w * (rankdata(p[:, col]) - 0.5) / len(p)
        blended[:, col] = col_ranks
    return blended
```

### Phase 4: Dual T4 GPU Execution
Shard test cases using `torch.multiprocessing`:
- Process 0 handles even study indices on `cuda:0`
- Process 1 handles odd study indices on `cuda:1`
- Concat results before writing `submission.csv`. Total runtime drops to **~6 to 8 minutes**.
