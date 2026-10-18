# Version 5 Execution Bucket: Architectural Blueprint to Exceed 0.960+ Public AUC

> **Purpose:** Consolidated high-reasoning task specifications prepared for Claude 3.7 Sonnet execution.  
> **Target:** Push Kaggle Public Macro AUC from **0.879 $\rightarrow$ 0.960 – 0.965+** while maintaining Model Efficiency Track qualification (<10 min runtime).  
> **Timestamp:** October 7, 2026

---

## 1. High-Reasoning Component Breakdown

```
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                 VERSION 5 ARCHITECTURAL BUCKET                                   ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                  ║
║  1. Clinical Anatomical Prior Masking Layer (`SLOT_PRIOR_TABLE`)                                 ║
║     • Orthopedic view-relevance tensor masking over 6 canonical slots.                           ║
║     • Guarantees heads only attend to diagnostic pulse sequences, cutting false positives.       ║
║                                                                                                  ║
║  2. Tri-Model Family Fusion & Finding-Specific Weight Routing Matrix                             ║
║     • Model 1: DINOv3 ViT-Small (Fine-grained soft-tissue attention)                             ║
║     • Model 2: RadImageNet ResNet50 (1.3M medical CT/MRI pretraining)                            ║
║     • Model 3: CoAtNet / ConvNeXt-Tiny (High-capacity convolutional inductive bias)              ║
║     • Per-Finding Routing Matrix: Calibrated weighting optimized per target column.              ║
║                                                                                                  ║
║  3. Multilingual Clinical NLP Severity & Calibration Engine                                      ║
║     • NegEx + directional scoping across English, Spanish, and Dutch radiology narratives.      ║
║     • 4-Tier linguistic severity mapping (Definitive, Moderate, Equivocal, Negative).            ║
║     • Report score rank-blending (exploits 0.896 text AUC on gold annotations).                 ║
║                                                                                                  ║
║  4. Dual-GPU T4 Parallel Process Sharder (`torch.multiprocessing`)                               ║
║     • Concurrent study queue distribution across `cuda:0` and `cuda:1`.                          ║
║     • 2x throughput acceleration: Drops full test evaluation from ~25 min to <8 min.            ║
║     • Fully qualifies for the $17,000 Model Efficiency Track prize.                             ║
║                                                                                                  ║
║  5. Air-Gapped Code Synthesis & Kaggle Deployment Package                                        ║
║     • Complete standalone `submission/submission.py` & `rsna_knee_submission.ipynb`.             ║
║     • Updated `kernel-metadata.json` with external dataset mounts.                               ║
║     • Local dry-run verification $\rightarrow$ push V5 $\rightarrow$ submit to competition LB.   ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
```

---

## 2. Component Specifications

### Component 1: Clinical Anatomical Prior Masking Layer

Clinical imaging guidelines dictate which pulse sequences visualize specific knee structures:
- **ACL / PCL:** Visualized on Sagittal (FS and T1) + Axial.
- **MCL / LCL:** Visualized strictly on Coronal planes (FS highlights edema/sprain; T1 shows ligament continuity).
- **Menisci (Medial & Lateral):** Visualized on Sagittal (horns) + Coronal (body).
- **Osteoarthritis (Medial/Lateral/PF):** Visualized on Coronal (joint space narrowing) + Axial/Sagittal (patellofemoral cartilage).
- **Effusion / Synovitis:** Visualized on Sagittal + Axial Fluid-Sensitive sequences.
- **Baker's (Popliteal) Cyst:** Visualized almost exclusively on Sagittal Fluid-Sensitive sequences.

**Tensor Implementation:**
```python
# 6 Slots: [0: Sagittal_FS, 1: Coronal_FS, 2: Axial_FS, 3: Sagittal_T1, 4: Coronal_T1, 5: Axial_T1]
SLOT_PRIOR_WEIGHTS = torch.tensor([
    # Sag_FS, Cor_FS, Ax_FS, Sag_T1, Cor_T1, Ax_T1
    [1.00,    0.15,   0.50,  0.80,   0.15,   0.40],  # ACL
    [0.10,    1.00,   0.10,  0.10,   0.90,   0.10],  # MCL
    [0.95,    0.95,   0.15,  0.80,   0.80,   0.15],  # Medial Meniscus
    [0.95,    0.95,   0.15,  0.80,   0.80,   0.15],  # Lateral Meniscus
    [0.20,    1.00,   0.20,  0.20,   1.00,   0.50],  # Medial OA
    [0.20,    1.00,   0.20,  0.20,   1.00,   0.50],  # Lateral OA
    [0.85,    0.15,   1.00,  0.60,   0.15,   0.85],  # PF OA
    [1.00,    0.30,   1.00,  0.20,   0.10,   0.20],  # Effusion
    [1.00,    0.30,   1.00,  0.20,   0.10,   0.20],  # Synovitis
    [1.00,    0.05,   0.10,  0.30,   0.05,   0.05],  # Baker's Cyst
    [0.85,    0.85,   0.85,  0.35,   0.35,   0.35],  # Contusion
    [0.80,    0.80,   0.80,  1.00,   1.00,   1.00],  # Fracture
], dtype=torch.float32)
```

---

### Component 2: Finding-Specific Weight Routing Matrix

Rather than uniform averaging across models, each model family possesses distinct diagnostic competencies:
- **CoAtNet / ConvNeXt-Tiny:** Exceptional high-frequency edge detection for meniscal tears and ligament disruption.
- **RadImageNet ResNet50:** High sensitivity to cortical disruption, subchondral sclerosis, and bone contusions.
- **DINOv3 ViT:** Global contextual representation for effusion, synovitis, and soft-tissue masses.

**Routing Matrix:**
$$\mathbf{P}_{\text{ensemble}}^{(j)} = w_{\text{coat}}^{(j)} \cdot \mathbf{R}\left(P_{\text{coat}}^{(j)}\right) + w_{\text{rad}}^{(j)} \cdot \mathbf{R}\left(P_{\text{rad}}^{(j)}\right) + w_{\text{dino}}^{(j)} \cdot \mathbf{R}\left(P_{\text{dino}}^{(j)}\right)$$

Where $\mathbf{R}(P)$ is monotonic percentile rank-scaling:
$$\mathbf{R}(P) = \frac{\text{rank}(P) - 0.5}{N}$$

**Calibrated Target Ratios:**
- `Lateral Meniscus`: `[CoAt: 0.70, Rad: 0.15, DINO: 0.15]`
- `Medial Meniscus`: `[CoAt: 0.65, Rad: 0.15, DINO: 0.20]`
- `ACL`: `[CoAt: 0.55, Rad: 0.25, DINO: 0.20]`
- `MCL`: `[CoAt: 0.50, Rad: 0.30, DINO: 0.20]`
- `Fracture`: `[CoAt: 0.35, Rad: 0.45, DINO: 0.20]`
- `Contusion`: `[CoAt: 0.35, Rad: 0.45, DINO: 0.20]`
- `Effusion / Synovitis`: `[CoAt: 0.30, Rad: 0.20, DINO: 0.50]`
- `Medial OA / Lateral OA / PF OA`: `[CoAt: 0.40, Rad: 0.35, DINO: 0.25]`
- `Baker's Cyst`: `[CoAt: 0.30, Rad: 0.25, DINO: 0.45]`

---

### Component 3: Multilingual Clinical NLP Severity Scorer

A unified regex parser extracting polarity and clinical graded severity across English, Spanish, and Dutch radiology impressions:
- Detects anatomical mention $\rightarrow$ evaluates proximity negation window (pre- and post-negation tokens) $\rightarrow$ calculates pathology severity scalar.
- Generates a clinical prior vector $\mathbf{P}_{\text{nlp}} \in [0.04, 0.97]^{12}$.
- Blended at $\alpha = 0.20$ into the visual ensemble ranks.

---

### Component 4: Dual-GPU T4 Parallel Process Sharder

Kaggle environment exposes 2x NVIDIA T4 GPUs (`cuda:0`, `cuda:1`).

**Architecture:**
- Uses Python `torch.multiprocessing` with `spawn` context.
- Shards `test_df` indices evenly:
  - Worker 0: Even indices on `cuda:0`
  - Worker 1: Odd indices on `cuda:1`
- Each worker processes raw DICOMs $\rightarrow$ center-crops to 130mm FOV $\rightarrow$ runs fp16 forward passes.
- Gathers output dataframes via inter-process queue or temporary shard files (`/kaggle/working/_shard_0.csv`, `_shard_1.csv`).
- Re-merges into `/kaggle/working/submission.csv` preserving exact test row ordering.
- Total execution time: **$\approx 6$ to 8 minutes** for 1,200 studies.

---

## 3. Step-by-Step Execution Plan

1. **Step 1:** Construct the complete, self-contained `submission/submission.py` integrating all 4 components.
2. **Step 2:** Update `submission/kernel-metadata.json` with external weights datasets (`mattiaangeli/knee-mri-fold-weights`, `marwanmath/resnet-50-radimagenet-marwan`, `antoinegg1/rsna-knee-e11-diverse-heads-v20`).
3. **Step 3:** Re-export `submission/rsna_knee_submission.ipynb`.
4. **Step 4:** Execute local test run (`.venv/bin/python submission/submission.py`) to verify multi-processing, shape `(3, 13)`, 0 NaNs, and exact schema matching.
5. **Step 5:** Push Version 5 to Kaggle (`kaggle kernels push -p submission/`) and submit to competition leaderboard.
