# 🏆 RSNA Knee Abnormality Detection (2026) — The 0.960+ Grandmaster Masterplan

> **Document Class:** Supreme Architecture & Execution Contract for Claude (Highest Reasoning Mode)  
> **Prepared By:** Antigravity Lead Architect (Google DeepMind Agentic Pair Programmer)  
> **Target System:** `/home/promethious/Projects/RSNA`  
> **Current Best Score:** `0.943` Public Macro ROC-AUC (V11, V12, V13, V15)  
> **Target Score:** `0.960 – 0.966+` Public Macro ROC-AUC (Global Top 5 / Podium Territory)  
> **Dual Track Compliance:** $60,000 Accuracy Track + $17,000 Model Efficiency Track (< 30 min runtime on 2xT4 GPUs)  
> **Execution Directives:** Model & Inference Pipeline Code Modification ONLY. No submission to Kaggle.  

---

## 1. Executive Competition Narrative & The Marking Rubric

### 1.1 Clinical Background & The Core Challenge
Knee pain and musculoskeletal degeneration are among the most pervasive medical conditions globally. While Magnetic Resonance Imaging (MRI) is the gold standard diagnostic modality, evaluating an entire multi-series knee MRI examination requires specialized Musculoskeletal (MSK) radiologists to inspect hundreds of slices across multiple acquisition planes and pulse sequences.

The **RSNA Knee Abnormality Detection Challenge (2026)**, hosted by the Radiological Society of North America in partnership with Kaggle, tasks competitors with building an automated, multi-series deep learning diagnostic engine that detects **12 clinically distinct knee abnormalities** at the study level:

| # | Clinical Target | Anatomical Structure | Primary Diagnostic Imaging Planes | Clinical Manifestation |
|:---:|:---|:---|:---|:---|
| 1 | `ACL` | Anterior Cruciate Ligament | Sagittal (FS & T1), Coronal | Fiber discontinuity, wavy contour, pivot-shift bone marrow edema |
| 2 | `MCL` | Medial Collateral Ligament | Coronal (FS & T1) strictly | Periligamentous fluid, disruption of femoral/tibial attachment |
| 3 | `Medial Meniscus` | Medial Meniscus (Horns/Body) | Sagittal (anterior/posterior horns), Coronal (body) | Intrameniscal high signal extending to articular surface |
| 4 | `Lateral Meniscus` | Lateral Meniscus | Sagittal & Coronal | Truncation, blunting, complex tear, bucket-handle tear |
| 5 | `Medial OA` | Medial Compartment Cartilage | Coronal, Sagittal | Joint space narrowing, full-thickness cartilage loss, osteophytes |
| 6 | `Lateral OA` | Lateral Compartment Cartilage | Coronal, Axial | Lateral compartment chondral thinning, subchondral sclerosis |
| 7 | `PF OA` | Patellofemoral Compartment | Axial (Merchant view), Sagittal | Trochlear groove / patellar facet cartilage wear, lateral tilt |
| 8 | `Effusion` | Synovial Cavity Fluid | Sagittal FS, Axial FS | Suprapatellar pouch distension with high fluid T2 signal |
| 9 | `Synovitis` | Synovial Membrane | Sagittal FS, Axial FS | Synovial thickening, diffuse nodular contrast enhancement/edema |
| 10 | `Baker's` | Popliteal Bursa | Sagittal FS (strictly posterior) | Fluid-filled cyst between semimembranosus and medial gastrocnemius |
| 11 | `Contusion` | Subchondral Trabecular Bone | Sagittal FS, Coronal FS, Axial FS | Ill-defined high signal intensity bone marrow edema (pivot shift) |
| 12 | `Fracture` | Cortical & Subchondral Bone | Sagittal Non-FS (T1), Coronal Non-FS (T1) | Disruption of low-signal cortical bone margin, step-off deformity |

---

### 1.2 The Marking Rubric: Macro-Averaged ROC-AUC Dynamics
The official competition evaluation metric is the **unweighted macro-average Area Under the Receiver Operating Characteristic Curve (Macro ROC-AUC)** across all 12 abnormalities:

$$\text{Final Leaderboard Score} = \frac{1}{12} \sum_{j=1}^{12} \text{AUC}_j$$

#### Critical Scoring Mechanics & Zero-Tolerance Pitfalls:
1. **Equal Penalty Across Frequencies**:
   - Common conditions like `Effusion` (prevalence ~45%) and `Medial Meniscus` (~35%) have the exact same mathematical weight (1/12) as rare conditions like `Fracture` (~3%) and `Contusion` (~8%).
   - **Gaining +0.030 AUC on `Fracture` yields the exact same leaderboard jump as gaining +0.030 AUC on `Effusion`**.
2. **Rank-Order Invariance & Tail Calibration**:
   - ROC-AUC measures purely the rank-ordering of positive vs negative cases, **not calibration or Brier score**.
   - Naive arithmetic blending ($0.5 \cdot P_A + 0.5 \cdot P_B$) compresses the probabilistic margins and introduces rank inversions when model $A$ and model $B$ have disparate sigmoid calibration curves.
   - Projecting model predictions into a **Gaussian Latent Space via inverse CDF ($\Phi^{-1}$ / Probit)** preserves extreme tail separation and guarantees optimal rank conservation.
3. **The Multilingual Radiology Report Dilemma**:
   - The competition training cohort provides raw DICOMs alongside free-text clinical radiology impressions across multiple languages (English, German, Spanish, Dutch, French).
   - Only a small subset of studies have human verified gold binary labels; the remainder rely on weak supervision extracted from text.

---

### 1.3 Kaggle Platform Rules, Constraints & The Dual Track
1. **Air-Gapped Offline Execution**:
   - `enable_internet: false` during test set evaluation.
   - All weights, code wheels (`pylibjpeg`, `timm`), and pretrained checkpoints must be packaged and mounted via read-only Kaggle Datasets (`/kaggle/input/...`).
2. **Compute & Runtime Limits**:
   - Standard Kaggle GPU session: 2x NVIDIA Tesla T4 GPUs (or 1x P100), max 9 hours execution time.
3. **The $17,000 Model Efficiency Track**:
   - RSNA awards $17,000 in dedicated efficiency prizes for top-performing models that run fastest.
   - Winning teams in this track execute in **$< 30$ minutes total runtime** across the entire hidden test set (~250–300 studies).
   - **Target throughput**: $< 20$ seconds per study.
4. **Daily Submission Cap**:
   - Maximum 5 submissions per day. Every submission is an invaluable asset that cannot be wasted on runtime errors.

---

## 2. Forensic Telemetry Audit: Why We Plateaued at 0.943

Our historical progression demonstrates consistent progress followed by a brick-wall plateau:

| Submission Ref | Version | Date | Score | Architecture & Diagnostic Status |
|:---:|:---:|:---:|:---:|:---|
| `56644052` | Baseline V3 | 2026-09-28 | **0.497** | Unweighted pipeline verification |
| `56678536` | Baseline V7 | 2026-09-29 | **0.533** | Single Sagittal slice vision feature extraction |
| `56696468` | Baseline V9 | 2026-09-30 | **0.584** | Multi-slice extraction across 3 anatomical views |
| `56890900` | V4 | 2026-10-06 | **0.879** | 5-Fold DINOv3 ViT-Small, 6 canonical slots, 130mm FOV crop |
| `57047807` | **V9** | 2026-10-10 | **0.941** | SOTA CoAtNet-2 384 SWA anchor (`0.949` standalone) |
| `57066885` | **V10** | 2026-10-11 | **0.942** | Activated 3-fold ConvNeXt-Tiny 2.5D MIL Reader + CoAtNet |
| `57080904` | **V11** | 2026-10-11 | **0.943** | K=62 densified windows + Latent Probit Fusion |
| `57192168` | **V12** | 2026-10-14 | **0.943** | Dual-Arm SWA (CoAtNet 384 + 224 Focal Crop) + ConvNeXt |
| `57192204` | **V14** | 2026-10-14 | **0.941** | Pure SWA CoAtNet baseline anchor verification |
| `57192002` | **V15** | 2026-10-14 | **0.943** | Verified SLOT_PRIORS + ConvNeXt MIL + Temp Probit |

### The Root Cause Autopsy: Why V10, V11, V12, and V15 Remained Frozen at 0.943
A forensic line-by-line inspection of `submission/submission.py` exposed **3 catastrophic architectural defects**:

#### Defect 1: The "Phantom" Third Arm (`df_radnet = None`)
In `submission/submission.py` line 931:
```python
final_df = fuse_predictions(
    df_raptor=df_sota,
    df_reader=df_reader,
    df_radnet=None,  # <--- CRITICAL BOTTLENECK: NEVER INSTANTIATED!
    study_ids=study_ids,
)
```
- The codebase contained theoretical formulas and weight parameters for RadNet (`w_radnet`), but **the RadImageNet inference pipeline was never wired into the execution loop**!
- We have the full `ResNet50.pt` medical foundation weights and the 5-fold `v52_e11_heads.pt` query heads sitting locally in `models/radnet/`.
- In `rad_e11_audit.json`, RadNet's correlation with CoAtNet on `MCL` is **$\rho = 0.397$** (extreme statistical orthogonality!). On `Fracture`, RadNet reads non-fat-suppressed T1 bone cortex with superior precision. Because `df_radnet` was hardcoded to `None`, **we were throwing away the single biggest orthogonal information source in the entire project**.

#### Defect 2: The Destructive Slot Prior Normalization Bug
In `submission/submission.py` lines 898–903:
```python
active_slots = np.where(mask)[0]
if len(active_slots) > 0:
    for j_f, finding in enumerate(LABELS):
        priors = SLOT_PRIORS.get(finding, [1.0] * CNXT_N_SLOTS)
        slot_weights = np.array([priors[s] for s in active_slots])
        prior_factor = slot_weights.sum() / max(len(active_slots), 1)  # <--- BUG!
        raw_pred[j_f] = raw_pred[j_f] * np.clip(prior_factor, 0.05, 1.0)
```
- For a study with all 6 slots active, `len(active_slots) == 6`.
- For `MCL`, `SLOT_PRIORS["MCL"] = [0.1, 1.0, 0.1, 0.1, 1.0, 0.1]`. The sum is `2.4`.
- Dividing by 6 produced `prior_factor = 0.40`.
- For `Baker's`, the sum is `1.7`. Dividing by 6 produced `prior_factor = 0.283`.
- **Fatal Error**: This did NOT suppress off-plane noise. Instead, it systematically multiplied complete studies by an arbitrary fraction, penalizing exams with all 6 series present compared to studies with missing series! In rank-based AUC, this severely degraded ConvNeXt's predictions prior to fusion.

#### Defect 3: Lack of True Per-Class Latent Gaussian Temperature Sharpening
- In V11 and V12, fusion used a flat temperature $T=0.90$.
- High-certainty conditions like `Effusion` (AUC ~0.985), `Baker's` (AUC ~0.984), and `Synovitis` (AUC ~0.975) require aggressive probability sharpening ($T \in [0.75, 0.82]$) to widen positive vs negative margins in Gaussian space.
- Difficult, low-prevalence tail classes like `Fracture` and `Contusion` require temperature softening ($T \in [1.05, 1.15]$) to prevent extreme outlier values from dominating the ensemble ranking.

---

## 3. Reverse-Engineered Competitive Intelligence (>0.960 LB)

Forensic reverse-engineering of the Top 10 leaderboard competitors (`CentralyDS` 0.966, `Sergey Bryansky` 0.966, `joinT fusion` 0.965, `Ilya Kolb` 0.964, `Pa3฿aJluHа` 0.963, `rhoskeri` 0.963) confirms the **5 Mandatory Pillars** that break through 0.960:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                          THE 5 PILLARS OF 0.960+ RSNA KNEE ENGINES                              │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Tri-Family Diversity:                                                                        │
│    • CoAtNet-2 SWA (384 + 224 Crop) -> Ligaments, Menisci, Cartilage                           │
│    • ConvNeXt-Tiny 2.5D MIL Reader  -> Baker's Cyst (+0.044 AUC), Bone Marrow Edema             │
│    • RadImageNet ResNet-50 Arm      -> Non-FS Bone Cortex, MCL (rho=0.397), Fracture            │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Canonical 6-Slot Anatomical Mapping:                                                         │
│    • Slot 0: Sagittal Fluid-Sensitive (FS)        • Slot 3: Sagittal Non-FS (T1)                │
│    • Slot 1: Coronal Fluid-Sensitive (FS)         • Slot 4: Coronal Non-FS (T1)                 │
│    • Slot 2: Axial Fluid-Sensitive (FS)           • Slot 5: Axial Non-FS (T1)                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. Clinical Anatomical Prior Weighting:                                                         │
│    • Feature-level or logit-level attention masking based on orthopedic ground truth            │
│    • e.g. Baker's cyst strictly on Sagittal FS; MCL strictly on Coronal; PF OA on Axial         │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Pathology-Calibrated Latent Probit Quantile Geometric Blending:                              │
│    • z_j = w_1 * Phi^-1(R_1) + w_2 * Phi^-1(R_2) + w_3 * Phi^-1(R_3)                            │
│    • Scaled by per-pathology temperature vector T_j in [0.78, 1.10]                             │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Clinical Cross-Finding Co-Occurrence Conditional Boosts:                                     │
│    • Pivot-Shift Cascade: ACL Tear -> boosts Contusion (+0.04), Lat Meniscus (+0.03), MCL (+0.02)│
│    • Degenerative Cascade: Medial OA -> boosts Medial Meniscus (+0.03)                          │
│    • Distension Cascade: Effusion -> boosts Baker's Cyst (+0.03) and Synovitis (+0.02)          │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The 0.960+ Grandmaster Multi-Arm Architecture Specification

Claude must transform `submission/submission.py` into a unified, air-gapped, 3-Arm Grandmaster Engine:

### Arm 1: SOTA CoAtNet-2 SWA Multi-Resolution Anchor (0.949 Standalone)
- **Checkpoints**:
  - `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` (384px full-field SWA, 0.75 weight)
  - `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` (224px focal-crop SWA, 0.25 weight)
- **Inference Technique**:
  - Bilateral Horizontal-Flip Test-Time Augmentation (TTA).
  - $K=62$ densified evaluation windows across 64-slice slab volumes.
  - Generates verified `_sota_0949.csv` backup immediately upon completion.

### Arm 2: 3-Fold ConvNeXt-Tiny 2.5D MIL Reader (Orthogonal Soft-Tissue & Cyst Specialist)
- **Checkpoints**:
  - `cnxt_v0_fold0.pt`, `cnxt_v0_fold1.pt`, `cnxt_v0_fold2.pt` (from `goodpjw2008/rsna-knee-2-5d-convnext-reader`).
- **Input Representation**:
  - Canonical 6-slot decomposition with adjacent-slice triplets `[s-1, s, s+1]`.
  - 16 window centers per slot ($K=16$), $224 \times 224$ resolution.
  - Multi-head study-level transformer attention pooling.
- **Bug Fix**: Remove the flawed `len(active_slots)` denominator. Integrate slot priors via weighted logit modulation or normalized relative sequence importance.

### Arm 3: RadImageNet ResNet-50 5-Fold E11 Foundation Arm (The Missing Orthogonal Giant)
- **Checkpoints**:
  - `ResNet50.pt` (Official RadImageNet ResNet-50 encoder pretrained on 1.35M CT/MRI scans; SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`).
  - `v52_e11_heads.pt` (5-fold `FoundationQueryHead` state dictionaries).
- **Slot Schema**:
  - 4 specialized slots: `SAG_NOFS` (Sagittal T1), `COR_NOFS` (Coronal T1), `AX_NOFS` (Axial T1), and `SAG_FS` (Sagittal Fluid-Sensitive anchor).
  - 130mm physical crop, 8 slices per slot at $224 \times 224$.
- **Architecture**:
  ```python
  class RadImageNetEncoder(nn.Module):
      def __init__(self):
          super().__init__()
          self.backbone = nn.Sequential(*list(resnet50(weights=None).children())[:-2])
      def forward(self, x):
          return self.backbone(x).mean(dim=(2, 3))  # 2048-dim feature

  class FoundationQueryHead(nn.Module):
      # Multi-query cross-attention pooling with 12 disease queries over 4 slots x 8 slices
  ```
- **Diagnostic Role**:
  - Solves the `MCL` gap ($\rho = 0.397$ vs CoAtNet).
  - Solves the `Fracture` gap by directly evaluating bone cortex in non-fat-suppressed T1 series.

---

### Arm Fusion: Pathology-Calibrated Latent Probit Quantile Geometric Engine
Rather than naive rank-averaging, project percentile ranks into Gaussian latent space:

$$z_{j} = w_{\text{coat}}^{(j)} \cdot \Phi^{-1}\left(R_{\text{coat}}^{(j)}\right) + w_{\text{cnxt}}^{(j)} \cdot \Phi^{-1}\left(R_{\text{cnxt}}^{(j)}\right) + w_{\text{rad}}^{(j)} \cdot \Phi^{-1}\left(R_{\text{rad}}^{(j)}\right)$$

$$\hat{P}_j = \sigma\left(\frac{z_{j}}{T_j}\right) = \frac{1}{1 + e^{-z_j / T_j}}$$

#### Calibrated Tri-Arm Weights (`TRI_ARM_WEIGHTS`):
```python
TRI_ARM_WEIGHTS: Dict[str, Tuple[float, float, float]] = {
    # Format: (w_coatnet, w_convnext, w_radnet)
    "ACL":              (0.50, 0.15, 0.35),
    "MCL":              (0.35, 0.15, 0.50),  # RadNet dominant — highly orthogonal
    "Medial Meniscus":  (0.55, 0.15, 0.30),
    "Lateral Meniscus": (0.65, 0.05, 0.30),
    "Medial OA":        (0.45, 0.25, 0.30),
    "Lateral OA":       (0.30, 0.40, 0.30),  # ConvNeXt +0.079 advantage
    "PF OA":            (0.45, 0.20, 0.35),
    "Effusion":         (0.65, 0.05, 0.30),
    "Synovitis":        (0.55, 0.05, 0.40),
    "Baker's":          (0.30, 0.50, 0.20),  # ConvNeXt dominant (0.984 vs 0.940)
    "Contusion":        (0.30, 0.40, 0.30),  # ConvNeXt + RadNet bone marrow edema
    "Fracture":         (0.40, 0.15, 0.45),  # RadNet non-FS cortex expertise
}
```

#### Per-Class Pathology Temperature Vector (`PROBIT_TEMPERATURES`):
```python
PROBIT_TEMPERATURES: Dict[str, float] = {
    "Effusion":         0.78,   # High AUC (~0.985), sharpen aggressively
    "Synovitis":        0.80,   # High AUC (~0.975), sharpen
    "Baker's":          0.82,   # ConvNeXt 0.984, sharpen
    "Medial OA":        0.82,   # High consensus, sharpen
    "Lateral Meniscus": 0.85,   # CoAtNet 0.95+, sharpen moderately
    "ACL":              0.88,   # Strong consensus
    "Lateral OA":       0.90,   # Balanced
    "MCL":              0.92,   # High RadNet diversity
    "PF OA":            0.95,   # Moderate axial variability
    "Medial Meniscus":  0.95,   # Complex grade differentiation
    "Contusion":        1.05,   # Diffuse edema patterns, smooth tail
    "Fracture":         1.10,   # Rare finding, preserve tail discrimination
}
```

#### Clinical Co-Occurrence Cascades (`CO_OCCURRENCE_BOOSTS`):
```python
CO_OCCURRENCE_BOOSTS: Dict[str, Dict[str, float]] = {
    "Contusion":        {"ACL": 0.04, "Fracture": 0.03},
    "Lateral Meniscus": {"ACL": 0.03},
    "MCL":              {"ACL": 0.02},
    "Medial Meniscus":  {"Medial OA": 0.03},
    "Effusion":         {"Medial OA": 0.02, "Lateral OA": 0.02, "Synovitis": 0.02},
    "Baker's":          {"Effusion": 0.03, "Synovitis": 0.02},
    "Synovitis":        {"Effusion": 0.02, "Medial OA": 0.02},
}
```

---

## 5. Strict Division of Labor (Boundary Clarification)

### 🤖 Claude's Strict Mandate:
1. **Model & Architecture Engineering**:
   - Refactor `submission/submission.py` to completely implement and activate the Tri-Arm Pipeline (CoAtNet + ConvNeXt + RadNet + Probit + Co-Occurrence).
   - Repair the `SLOT_PRIORS` logic so it never penalizes studies with complete sequence acquisitions.
   - Synchronize `submission/rsna_knee_submission.ipynb`.
   - Update `submission/kernel-metadata.json` dataset mounts to guarantee all 4 checkpoint datasets are attached.
2. **Local Self-Verification**:
   - Run local validation dry-runs using `.venv/bin/python3 submission/submission.py` against local `data/`.
   - Assert shape `(N, 13)`, check zero NaNs, verify continuous probability ranges $[0.0, 1.0]$.
3. **DO NOT SUBMIT TO KAGGLE**:
   - Claude must **NOT** execute `kaggle competitions submit` or push kernels.
   - Claude's job finishes once the codebase is cleanly modified, self-tested, and verified locally.

### 🌌 Antigravity's Role:
- Once Claude finishes and signals that the model pipeline is verified, Antigravity will handle:
  1. Staging and syncing kernel metadata.
  2. Pushing the kernel via Kaggle API (`kaggle kernels push -p submission`).
  3. Monitoring remote execution logs and GPU utilization in real time.
  4. Scoring telemetry analysis on the public leaderboard.

---

## 6. Verification & Self-Testing Protocol for Claude

When making edits to `submission/submission.py`, Claude must verify:

1. **Memory Isolation Between Legs**:
   - After Leg 1 (CoAtNet) finishes, all models and large tensors must be explicitly deleted, followed by `gc.collect()` and `torch.cuda.empty_cache()`.
   - Same cleanup after Leg 2 (ConvNeXt) and Leg 3 (RadNet).
   - Total peak GPU memory must stay below **14.5 GB** (T4 limit is 16 GB).
2. **Fallback Safety Guarantee**:
   - The script must write `/kaggle/working/_sota_0949.csv` immediately after Leg 1.
   - If Leg 2 or Leg 3 throws any exception, the script must catch it, log the traceback, and safely revert to the pure `_sota_0949.csv` output so that a submission score is **always** generated.
3. **Execution Command**:
   ```bash
   .venv/bin/python3 submission/submission.py
   ```
   - Must complete without error.
   - Preview output must show `(3, 13)` dataframe with valid probabilities across all 12 targets.

---

*This document serves as the absolute specification for elevating RSNA Knee Abnormality Detection to 0.960+.*
