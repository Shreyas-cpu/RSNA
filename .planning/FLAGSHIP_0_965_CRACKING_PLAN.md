# RSNA Knee Abnormality Detection: Flagship 0.965+ Grandmaster Implementation Plan
**Target:** Break 0.965+ Public Leaderboard Macro ROC-AUC & Secure Top 10 Podiums  
**Track Qualification:** \$17,000 Model Efficiency Track (< 30 min runtime on Kaggle 2xT4 GPUs, air-gapped `"enable_internet": "false"`)  
**Status:** ACTIVE MASTER PLAN  

---

## 1. Executive Strategy & Root Cause Analysis

### Why Previous Versions Hit 0.943 and V16 Dropped to 0.893
1. **The V16 RadNet Failure (0.893)**:
   - In Version 16, RadNet ResNet-50 was given massive weights ($0.30 - 0.50$) across all findings (e.g. MCL: $0.50$, Fracture: $0.45$, Synovitis: $0.40$).
   - RadNet was trained on a coarse slice extractor with low validation resolution. Its predictions had high variance and poor rank separation, actively diluting the high-precision 0.949 CoAtNet + ConvNeXt signal.
   - **Resolution:** Purge RadNet completely from the primary ensemble.
2. **The 0.943 Ceiling in V11–V15**:
   - Strongest baseline achieved consistently ($0.943$).
   - Limitations: Only CoAtNet had bilateral TTA; ConvNeXt ran single-pass; Kaggle GPU 1 remained 100% idle; missing non-local transformer representation for diffuse synovial inflammation.
3. **The 0.965+ Top 10 Formula**:
   - **Quad-Family Diversity**: CoAtNet-2 384 SWA ($0.949$) + CoAtNet-2 224 Focal Crop SWA ($0.945$) + ConvNeXt-Tiny 2.5D MIL Reader ($3\text{ folds}$) + DINOv3 ViT-Small/14.
   - **Universal Bilateral TTA**: Symmetrical anatomical horizontal mirroring across all models ($+0.005$ AUC).
   - **Dual-T4 Process Sharding**: Full 2x hardware utilization via `torch.multiprocessing`, halving runtime to $<7\text{ min}$.
   - **Bi-Directional Kinematic Cascades**: Pivot-shift reverse prior (Contusion $\leftrightarrow$ ACL), effusion/synovitis mutual distension coupling.
   - **Temperature-Calibrated Latent Probit Quantiles**: $\Phi^{-1}$ Gaussian mapping with per-finding temperature vector $T_j \in [0.78, 1.10]$.

---

## 2. Model Architecture: The Quad-Arm Arsenal

```
                              [Test Study DICOM Volumes]
                                          |
                      +-------------------+-------------------+
                      |                                       |
             [Kaggle GPU 0 (cuda:0)]                 [Kaggle GPU 1 (cuda:1)]
             Even Studies (0, 2, 4...)               Odd Studies (1, 3, 5...)
                      |                                       |
                      +-------------------+-------------------+
                                          |
             +----------------------------+----------------------------+
             |                            |                            |
             v                            v                            v
   +--------------------+       +--------------------+       +--------------------+
   |       ARM 1        |       |       ARM 2        |       |       ARM 3        |
   | CoAtNet-2 384 SWA  |       | CoAtNet-2 224 SWA  |       | ConvNeXt 2.5D MIL  |
   | (Global Joint SOTA)|       | (Focal Micro-Tear) |       | (3-Fold Reader)    |
   | 0.949 Standalone   |       | 0.945 Standalone   |       | Baker's / Contusion|
   +--------------------+       +--------------------+       +--------------------+
             |                            |                            |
      Bilateral TTA                Bilateral TTA                Bilateral TTA
             |                            |                            |
             +-------------+--------------+                            |
                           |                                           |
                           v                                           |
              [SWA Master Anchor Blend]                                |
              (75% 384 + 25% 224 Crop)                                 |
                           |                                           |
                           +---------------------+---------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |                      ARM 4                        |
                       |       DINOv3 ViT-Small Self-Supervised Arm        |
                       |   (Non-Local Attention for Synovitis/Effusion)    |
                       +---------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |    Pathology-Calibrated Latent Probit Quantile    |
                       |      Geometric Fusion + Temperature Scaling       |
                       |                  (Tⱼ ∈ [0.78, 1.10])              |
                       +---------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |    Bi-Directional Biomechanical Trauma Cascades   |
                       |    - Contusion ↔ ACL Pivot-Shift Resonance        |
                       |    - Effusion ↔ Synovitis Distension Coupling     |
                       |    - Medial OA ↔ Medial Meniscus Wear             |
                       +---------------------------------------------------+
                                                 |
                                                 v
                       +---------------------------------------------------+
                       |            FINAL SUBMISSION (13 Columns)          |
                       |      Guaranteed Safe Fallback (_sota_0949.csv)    |
                       +---------------------------------------------------+
```

---

## 3. Detailed Technical Components

### Component 1: Dual-T4 Process Sharding (`torch.multiprocessing`)
* **Objective:** Cut inference latency from 14 minutes down to 6.5 minutes.
* **Implementation:**
  ```python
  import torch.multiprocessing as mp

  def worker_infer(gpu_id: int, study_subset: List[str], output_queue: mp.Queue):
      device = torch.device(f"cuda:{gpu_id}")
      # Run inference on study_subset
      output_queue.put(results_df)

  # In main runner:
  mp.set_start_method("spawn", force=True)
  half = len(study_ids) // 2
  q = mp.Queue()
  p0 = mp.Process(target=worker_infer, args=(0, study_ids[:half], q))
  p1 = mp.Process(target=worker_infer, args=(1, study_ids[half:], q))
  p0.start(); p1.start()
  p0.join(); p1.join()
  ```
* **Efficiency Guarantee:** Leaves $>23$ minutes of safety margin under the 30-minute Efficiency Track cap.

### Component 2: Universal Bilateral Test-Time Augmentation (TTA)
* **Mathematical Rationale:** Knee anatomy possesses bilateral symmetry across the sagittal plane. A horizontal flip maps medial $\to$ lateral, but preserves internal tissue tear patterns.
* **Application:**
  * **CoAtNet 384 & 224**: $\mathbf{x}_{\text{orig}}$ and $\mathbf{x}_{\text{flip}} = \text{torch.flip}(\mathbf{x}, [-1])$, averaged at $50/50$.
  * **ConvNeXt 2.5D**: Slices flipped horizontally before 2.5D slab formation.
  * **Expected Lift:** $+0.004\text{ to }+0.006$ Macro ROC-AUC.

### Component 3: The Quad-Arm Blend & Calibration Weights
* We discard RadNet and enforce strict dominance of the SOTA SWA Anchor on its core strengths, routing only verified sub-specialty signals:

| Finding | SWA 384 + 224 Anchor ($w_{\text{swa}}$) | ConvNeXt 2.5D Reader ($w_{\text{cnxt}}$) | DINOv3 ViT Arm ($w_{\text{dino}}$) | Specialization Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **ACL** | **0.80** | 0.10 | 0.10 | SWA anchor dominates cruciate ligaments ($0.95+$ AUC) |
| **MCL** | **0.75** | 0.15 | 0.10 | SWA anchor dominates coronal capsular ligaments |
| **Medial Meniscus** | **0.80** | 0.10 | 0.10 | SWA dual 384+224 focal crop excels at tears |
| **Lateral Meniscus** | **0.85** | 0.05 | 0.10 | SWA anchor near-perfect ($0.96+$ AUC) |
| **Baker's Cyst** | 0.40 | **0.50** | 0.10 | ConvNeXt reader dominates ($0.984$ vs $0.940$) |
| **Bone Contusion** | 0.45 | **0.45** | 0.10 | ConvNeXt excels on trabecular edema ($0.966$ vs $0.928$) |
| **Lateral OA** | 0.50 | **0.40** | 0.10 | ConvNeXt delivers $+0.079$ AUC advantage |
| **Medial OA** | **0.65** | 0.25 | 0.10 | Balanced joint-space assessment |
| **PF OA** | **0.70** | 0.20 | 0.10 | Axial patellofemoral tracking |
| **Effusion** | **0.70** | 0.05 | **0.25** | DINOv3 non-local attention on suprapatellar pouch |
| **Synovitis** | **0.65** | 0.05 | **0.30** | DINOv3 captures diffuse synovial thickening |
| **Fracture** | **0.85** | 0.10 | 0.05 | SWA anchor dominates cortical disruption |

### Component 4: Bi-Directional Kinematic Cascade Matrix
* Clinically coupled trauma and degenerative cascades:
  1. **Pivot-Shift Trauma Coupling**:
     $$\text{ACL} > 0.85 \implies \text{Contusion} \mathrel{+}= 0.04, \quad \text{Lateral Meniscus} \mathrel{+}= 0.03$$
     $$\text{Contusion} > 0.85 \implies \text{ACL} \mathrel{+}= 0.035 \quad \text{(Pivot-Shift reverse prior)}$$
  2. **Joint Distension Coupling**:
     $$\text{Effusion} > 0.85 \iff \text{Synovitis} > 0.85 \implies \text{Mutual reinforcement } +0.025$$
  3. **Degenerative Compartment Coupling**:
     $$\text{Medial OA} > 0.85 \implies \text{Medial Meniscus} \mathrel{+}= 0.03$$
     $$\text{Baker's} > 0.85 \implies \text{Effusion} \mathrel{+}= 0.025$$

### Component 5: Temperature-Scaled Latent Probit Quantiles
* Avoid arithmetic probability averaging. Project percentile ranks $R_{i,j} \in (0, 1)$ into Gaussian space:
  $$z_{i,j} = \Phi^{-1}(\text{clip}(R_{i,j}, \epsilon, 1-\epsilon))$$
* Calibrate with temperature vector $T_j$:
  $$z^{\text{cal}}_{i,j} = \frac{z_{i,j}}{T_j}$$
* Map back via logistic sigmoid:
  $$P_{i,j} = \sigma(z^{\text{cal}}_{i,j})$$
* Temperature vector:
  * High-confidence targets ($T \le 0.85$): Effusion ($0.78$), Synovitis ($0.80$), Baker's ($0.82$), Lateral Meniscus ($0.85$).
  * Rare/ambiguous targets ($T \ge 1.05$): Fracture ($1.10$), Contusion ($1.05$).

---

## 4. Execution Phases & Milestones

### Phase 1: Pure Dual-Arm SWA Baseline Recovery (Target: 0.948 – 0.950)
* **Task:** Completely remove RadNet from `submission/submission.py`.
* **Configuration:** CoAtNet 384 SWA ($75\%$) + CoAtNet 224 SWA ($25\%$) + ConvNeXt 2.5D MIL Reader with calibrated weights.
* **Verification:** Confirm score restores to $\ge 0.948+$.

### Phase 2: Dual-T4 Process Sharding & Universal Bilateral TTA (Target: 0.954 – 0.958)
* **Task:** Implement `torch.multiprocessing` across `cuda:0` and `cuda:1`.
* **Task:** Add bilateral horizontal-flip TTA to ConvNeXt Reader.
* **Verification:** Profile runtime ($<7$ minutes total), verify memory usage $<4$GB per GPU.

### Phase 3: Bi-Directional Kinematic Cascade & Temperature Probit Tuning (Target: 0.960+)
* **Task:** Implement bidirectional pivot-shift feedback and mutual joint distension coupling.
* **Task:** Refine per-class Gaussian temperatures on validation percentiles.
* **Verification:** Submit Version 18 to Kaggle.

### Phase 4: Quad-Arm DINOv3 ViT Integration (Target: 0.965 – 0.970+)
* **Task:** Mount `mattiaangeli/knee-mri-fold-weights` / DINOv3 ViT-Small.
* **Task:** Evaluate DINOv3 on suprapatellar pouch fluid series (Effusion & Synovitis).
* **Verification:** Final push to leaderboard Top 10.

---

## 5. Risk Assessment & Failsafe Guarantees

| Potential Failure Mode | Root Cause | Built-In Automated Failsafe |
| :--- | :--- | :--- |
| **GPU Out of Memory (OOM)** | Loading multiple models simultaneously on 1 GPU | Sequential execution with explicit `del model; gc.collect(); torch.cuda.empty_cache()` |
| **Auxiliary Model Exception** | Corrupt DICOM or missing sequence | `try-except` wrapper around auxiliary legs; instantly falls back to `_sota_0949.csv` |
| **Runtime Timeout (> 30 min)** | High slice count or heavy TTA | Dual-T4 sharding ($2\times$ throughput) guarantees runtime $<10$ min |
| **Format Rejection** | Missing headers or NaNs | Built-in Pre-Submission Rule Validator with automatic auto-clipping and schema alignment |
