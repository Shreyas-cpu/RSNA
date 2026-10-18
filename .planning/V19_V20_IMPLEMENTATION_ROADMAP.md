# GSD Implementation Roadmap: V19 & V20

## Mission Objective
Drive the RSNA Knee Abnormality Detection Public Macro ROC-AUC from **0.943** past **0.950+** up to **0.965+**, while strictly adhering to local-only execution, $<30$ min runtime on Kaggle dual-T4 GPUs, and zero submission failures.

---

## 1. Post-Mortem of Submissions So Far

| Version | LB Score | Key Architecture / Techniques | Why it Scored This Value |
| :---: | :---: | :--- | :--- |
| **V12** | **0.943** | Dual-Arm SWA CoAtNet (384 + 224) + ConvNeXt Reader | Strong CoAtNet anchor with basic ConvNeXt complement. |
| **V15** | **0.943** | SWA CoAtNet + ConvNeXt MIL + Temp Probit + Co-Occurrence | Verified slot priors; probit preserved extreme percentiles. |
| **V17** | **0.942** | V15 + ConvNeXt Bilateral TTA (Arithmetic Mean) | Flip-averaging diluted focal tear signals on edge slices. |
| **V18** | **0.939** | Tri-Family SWA + ConvNeXt TTA + DINOv3 ViT (Static $w_j$) | Fixed $25\%$ DINO weight across all studies caused negative transfer on missing/fallback slots without correlation gating. |

---

## 2. Version 19: Grandmaster Anchor + Adaptive Consensus + Target-Specific TTA

### Objective
Restore the rock-solid **0.943 baseline** as the primary anchor ($85\%$) and inject ConvNeXt and DINOv3 strictly through **Adaptive Consensus Gating** ($\alpha \in [0.10, 0.15]$) and **Target-Specific Window Pooling**. Expected score: **0.952 – 0.958**.

### Architectural Plan:
1. **Target-Specific TTA Pooling (`PUBLIC_FRONTIER_TARGET_POOL`)**:
   - Instead of arithmetic mean $\frac{p_{\text{orig}} + p_{\text{flip}}}{2}$:
     - **Focal Tears** (`Fracture`, `Contusion`, `Medial Meniscus`, `Lateral Meniscus`, `Baker's`): $\max(p_{\text{orig}}, p_{\text{flip}})$.
     - **Cruciate Ligaments** (`ACL`, `MCL`): $\text{Top-2 Mean}$.
     - **Diffuse Joint Volume** (`Effusion`, `Synovitis`, `OA`): $\text{Mean}$.
2. **Anchor Dominance Guarantee**:
   - CoAtNet Dual-Anchor (384 + 224) holds $85\%$ base weight ($1 - \alpha$).
3. **Adaptive Consensus Gating ($\alpha_{\text{adaptive}}$)**:
   - For each finding $t$, compute rank correlation $\rho_t = \text{corr}(\text{rank}_{\text{anchor}}[:, t], \text{rank}_{\text{aux}}[:, t])$.
   - When $\rho_t > 0.65$ (models agree), permit $\alpha_t \le 0.15$ injection for variance reduction.
   - When $\rho_t \le 0.40$ (models diverge), throttle $\alpha_t \to 0.05$ to prevent corruption.
4. **Instant Commit**:
   - Commit to branch `v19-adaptive-blend` and push to remote.

---

## 3. Version 20: Dual-GPU Sharded Pipeline + Protocol Calibration Matrix

### Objective
Maximize performance and inference throughput, targeting **0.960 – 0.965+**.

### Architectural Plan:
1. **Study Protocol Matrix Calibration**:
   - Extract sequence metadata counts from `test_series.csv` (`n_series`, `n_sag`, `n_cor`, `n_ax`, `n_fluid`, `n_nofs`).
   - Calibrate predictions using protocol-conditioned logistic intercepts for structural findings.
2. **Bi-Directional Trauma Matrix (Pivot Shift & O'Donoghue's Triad)**:
   - Refined Bayesian prior linking ACL tears with Lateral Meniscus tears, bone contusions (lateral femoral condyle), and MCL sprains.
3. **Dual-T4 Sharding (`torch.multiprocessing`)**:
   - Shard test set 50/50 across GPU 0 and GPU 1 to cut total runtime under 45s.
4. **Instant Commit**:
   - Commit to branch `v20-protocol-calibrated` and push to remote.
