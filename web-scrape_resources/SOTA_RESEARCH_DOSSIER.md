# 📚 RSNA Knee Abnormality Detection: Grandmaster Web-Scraped Research & SOTA Dossier

> **Compiled For**: Lead Model Architect & Antigravity Reasoning Agent  
> **Source Repository**: `/home/promethious/projects/RSNA/web-scrape_resources/`  
> **Purpose**: SOTA Architecture, Training Tricks, Pre-Processing, Loss Optimization, and Ensembling from Gold-Medal Solutions and Peer-Reviewed Literature.  
> **Date**: October 14, 2026  

---

## 1. Landmark Architecture Reference: Stanford MRNet (PLOS Medicine / ArXiv)
* **Paper**: *Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet* (Bien et al., Stanford ML Group, PLOS Medicine 2018).
* **Key Architecture**:
  - Knee MRI is decomposed into 3 independent sequence views: **Sagittal, Coronal, and Axial**.
  - A feature backbone (AlexNet / ResNet / ConvNeXt) extracts a slice-level representation: $\mathbf{f}_s \in \mathbb{R}^{D}$.
  - A global slice max-pooling or attention-pooling operation aggregates slice features into a study-level vector: $\mathbf{F} = \text{Pool}(\{\mathbf{f}_1, \dots, \mathbf{f}_S\})$.
  - Logistic regression / MLP head predicts abnormality.
* **Winning Extension (Triple-MRNet & 2.5D Multi-Plane)**:
  - Feeding adjacent slices as 3-channel RGB inputs ($s-1, s, s+1$) allows standard 2D backbones to learn through-plane 3D spatial continuity without requiring heavy 3D convolution kernels.
  - Multi-Plane Attention Fusion concatenates representations across all available anatomical planes before classification.

---

## 2. Medical Foundation Vision Models: RadImageNet (Radiology 2022)
* **Paper & Code**: *RadImageNet: An Open Radiologic Deep Learning Research Dataset for Effective Transfer Learning* (Mei et al., Radiology 2022; GitHub: `BMEII-AI/RadImageNet`).
* **Why It Beats ImageNet on Knee MRI**:
  - Pretrained on **1.35 million medical imaging scans** across CT, MRI, and Ultrasound covering 11 anatomical regions (including musculoskeletal/knee).
  - Natural ImageNet weights learn high-frequency RGB color textures (dogs, cars, flowers) that do not translate well to grayscale radiological contrast, Hounsfield units, and T1/T2 relaxation times.
  - On knee MRI tasks (specifically **ACL tears and Meniscal tears**), fine-tuning RadImageNet-pretrained ResNet-50 / DenseNet-121 achieves **+0.03 to +0.06 higher AUC** compared to vanilla ImageNet baselines with significantly faster convergence.
* **Our Project Asset**: We already have `models/radnet/ResNet50.pt` + `v52_e11_heads.pt` ready for integration as our Leg 3 orthogonal backbone!

---

## 3. Loss Functions for Extreme Imbalanced Multi-Label Classification: Asymmetric Loss (ASL)
* **Paper & Code**: *Asymmetric Loss For Multi-Label Classification* (Ridnik et al., Alibaba-MIIL, ICCV 2021; GitHub: `Alibaba-MIIL/ASL`).
* **Why Standard BCE & Focal Loss Degrade Macro AUC**:
  - In our 12-target knee task, most targets have positive prevalence of only 5% to 20% (e.g., Fractures, Lateral OA, Synovitis).
  - In standard Binary Cross-Entropy (BCE), easy negative gradients overwhelm the scarce positive gradients, pulling predicted probabilities toward zero and flattening rank discrimination.
  - Focal Loss downweights easy examples, but treats positive and negative samples symmetrically.
* **How ASL Solves This**:
  $$\mathcal{L}_{\text{ASL}} = - y \cdot (1 - p_m)^{\gamma_+} \log(p_m) - (1 - y) \cdot (p_m)^{\gamma_-} \log(1 - p_m)$$
  where $p_m = \max(p - m, 0)$ is the probability margin shift.
  - **Asymmetric Focusing**: Setting $\gamma_- = 4$ and $\gamma_+ = 0$ aggressively suppresses easy negatives while keeping full gradient pressure on positive findings.
  - **Probability Margin Shifting ($m \approx 0.05$)**: Hard-discards background negatives whose predicted probability is below $m$, preventing background noise from polluting gradients.
  - Models optimized with ASL yield consistently higher **Macro ROC-AUC and mAP** on imbalanced radiological targets.

---

## 4. Kaggle RSNA Lumbar Spine (2024) 1st Place Winning Architecture
* **Winning Strategy (Zhengru Li, Sergey Bryansky, etc.)**:
  1. **Two-Stage Detection/Classification Pipeline**: First stage localizes the coordinates/slices of the joint space; second stage classifies pathology severity on zoomed crops.
  2. **Attention-based Multiple Instance Learning (MIL)**: Rather than simple max-pooling or mean-pooling across slices, an attention mechanism:
     $$\mathbf{a} = \text{softmax}(\mathbf{w}^T \tanh(\mathbf{V} \mathbf{h}^T))$$
     dynamically weights the most pathologically relevant slices (e.g. focusing on the exact 2-3 slices showing an ACL tear while ignoring normal outer slices).
  3. **Bi-Directional Sequence Modeling (Bi-LSTM / 1D Transformer)**: Placed after the 2D feature extractor across the slice dimension, capturing anatomical transitions across the joint.

---

## 5. Reverse-Engineered Top 10 Tricks from Current RSNA Knee Competitors
From monitoring discussions by `ilyakolb`, `goodpjw2008`, `wassimdobbi`, and `joinT fusion`:

1. **Clinical View Prior Masking (`SLOT_PRIORS`)**:
   - Enforces medical ground truths across the 6 anatomical slots.
   - Suppresses cross-plane hallucinations (e.g. Baker's cyst only on Sagittal; MCL only on Coronal; PF OA on Axial/Sagittal).
2. **Dual-Resolution Stochastic Weight Averaging (SWA)**:
   - CoAtNet-2 384 (global context) blended 75/25 with CoAtNet-2 224 Focal Crop (localized articular surface detail).
   - SWA weights average multiple training epochs, finding flat loss minima that do not overfit to public leaderboard test samples.
3. **Latent Probit Quantile Geometric Blending**:
   - Normal CDF inverse transformation ($\Phi^{-1}$ / `ndtri`) maps percentile ranks into a Gaussian latent space, preserving discrimination in distribution tails far better than linear rank averaging.
4. **Anatomical Bilateral TTA (Horizontal Mirroring)**:
   - Averages $p(x)$ and $p(\text{flip}(x, \text{axis}=-1))$, exploiting left/right knee symmetry for $+0.005$ to $+0.007$ AUC.
5. **Efficiency Track Sequential Garbage Collection**:
   - Sequential model load, inference, and immediate `del model; gc.collect(); torch.cuda.empty_cache()` guarantees $< 4.5$ GB VRAM per GPU and $< 16$ min runtime on Kaggle 2xT4 GPUs.
