# Phase 6 Research: Multi-Plane Feature Fusion & Joint Silver Training

## 1. Multi-Plane Clinical Imaging Mechanics

Magnetic Resonance Imaging (MRI) of the knee relies on orthogonal and oblique series planes to evaluate distinct anatomical structures:

```text
                  ┌───────────────────────┐
                  │    Knee MRI Study     │
                  └───────────┬───────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │ Sagittal  │       │  Coronal  │       │   Axial   │
    │  Series   │       │  Series   │       │  Series   │
    └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
          │                   │                   │
   • ACL, PCL          • MCL, LCL          • Patellofemoral
   • Meniscal Horns    • Meniscal Bodies   • Patellar Cartilage
   • Extensor Tendons  • Tibiofemoral OA   • Joint Effusion
          │                   │                   │
          ▼                   ▼                   ▼
    ┌───────────┐       ┌───────────┐       ┌───────────┐
    │MIL Pooling│       │MIL Pooling│       │MIL Pooling│
    └─────┬─────┘       └─────┬─────┘       └─────┬─────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  Multi-Plane Fusion   │
                  │(Cross-Plane Attention)│
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │  12-Class Pred Head   │
                  └───────────────────────┘
```

Single-plane models suffer from substantial diagnostic blind spots. A multi-plane architecture fusing slice sequences from all three orientations creates a holistic 3D joint representation without the computational explosion of dense 3D convolutions.

---

## 2. Multi-Plane Fusion Formulation

### Stage 1: Plane-Specific Feature Extraction & MIL Pooling
For each plane $p \in \{\text{sagittal}, \text{coronal}, \text{axial}\}$:
1. Input tensor $X_p \in \mathbb{R}^{B \times S_p \times 3 \times H \times W}$ is folded and processed through `SliceBackbone`:
   $$F_p = \text{SliceBackbone}(X_p) \in \mathbb{R}^{B \times S_p \times D}$$
2. Plane-specific `GatedAttentionPool` aggregates slices into a fixed-dimensional plane representation:
   $$h_p = \text{GatedAttentionPool}_p(F_p) \in \mathbb{R}^{B \times D}$$

### Stage 2: Cross-Plane Fusion
Plane representations are stacked:
$$H = [h_{\text{sag}}, h_{\text{cor}}, h_{\text{ax}}] \in \mathbb{R}^{B \times 3 \times D}$$
We apply a multi-plane fusion layer combining:
1. **Learnable Plane Positional Embeddings**: $E_{\text{plane}} \in \mathbb{R}^{3 \times D}$ to differentiate spatial orientation.
2. **Cross-Plane Multi-Head Self-Attention**:
   $$\tilde{H} = \text{MultiheadAttention}(H + E_{\text{plane}})$$
3. **Gated Concatenation & Projection**:
   $$h_{\text{fused}} = \text{LayerNorm}(\text{Linear}([h_{\text{sag}}; h_{\text{cor}}; h_{\text{ax}}])) \in \mathbb{R}^{B \times D}$$
4. **Missing Plane Masking**: If a series is absent, its attention mask zero-weights its contribution while learned bias weights compensate smoothly.

---

## 3. Joint Training with Sample-Weighted Loss

To leverage the ~3–5x expanded dataset from Phase 5 without noisy weak labels corrupting ground-truth weights:

### Loss Function with Dual Weighting
For each sample $i$ with weight $w_i \in [0.1, 1.0]$ and ground truth / pseudo-label $y_{ic} \in [0.0, 1.0]$:
$$\mathcal{L}_i = \sum_{c=1}^{12} \left[ - \text{pos\_weight}_c \cdot y_{ic} \log \sigma(z_{ic}) - (1 - y_{ic}) \log(1 - \sigma(z_{ic})) \right]$$
$$\mathcal{L}_{\text{batch}} = \frac{\sum_{i=1}^B w_i \mathcal{L}_i}{\sum_{i=1}^B w_i}$$

- **Gold Samples**: $w_i = 1.0$ (hard binary targets $y \in \{0, 1\}$).
- **Silver Samples**: $w_i \in [0.1, 1.0]$ based on certainty score $w_i = \frac{1}{12}\sum_{c=1}^{12} 2 \cdot |p_{ic} - 0.5|$.
- **Curriculum**: Training jointly with sample weights stabilizes representations across common and rare pathologies alike.
