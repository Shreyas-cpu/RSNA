# Phase 3 Research: 2.5D Vision Backbones & Gated Attention Pooling

## 1. Multiple Instance Learning (MIL) for Knee MRI

### Why Not 3D CNNs?
- Standard 3D dense volumetric networks (e.g. 3D ResNet, UNet, Med3D) require volumetric convolutions across the entire $(D, H, W)$ grid.
- Knee MRI series exhibit non-isotropic resolution (slice thickness typically 3-4mm, in-plane resolution ~0.4-0.6mm).
- High VRAM usage limits batch size to 1 or 2, severely impeding BatchNorm/LayerNorm stability and iteration velocity on standard hardware or Kaggle T4 GPUs.

### Why 2.5D Slabs + MIL Attention?
- A knee MRI study consists of $S$ slices. Pathologies such as an ACL tear or meniscus tear are typically focal: they are distinctly visible on only 2 to 4 slices out of 25-35 slices in a sequence.
- Simple Mean or Max pooling across slices fails:
  - Mean pooling dilutes the localized tear signal with 20+ normal slices.
  - Max pooling is noisy and lacks differentiable gradient weighting across contextual slices.
- **Gated Attention Mechanism (Ilse et al., ICML 2018)**:
  Let $\mathbf{H} = \{\mathbf{h}_1, \dots, \mathbf{h}_S\} \in \mathbb{R}^{S \times D}$ be the slice embeddings produced by the 2D backbone.
  Attention weights $\mathbf{a} \in \mathbb{R}^S$ are calculated as:
  $$\mathbf{a} = \text{softmax}\left(\mathbf{w}^T \left(\tanh(\mathbf{V} \mathbf{H}^T) \odot \sigma(\mathbf{U} \mathbf{H}^T)\right)\right)$$
  Where:
  - $\mathbf{V} \in \mathbb{R}^{L \times D}$ and $\mathbf{U} \in \mathbb{R}^{L \times D}$ are learnable linear projections.
  - $\odot$ is element-wise multiplication.
  - $\sigma(\cdot)$ is the sigmoid gating activation, controlling which features contribute to the attention score.
  - $\mathbf{w} \in \mathbb{R}^{L \times 1}$ is the final projection vector.
  - The pooled study representation is $\mathbf{z} = \sum_{s=1}^S a_s \mathbf{h}_s \in \mathbb{R}^D$.

## 2. 2D Vision Backbone Selection in `timm`

1. **`convnext_tiny`**: Modern depthwise separable convolution architecture with $7 \times 7$ kernels, LayerNorm, and GELU activations. Exceptional inductive bias for medical textures with low latency. Feature dimension: 768.
2. **`efficientnet_b0` / `efficientnet_b2`**: Standard, proven, lightweight compound scaling baseline. Excellent for fast iterations and efficiency track. Feature dimension: 1280 (b0) / 1408 (b2).
3. **`resnet34d`**: Robust residual network with anti-aliased downsampling, fast inference, and stable gradient flow. Feature dimension: 512.

## 3. Forward Pass Tensor Reshaping & VRAM Optimization

```python
# Input tensor shape: (B, S, 3, H, W) where B = batch size, S = slabs count
B, S, C, H, W = x.shape
x_flat = x.view(B * S, C, H, W)  # Fold slabs into batch dimension

features = backbone(x_flat)       # Shape: (B * S, D)
features = features.view(B, S, -1) # Unfold back to sequence: (B, S, D)

pooled, attn_weights = attention_pool(features) # Shape: (B, D)
logits = classifier(pooled)       # Shape: (B, 12)
```

### PyTorch AMP Compatibility
- Forward pass supports `torch.cuda.amp.autocast()` with `fp16` / `bf16`.
- Numerically stable softmax: subtract max before exponentiation or use `torch.softmax(dim=-1)` natively.
