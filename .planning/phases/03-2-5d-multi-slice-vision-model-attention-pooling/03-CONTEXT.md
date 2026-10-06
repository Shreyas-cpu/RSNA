# Phase 3 Context: 2.5D Multi-Slice Vision Model & Attention Pooling

## Overview
Phase 3 builds the core neural network architecture for the RSNA Knee Abnormality Detection pipeline. It couples a pretrained 2D slice feature extractor with a Gated Attention Multiple Instance Learning (MIL) pooling module, terminating in a 12-class multi-label classification head that outputs continuous probabilities for all target knee abnormalities.

## Decisions

- **D-10:** Modular Backbone Feature Extractor (MODEL-01) — Encapsulate `timm` pretrained models (`convnext_tiny`, `efficientnet_b0`, `resnet34d`) to map $(B, S, 3, H, W)$ slice slabs into $(B, S, D)$ spatial feature sequences via efficient $(B \cdot S)$ tensor folding.
- **D-11:** Gated Attention / MIL Sequence Pooling (MODEL-02) — Implement Ilse et al. Gated Attention pooling $\mathbf{w}^T (\tanh(\mathbf{V}\mathbf{h}) \odot \text{sigm}(\mathbf{U}\mathbf{h}))$ to dynamically weight slices showing focal pathology and collapse $(B, S, D)$ into $(B, D)$ study embeddings without losing localized diagnostic signals.
- **D-12:** 12-Class Multi-Label Prediction Head & Continuous Calibrated Probabilities (MODEL-04) — Project pooled study embeddings to 12 raw logits with sigmoid activation providing continuous probabilities strictly bounded in $[0.0, 1.0]$.
- **D-13:** Memory-Efficient Forward & AMP Support — Support PyTorch Automatic Mixed Precision (AMP fp16/bf16) and gradient checkpointing to guarantee execution on 8GB VRAM and Kaggle T4 GPUs without OOM errors.

## Key Artifacts to Produce
- `src/models/__init__.py`: Package exports.
- `src/models/backbone.py`: `SliceBackbone` class wrapping `timm`.
- `src/models/attention.py`: `GatedAttentionPool` class.
- `src/models/model.py`: `RSNAKneeModel` unifying backbone, pooling, and multi-label classifier.
- `tests/test_backbone.py`: Unit tests for feature extraction across backbone architectures.
- `tests/test_attention.py`: Unit tests for attention weight normalization and shape reduction.
- `tests/test_model.py`: Unit tests for full model forward pass, gradient flow, and AMP compatibility.
