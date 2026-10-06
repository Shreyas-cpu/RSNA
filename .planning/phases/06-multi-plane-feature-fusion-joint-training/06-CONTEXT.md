# Phase 6 Context: Multi-Plane Feature Fusion & Joint Training

## Overview
Phase 6 implements the multi-plane fusion architecture (`MultiPlaneModel`) and joint silver + gold training curriculum. In knee MRI diagnosis, no single anatomical plane is sufficient: Sagittal is optimal for cruciate ligaments and menisci; Coronal is essential for collateral ligaments and tibiofemoral cartilage/meniscal bodies; and Axial is vital for the patellofemoral compartment, patellar cartilage, and effusion. Fusing all three planes and training jointly on human gold labels and Phase 5 silver pseudo-labels provides the critical performance leap needed to decisively surpass the 0.584 baseline (> 0.65 CV Macro AUC).

## Decisions

- **D-19:** Multi-Plane Attention Fusion Architecture (MODEL-03) — Implement `MultiPlaneFusion` combining Sagittal, Coronal, and Axial series representations. Slices from each plane are extracted via `SliceBackbone`, aggregated into plane-level embeddings via plane-specific `GatedAttentionPool`, and fused via a cross-plane multi-head attention / gated concatenation layer into a unified study embedding ($D_{\text{study}}$), natively handling missing series or variable slice counts.
- **D-20:** Certainty-Weighted Joint Curriculum Training (TRAIN-03) — Implement joint training supporting both gold ground-truth labels and silver NLP pseudo-labels generated in Phase 5. The loss function integrates sample certainty weighting $w_i \in [0.1, 1.0]$ alongside positive-class weighting $\text{pos\_weight}_c$, preventing noisy weak labels from dominating gradients while expanding effective training volume by 3–5x.

## Key Artifacts to Produce
- `src/models/multi_plane.py`: `MultiPlaneFusion` module and `MultiPlaneKneeModel` unifying 3-plane extraction, attention pooling, cross-plane fusion, and 12-class prediction head.
- `src/training/joint_trainer.py`: `JointTrainer` supporting sample-weighted multi-label BCE loss and joint curriculum scheduling.
- `tests/test_multi_plane.py`: Unit and integration tests for multi-plane forward pass, plane masking/missing planes, gradient flow, and continuous probability outputs.
- `tests/test_joint_trainer.py`: Unit tests for sample-weighted loss, gold + silver batch training, and validation metric tracking.
