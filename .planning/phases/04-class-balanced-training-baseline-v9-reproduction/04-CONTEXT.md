# Phase 4 Context: Class-Balanced Training & Baseline V9 Reproduction

## Overview
Phase 4 builds the full training engine that trains `RSNAKneeModel` across the 12 target classes. It implements class-imbalance mitigation via positive-weighted multi-label loss, a mixed-precision (PyTorch AMP) training loop with gradient accumulation, cosine annealing scheduling, and checkpointing, and establishes the local experimental harness to surpass the Version 9 baseline score (0.584 Public AUC).

## Decisions

- **D-14:** Class-Balanced Positive Weighting (TRAIN-01) — Mitigate severe class imbalance (positive rates ranging from 3% to 30%) by computing dynamic per-class positive weights $w_c = \text{clip}\left(\sqrt{\frac{N_{\text{neg}}}{N_{\text{pos}}}}, 1.0, 10.0\right)$ in `BCEWithLogitsLoss(pos_weight=...)` to prevent gradient starvation on rare ligament pathologies.
- **D-15:** Mixed-Precision AMP Training with Gradient Accumulation (TRAIN-02) — Implement `Trainer` utilizing `torch.cuda.amp.GradScaler`, AdamW optimizer, Cosine Annealing learning rate schedule with warmup, and gradient accumulation steps to maintain effective batch sizes of 16-32 without exceeding 8GB VRAM.
- **D-16:** Baseline V9 Reproduction & Metric Checkpoint Gate (EVAL-03) — Save model checkpoints strictly based on validation Macro ROC AUC, tracking per-class convergence to ensure local cross-validation systematically exceeds the 0.584 Public AUC baseline.

## Key Artifacts to Produce
- `src/training/__init__.py`: Package exports.
- `src/training/loss.py`: `compute_pos_weights` and `create_loss_fn` with multi-label positive weighting.
- `src/training/trainer.py`: `Trainer` class managing training/validation epochs, AMP scaling, accumulation, and checkpoint saving.
- `src/training/cv_experiment.py`: `CrossValidationExperiment` orchestrating 5-fold training and aggregating out-of-fold predictions.
- `tests/test_loss.py`: Unit tests for positive weighting, loss calculation, and gradient behavior.
- `tests/test_trainer.py`: Unit tests for training epoch, validation loop, gradient accumulation, and checkpoint management.
- `tests/test_cv_experiment.py`: Integration test verifying multi-fold execution and Macro AUC convergence.
