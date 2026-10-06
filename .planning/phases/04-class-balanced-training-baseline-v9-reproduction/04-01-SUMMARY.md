# Plan 04-01: Class-Balanced Loss & Mixed-Precision AMP Trainer Summary

## Overview
Implemented dynamic positive-class weighting (`compute_pos_weights`) to mitigate extreme multi-label imbalance across 12 knee pathologies (especially rare ligaments like PCL and LCL), and built the mixed-precision `Trainer` pipeline with gradient accumulation, AdamW optimizer, and validation AUC checkpointing.

## Key Deliverables & Artifacts
1. **`src/training/__init__.py`**:
   - Initialized training package exporting `compute_pos_weights`, `create_loss_fn`, `Trainer`, `CrossValidationExperiment`, and `run_baseline_reproduction`.
2. **`src/training/loss.py`**:
   - `compute_pos_weights`:
     - Computes dynamic positive-class weight ratios $w_c = \sqrt{N_{\text{neg}} / N_{\text{pos}}}$ per pathology column.
     - Square-root dampening prevents extreme gradient explosion on rare classes.
     - Clamps weights to a customizable range (default $[1.0, 10.0]$).
   - `create_loss_fn`:
     - Instantiates `nn.BCEWithLogitsLoss` pre-configured with positive class weights.
3. **`src/training/trainer.py`**:
   - `Trainer`:
     - Automatic Mixed Precision (AMP) via `torch.amp.autocast` and `torch.amp.GradScaler('cuda')`.
     - Gradient accumulation support for effective batch sizes on constrained GPU VRAM.
     - Cosine Annealing learning rate scheduling per epoch.
     - Integration with `MetricTracker` calculating macro ROC AUC and per-class metrics on validation batches.
     - Automatic model checkpoint saving based on validation Macro AUC improvement.
4. **`tests/test_loss.py`**:
   - Verified `compute_pos_weights` assigns higher weights to rare classes and respects clip boundaries.
   - Verified backward gradient propagation through positive-weighted loss.
5. **`tests/test_trainer.py`**:
   - Verified `train_epoch` loss reduction, `val_epoch` evaluation, AMP execution, and model checkpointing.

## Verification Results
- `pytest tests/test_loss.py tests/test_trainer.py -v`: 5 passed.
- Fulfills requirements **TRAIN-01** and **TRAIN-02**.

## Decisions Addressed
- **D-01**: Class-balanced weighting and stable AMP convergence form the foundational optimization core to beat the Version 9 baseline (0.584 Public AUC).
- **D-14**: Square-root dampened positive weighting ($[1.0, 10.0]$) eliminates gradient starvation for rare classes (~3% PCL, ~4% LCL).
- **D-15**: Native AMP + gradient accumulation enables training multi-slice models efficiently on 8GB VRAM without memory spikes.
