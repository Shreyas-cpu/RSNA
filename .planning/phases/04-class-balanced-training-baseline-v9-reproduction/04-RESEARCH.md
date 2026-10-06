# Phase 4 Research: Multi-Label Class-Balanced Training & Convergence

## 1. Multi-Label Imbalance & Loss Formulations

### The Challenge in Knee MRI
In the RSNA Knee dataset:
- Certain pathologies have low prevalence (PCL tears ~3%, LCL tears ~4%, patellar tendon abnormalities ~5%).
- Common findings (joint effusion, degenerative medial meniscus changes) have prevalence ~25-35%.
- If standard unweighted binary cross-entropy (BCE) is applied:
  $$\mathcal{L} = -\frac{1}{C}\sum_{c=1}^C \left[ y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
  The negative terms $(1 - y_c)$ heavily dominate the gradient:
  $$\frac{\partial \mathcal{L}}{\partial z_c} = \sigma(z_c) - y_c$$
  For rare classes where $y_c = 0$ for 97% of samples, gradients push logits $z_c \to -\infty$, resulting in zero predictions and collapsing the class AUC to ~0.50.

### Positive-Class Weighting Solution
PyTorch's `nn.BCEWithLogitsLoss` natively accepts `pos_weight`:
$$\mathcal{L}_c = - \left[ w_c \cdot y_c \log \sigma(z_c) + (1 - y_c) \log (1 - \sigma(z_c)) \right]$$
To avoid instability from extreme ratios (e.g. $w_c = 30.0$ producing massive gradient explosions on false negatives), we use square-root dampened weighting clamped to safe bounds:
$$w_c = \text{clip}\left(\sqrt{\frac{N_{\text{neg}, c}}{N_{\text{pos}, c}}}, 1.0, 10.0\right)$$
This balances the gradient magnitude without destabilizing mixed-precision training.

## 2. Mixed Precision & Memory Optimization

### PyTorch AMP Workflow
```python
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

with torch.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps

scaler.scale(loss).backward()

if (step + 1) % accumulation_steps == 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Learning Rate Schedule
- **Warmup**: Linear warmup for 1-2 epochs prevents destabilizing pretrained vision weights with large initial gradients from the randomly initialized attention pooling and head layers.
- **Cosine Annealing**: Decays learning rate smoothly to $\eta_{\text{min}} = 10^{-6}$ over remaining epochs, enabling convergence to flatter minima with higher validation Macro AUC.

## 3. Checkpoint Strategy
- Save `best_model.pt` evaluated strictly on `validation_macro_auc`.
- Log per-class AUC at the end of each validation epoch to verify that rare classes are improving above 0.55.
