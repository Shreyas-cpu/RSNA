# Phase 2 Research: Multi-Label Group Stratification & Macro ROC AUC

## 1. Multi-Label Stratified Group Splitting

### The Problem
We have $N$ studies, each associated with:
1. A unique group identifier (`StudyInstanceUID`).
2. A binary label vector $\mathbf{y} \in \{0, 1\}^{12}$ representing 12 knee abnormalities (ACL, PCL, MCL, LCL, Medial Meniscus, Lateral Meniscus, Patellar Tendon, Quadriceps Tendon, Cartilage Femur, Cartilage Tibia, Cartilage Patella, Joint Effusion).

Standard `sklearn.model_selection.GroupKFold` guarantees no group overlap, but does not balance multi-label targets, resulting in folds where rare classes (e.g., PCL tears with ~3% prevalence) may have zero positive instances.
Standard `StratifiedKFold` only handles single-label classification.

### The Algorithm: Iterative Stratified Group Partitioning
To achieve both:
1. Aggregate multi-label targets at the study level: $\mathbf{y}_{\text{study}} = \max_{\text{series}} \mathbf{y}_{\text{series}}$ (each study has 1 vector).
2. Rank labels by rarity (frequency of positive instances in ascending order).
3. Assign studies containing the rarest labels first to the fold currently having the lowest count of that label, balancing the total fold sizes.
4. Fallback/hybrid approach: Combine multi-label indicators into a binned representation or use `MultilabelStratifiedKFold` on unique studies, assigning fold IDs directly to `StudyInstanceUID`.

Because each `StudyInstanceUID` contains exactly one ground-truth label vector in RSNA training tables, performing multi-label stratification directly on unique `StudyInstanceUID` rows guarantees:
- **100% leak-free grouping**: all slices/series belonging to a `StudyInstanceUID` inherit the study's fold.
- **Optimal class balance**: multi-label distributions are stratified across all $K$ folds.

## 2. Macro-Averaged ROC AUC Metric

### Formula
For $C = 12$ classes:
$$\text{Macro AUC} = \frac{1}{C} \sum_{c=1}^C \text{AUC}_c$$
Where $\text{AUC}_c$ is calculated via trapezoidal integration of the ROC curve:
$$\text{AUC}_c = \int_0^1 \text{TPR}_c(\text{FPR}_c) \, d\text{FPR}_c$$

### Edge Cases & Defensive Engineering
1. **Single-class presence in small batch**: If $y_{\text{true}, c}$ has only 0s or only 1s (e.g. during a 1-batch test step), `roc_auc_score` raises `ValueError: Only one class present in y_true`.
   - Solution: Return `float('nan')` or 0.5 for that specific class, and compute macro average only over classes with both positive and negative samples present, logging a notice.
2. **Probability clipping**: Probabilities must remain continuous float values in $[0.0, 1.0]$. Never apply `> 0.5` thresholding, which destroys rank ordering and drops AUC to 0.5.
3. **Missing label support**: If weak labels or NaN labels are present, filter out NaNs per class before calculating the class AUC.

## 3. Reference Implementation Architecture
- `src/evaluation/split.py`:
  - `create_stratified_group_folds(df, group_col='StudyInstanceUID', target_cols=TARGET_COLS, n_splits=5, seed=42)`
  - `verify_leak_free_split(df, fold_col='fold', group_col='StudyInstanceUID') -> bool`
- `src/evaluation/metrics.py`:
  - `compute_multilabel_auc_dict(y_true, y_pred, class_names=TARGET_COLS) -> Dict[str, float]`
  - `compute_macro_auc(y_true, y_pred) -> float`
  - `MetricTracker`: Running accumulator across batches for validation loops.
