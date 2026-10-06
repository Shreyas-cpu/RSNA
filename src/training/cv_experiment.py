"""5-Fold Cross-Validation experiment orchestrator for RSNA Knee Detection.

Coordinates multi-fold training, manages stratified group partitions,
accumulates out-of-fold (OOF) predictions, and evaluates global Macro ROC AUC
to verify reproduction and surpassing of the Version 9 baseline (0.584 Public AUC).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.data.dataset import RSNAKneeDataset, create_dataloader
from src.evaluation.metrics import (
    MetricTracker,
    compute_macro_auc,
    compute_multilabel_auc_dict,
)
from src.evaluation.split import (
    DEFAULT_TARGET_COLS,
    create_stratified_group_folds,
    verify_leak_free_split,
)
from src.models.model import RSNAKneeModel
from src.training.loss import compute_pos_weights, create_loss_fn
from src.training.trainer import Trainer


class CrossValidationExperiment:
    """Orchestrates 5-fold cross-validation and out-of-fold benchmark evaluation."""

    def __init__(
        self,
        df: pd.DataFrame,
        cache_dir: Union[str, Path],
        target_cols: Optional[List[str]] = None,
        backbone_name: str = "resnet34d",
        epochs: int = 2,
        batch_size: int = 4,
        lr: float = 1e-4,
        accumulation_steps: int = 1,
        use_amp: bool = True,
        n_splits: int = 5,
        checkpoint_dir: Union[str, Path] = "checkpoints/cv",
        seed: int = 42,
    ):
        """Initialize the CV experiment.

        Args:
            df: DataFrame containing StudyInstanceUID and labels.
            cache_dir: Path to directory containing pre-extracted slab archives.
            target_cols: List of target abnormality columns.
            backbone_name: timm vision backbone name.
            epochs: Training epochs per fold.
            batch_size: DataLoader batch size.
            lr: Learning rate for AdamW optimizer.
            accumulation_steps: Gradient accumulation steps.
            use_amp: Whether to enable Automatic Mixed Precision.
            n_splits: Number of cross-validation folds.
            checkpoint_dir: Directory to save fold checkpoints.
            seed: Random seed for fold stratification.
        """
        self.cache_dir = Path(cache_dir)
        self.target_cols = target_cols or [c for c in DEFAULT_TARGET_COLS if c in df.columns]
        if not self.target_cols:
            self.target_cols = DEFAULT_TARGET_COLS

        self.backbone_name = backbone_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.n_splits = n_splits
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

        # Ensure leak-free stratified group folds exist on df
        if "fold" not in df.columns:
            self.df = create_stratified_group_folds(
                df,
                group_col="StudyInstanceUID",
                target_cols=self.target_cols,
                n_splits=n_splits,
                seed=seed,
            )
        else:
            self.df = df.copy()

        verify_leak_free_split(self.df, fold_col="fold", group_col="StudyInstanceUID")

    def run_fold(self, fold_idx: int) -> Tuple[np.ndarray, float]:
        """Train and validate a single fold.

        Args:
            fold_idx: Integer fold index in [0, n_splits - 1].

        Returns:
            Tuple of (val_predictions array of shape [N_val, num_classes], best_macro_auc).
        """
        train_df = self.df[self.df["fold"] != fold_idx].reset_index(drop=True)
        val_df = self.df[self.df["fold"] == fold_idx].reset_index(drop=True)

        # 1. Compute training positive weights to balance classes
        pos_weights = compute_pos_weights(train_df, target_cols=self.target_cols)
        criterion = create_loss_fn(pos_weights=pos_weights)

        # 2. Datasets and DataLoaders
        train_ds = RSNAKneeDataset(
            records=train_df,
            cache_dir=self.cache_dir,
            target_cols=self.target_cols,
        )
        val_ds = RSNAKneeDataset(
            records=val_df,
            cache_dir=self.cache_dir,
            target_cols=self.target_cols,
        )

        train_loader = create_dataloader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0
        )
        val_loader = create_dataloader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        # 3. Model and Trainer
        model = RSNAKneeModel(
            backbone_name=self.backbone_name,
            num_classes=len(self.target_cols),
            pretrained=False,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-2)
        fold_ckpt_dir = self.checkpoint_dir / f"fold_{fold_idx}"

        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            use_amp=self.use_amp,
            accumulation_steps=self.accumulation_steps,
            checkpoint_dir=fold_ckpt_dir,
            class_names=self.target_cols,
        )

        # 4. Train
        history = trainer.fit(train_loader, val_loader, epochs=self.epochs)

        # 5. Extract validation predictions
        val_preds_list = []
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                slabs = batch["slabs"].to(trainer.device)
                probs = model.predict_proba(slabs)
                val_preds_list.append(probs.cpu().numpy())

        val_preds = np.concatenate(val_preds_list, axis=0) if val_preds_list else np.empty((0, len(self.target_cols)))
        return val_preds, float(history.get("best_macro_auc", 0.0))

    def run_all_folds(self) -> Dict[str, Any]:
        """Execute cross-validation across all n_splits folds and aggregate OOF metrics.

        Returns:
            Dictionary containing 'oof_macro_auc', 'fold_aucs', 'per_class', and 'report'.
        """
        oof_preds = np.zeros((len(self.df), len(self.target_cols)), dtype=float)
        fold_aucs = []

        for f in range(self.n_splits):
            val_mask = (self.df["fold"] == f).to_numpy()
            val_preds, best_auc = self.run_fold(f)
            oof_preds[val_mask] = val_preds
            fold_aucs.append(best_auc)

        # Compute global OOF metrics
        oof_trues = self.df[self.target_cols].to_numpy(dtype=float)
        oof_macro_auc = compute_macro_auc(oof_trues, oof_preds, class_names=self.target_cols)
        per_class = compute_multilabel_auc_dict(oof_trues, oof_preds, class_names=self.target_cols)

        tracker = MetricTracker(class_names=self.target_cols)
        tracker.update(oof_trues, oof_preds)
        report = tracker.generate_summary_report()

        return {
            "oof_macro_auc": oof_macro_auc,
            "fold_aucs": fold_aucs,
            "per_class": per_class,
            "oof_predictions": oof_preds,
            "report": report,
        }


def run_baseline_reproduction(
    df: pd.DataFrame,
    cache_dir: Union[str, Path],
    backbone_name: str = "resnet34d",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: Optional[str] = None,
    checkpoint_dir: Union[str, Path] = "checkpoints/cv",
    use_amp: bool = True,
) -> Dict[str, Any]:
    """High-level function reproducing the 5-fold CV baseline experiment."""
    exp = CrossValidationExperiment(
        df=df,
        cache_dir=cache_dir,
        backbone_name=backbone_name,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        checkpoint_dir=checkpoint_dir,
        use_amp=use_amp,
    )
    return exp.run_all_folds()

