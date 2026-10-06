"""Joint Gold + Silver Training Engine with Sample-Weighted Multi-Label Loss.

Trains multi-plane vision models on gold ground truth and silver pseudo-labels,
weighting sample contributions by certainty to prevent label noise degradation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.evaluation.metrics import MetricTracker
from src.evaluation.split import DEFAULT_TARGET_COLS


class SampleWeightedBCEWithLogitsLoss(nn.Module):
    """Multi-label BCE loss with dual positive-class and per-sample certainty weighting."""

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        eps: float = 1e-7,
    ):
        """Initialize SampleWeightedBCEWithLogitsLoss.

        Args:
            pos_weight: Optional 1D tensor of shape (num_classes,) positive class weights.
            eps: Epsilon denominator guard for weight normalization.
        """
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute sample-weighted binary cross-entropy loss.

        Args:
            logits: Predicted logits tensor of shape (B, C).
            targets: Target labels (binary or pseudo-probabilities) of shape (B, C).
            sample_weights: Optional 1D or 2D tensor of shape (B,) or (B, 1) in [0.0, 1.0].

        Returns:
            Scalar loss tensor.
        """
        device = logits.device
        pw = self.pos_weight.to(device) if self.pos_weight is not None else None

        # Element-wise BCE loss: shape (B, C)
        raw_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )

        # Average over classes for each sample: shape (B,)
        per_sample_loss = raw_bce.mean(dim=-1)

        if sample_weights is None:
            return per_sample_loss.mean()

        w = sample_weights.view(-1).to(device=device, dtype=logits.dtype)
        weighted_loss = per_sample_loss * w
        weight_sum = w.sum()

        if weight_sum > 0:
            return weighted_loss.sum() / (weight_sum + self.eps)
        return per_sample_loss.mean()


class JointTrainer:
    """Trainer executing curriculum learning over combined gold and silver datasets."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[SampleWeightedBCEWithLogitsLoss] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Union[str, Path] = "checkpoints/joint",
        class_names: Optional[List[str]] = None,
    ):
        """Initialize JointTrainer.

        Args:
            model: MultiPlaneKneeModel or similar multi-plane architecture.
            optimizer: Optimizer instance.
            criterion: SampleWeightedBCEWithLogitsLoss instance.
            scheduler: Learning rate scheduler.
            device: 'cuda' or 'cpu'. Auto-detected if None.
            use_amp: Whether to enable Automatic Mixed Precision.
            accumulation_steps: Gradient accumulation steps.
            max_grad_norm: Gradient norm clipping threshold.
            checkpoint_dir: Directory to save best model checkpoints.
            class_names: Target class names for MetricTracker.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-2
        )
        self.criterion = (criterion or SampleWeightedBCEWithLogitsLoss()).to(self.device)
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type in ("cuda", "cpu"))
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or DEFAULT_TARGET_COLS

        # AMP GradScaler
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.use_amp and self.device.type == "cuda")
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.use_amp and self.device.type == "cuda")
            )

        self.best_macro_auc = -1.0
        self.best_checkpoint_path: Optional[Path] = None

    def _prepare_batch(self, batch: Any) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Normalize batch into (planes_dict, targets, sample_weights)."""
        if isinstance(batch, (list, tuple)):
            # Form: (plane_dict, targets, [weights])
            plane_dict = batch[0]
            targets = batch[1]
            weights = batch[2] if len(batch) > 2 else None
        elif isinstance(batch, dict):
            targets = batch["targets"]
            weights = batch.get("sample_weight", None)
            if "planes" in batch:
                plane_dict = batch["planes"]
            else:
                # Direct dictionary of planes
                plane_dict = {k: v for k, v in batch.items() if k not in ("targets", "sample_weight", "study_id")}
        else:
            raise ValueError(f"Unrecognized batch format: {type(batch)}")

        # Transfer tensors to device
        device_planes = {k: v.to(self.device) for k, v in plane_dict.items() if isinstance(v, torch.Tensor)}
        device_targets = targets.to(self.device, dtype=torch.float32)
        device_weights = weights.to(self.device, dtype=torch.float32) if weights is not None else None

        return device_planes, device_targets, device_weights

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        n_batches = len(train_loader)
        self.optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            planes, targets, weights = self._prepare_batch(batch)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(planes)
                loss = self.criterion(logits, targets, sample_weights=weights)
                loss_scaled = loss / self.accumulation_steps

            if self.use_amp and self.device.type == "cuda":
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == n_batches:
                if self.use_amp and self.device.type == "cuda":
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item()

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(1, n_batches)

    @torch.no_grad()
    def val_epoch(self, val_loader: DataLoader) -> Tuple[float, float, Dict[str, Any]]:
        """Evaluate validation epoch and calculate Macro ROC AUC via MetricTracker."""
        self.model.eval()
        total_loss = 0.0
        n_batches = len(val_loader)
        tracker = MetricTracker(class_names=self.class_names)

        for batch in val_loader:
            planes, targets, weights = self._prepare_batch(batch)

            with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                logits = self.model(planes)
                loss = self.criterion(logits, targets, sample_weights=weights)

            total_loss += loss.item()
            probs = torch.sigmoid(logits)

            tracker.update(
                y_true=targets.detach().cpu().numpy(),
                y_pred=probs.detach().cpu().numpy(),
            )

        val_loss = total_loss / max(1, n_batches)
        summary = tracker.compute()
        macro_auc = summary["macro_auc"]

        return val_loss, macro_auc, summary

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
        checkpoint_name: str = "best_multiplane_model.pt",
    ) -> Dict[str, List[float]]:
        """Fit model across epochs, logging metrics and saving the best checkpoint."""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auc": [],
        }

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_macro_auc, _ = self.val_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auc"].append(val_macro_auc)

            # Checkpoint on best validation Macro ROC AUC
            if val_macro_auc > self.best_macro_auc:
                self.best_macro_auc = val_macro_auc
                save_path = self.checkpoint_dir / checkpoint_name
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "macro_auc": val_macro_auc,
                    },
                    save_path,
                )
                self.best_checkpoint_path = save_path

        return history
