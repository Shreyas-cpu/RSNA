"""Mixed-precision AMP training and evaluation engine.

Implements Trainer supporting PyTorch AMP (GradScaler), gradient accumulation,
AdamW optimization, learning rate scheduling, and validation Macro AUC checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import MetricTracker
from src.evaluation.split import DEFAULT_TARGET_COLS


class Trainer:
    """Trains and evaluates 2.5D knee models with AMP, accumulation, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        criterion: Optional[nn.Module] = None,
        scheduler: Optional[Any] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: Union[str, Path] = "checkpoints",
        class_names: Optional[List[str]] = None,
    ):
        """Initialize Trainer.

        Args:
            model: PyTorch model instance (e.g., RSNAKneeModel).
            optimizer: Optimizer instance. Defaults to AdamW(lr=1e-4, weight_decay=1e-2).
            criterion: Loss function. Defaults to BCEWithLogitsLoss().
            scheduler: Optional learning rate scheduler.
            device: Computing device ('cuda' or 'cpu'). Auto-detected if None.
            use_amp: Whether to enable Automatic Mixed Precision (AMP).
            accumulation_steps: Steps to accumulate gradients before updating optimizer.
            max_grad_norm: Gradient norm clipping threshold.
            checkpoint_dir: Directory to store model checkpoints.
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
        self.criterion = (criterion or nn.BCEWithLogitsLoss()).to(self.device)
        self.scheduler = scheduler
        self.use_amp = use_amp and (self.device.type in ("cuda", "cpu"))
        self.accumulation_steps = max(1, accumulation_steps)
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = class_names or DEFAULT_TARGET_COLS

        # GradScaler is primarily active on CUDA; on CPU autocast runs unscaled
        if hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.use_amp and self.device.type == "cuda")
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=(self.use_amp and self.device.type == "cuda")
            )
        self.best_macro_auc = -1.0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run a single training epoch across batches.

        Args:
            dataloader: PyTorch DataLoader yielding batches with 'slabs' and 'targets'.

        Returns:
            Average training loss float across the epoch.
        """
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        device_type = self.device.type
        amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        for step, batch in enumerate(dataloader):
            slabs = batch["slabs"].to(self.device)
            targets = batch["targets"].to(self.device)

            with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.use_amp):
                logits = self.model(slabs)
                loss = self.criterion(logits, targets)
                scaled_loss = loss / self.accumulation_steps

            self.scaler.scale(scaled_loss).backward()
            total_loss += loss.item()

            if (step + 1) % self.accumulation_steps == 0 or (step + 1) == len(dataloader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

        return float(total_loss / max(len(dataloader), 1))

    def validate_epoch(self, dataloader: DataLoader) -> Dict[str, Any]:
        """Run a validation epoch and compute competition Macro ROC AUC.

        Args:
            dataloader: DataLoader yielding validation batches.

        Returns:
            Dictionary with 'val_loss', 'macro_auc', 'per_class', 'support', and 'report'.
        """
        self.model.eval()
        total_loss = 0.0
        tracker = MetricTracker(class_names=self.class_names)

        device_type = self.device.type
        amp_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with torch.no_grad():
            for batch in dataloader:
                slabs = batch["slabs"].to(self.device)
                targets = batch["targets"].to(self.device)

                with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=self.use_amp):
                    logits = self.model(slabs)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                tracker.update(targets, probs)

        metrics = tracker.compute()
        metrics["val_loss"] = float(total_loss / max(len(dataloader), 1))
        metrics["report"] = tracker.generate_summary_report()
        return metrics

    def save_checkpoint(self, path: Union[str, Path], meta: Optional[Dict[str, Any]] = None) -> Path:
        """Save model checkpoint to disk.

        Args:
            path: Destination file path.
            meta: Optional metadata dict.

        Returns:
            Path of saved checkpoint.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_macro_auc": self.best_macro_auc,
            "meta": meta or {},
        }
        torch.save(checkpoint, p)
        return p

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 3,
    ) -> Dict[str, Any]:
        """Run full training and validation lifecycle.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Number of epochs to train.

        Returns:
            Dictionary containing epoch training logs and best_macro_auc.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auc": [],
        }

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate_epoch(val_loader)
            val_loss = val_metrics["val_loss"]
            val_auc = val_metrics["macro_auc"]

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auc"].append(val_auc)

            # Checkpoint on improvement
            if val_auc > self.best_macro_auc:
                self.best_macro_auc = val_auc
                best_path = self.checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path, meta={"epoch": epoch, "val_macro_auc": val_auc})

        history["best_macro_auc"] = self.best_macro_auc
        return history
