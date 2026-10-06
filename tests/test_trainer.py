import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.model import RSNAKneeModel
from src.training.trainer import Trainer


class DummyKneeDataset(Dataset):
    """Synthetic dataset generating 5D slab tensors and 12-class binary targets."""

    def __init__(self, n_samples: int = 8):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Shape: (4 slabs, 3 channels, 32 H, 32 W)
        slabs = torch.randn(4, 3, 32, 32)
        # 12 targets: binary indicator
        targets = torch.tensor([float((idx + j) % 2) for j in range(12)], dtype=torch.float32)
        return {"slabs": slabs, "targets": targets}


def test_trainer_train_and_val_epoch(tmp_path):
    """Verify Trainer executes training and validation epochs cleanly."""
    train_ds = DummyKneeDataset(n_samples=8)
    val_ds = DummyKneeDataset(n_samples=4)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False)
    trainer = Trainer(
        model=model,
        accumulation_steps=2,
        checkpoint_dir=tmp_path / "ckpt",
        use_amp=True,
    )

    train_loss = trainer.train_epoch(train_loader)
    assert train_loss > 0.0
    assert not torch.isnan(torch.tensor(train_loss))

    val_metrics = trainer.validate_epoch(val_loader)
    assert "val_loss" in val_metrics
    assert "macro_auc" in val_metrics
    assert "report" in val_metrics


def test_trainer_fit_and_checkpoint(tmp_path):
    """Verify Trainer fit loop saves best_model.pt on improvement."""
    train_ds = DummyKneeDataset(n_samples=6)
    val_ds = DummyKneeDataset(n_samples=4)
    train_loader = DataLoader(train_ds, batch_size=2)
    val_loader = DataLoader(val_ds, batch_size=2)

    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False)
    ckpt_dir = tmp_path / "checkpoints"

    trainer = Trainer(
        model=model,
        checkpoint_dir=ckpt_dir,
        use_amp=True,
    )

    history = trainer.fit(train_loader, val_loader, epochs=2)

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["val_macro_auc"]) == 2
    assert (ckpt_dir / "best_model.pt").exists()
