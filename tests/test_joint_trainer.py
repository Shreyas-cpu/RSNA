"""Unit tests for SampleWeightedBCEWithLogitsLoss and JointTrainer."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.models.multi_plane import MultiPlaneKneeModel
from src.training.joint_trainer import (
    JointTrainer,
    SampleWeightedBCEWithLogitsLoss,
)


def test_sample_weighted_loss_computation():
    """Verify SampleWeightedBCEWithLogitsLoss weights samples and ignores zero-weight cases."""
    B, C = 4, 12
    logits = torch.randn(B, C, requires_grad=True)
    targets = torch.randint(0, 2, (B, C)).float()

    criterion = SampleWeightedBCEWithLogitsLoss()

    # 1. Unweighted should match standard BCE
    loss_unweighted = criterion(logits, targets)
    assert loss_unweighted.item() > 0

    # 2. Zero-weight sample verification
    # Give sample 0 zero weight, and samples 1-3 weight 1.0
    weights = torch.tensor([0.0, 1.0, 1.0, 1.0])
    loss_weighted = criterion(logits, targets, sample_weights=weights)

    # Compute loss on subset [1:] directly
    loss_subset = criterion(logits[1:], targets[1:])
    assert torch.isclose(loss_weighted, loss_subset, atol=1e-5)


def test_sample_weighted_loss_pos_weight():
    """Verify pos_weight scaling increases loss on positive targets."""
    logits = torch.zeros(2, 3)
    targets = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    pos_w = torch.tensor([5.0, 5.0, 5.0])
    criterion_weighted = SampleWeightedBCEWithLogitsLoss(pos_weight=pos_w)
    criterion_default = SampleWeightedBCEWithLogitsLoss()

    loss_pos_w = criterion_weighted(logits, targets)
    loss_base = criterion_default(logits, targets)

    assert loss_pos_w > loss_base


class DummyMultiPlaneDataset:
    """Mock dataset yielding multi-plane dictionaries with targets and sample weights."""

    def __init__(self, n_samples: int = 4):
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        planes = {
            "sagittal": torch.randn(2, 3, 32, 32),
            "coronal": torch.randn(2, 3, 32, 32),
            "axial": torch.randn(2, 3, 32, 32),
        }
        targets = torch.randint(0, 2, (len(DEFAULT_TARGET_COLS),)).float()
        # Half gold (1.0), half silver (0.6)
        weight = torch.tensor(1.0 if idx % 2 == 0 else 0.6)
        return {"planes": planes, "targets": targets, "sample_weight": weight}


def test_joint_trainer_fit_and_checkpoint(tmp_path):
    """Verify JointTrainer training/val epoch, AMP scaling, and checkpoint saving."""
    dataset = DummyMultiPlaneDataset(n_samples=4)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

    model = MultiPlaneKneeModel(
        backbone_name="resnet34d",
        pretrained=False,
        num_classes=len(DEFAULT_TARGET_COLS),
        fusion_heads=2,
    )

    trainer = JointTrainer(
        model=model,
        checkpoint_dir=tmp_path / "joint_ckpt",
        use_amp=True,
    )

    history = trainer.fit(train_loader, val_loader, epochs=1)

    assert "train_loss" in history
    assert "val_loss" in history
    assert "val_macro_auc" in history
    assert len(history["train_loss"]) == 1

    # Verify best checkpoint was created
    ckpt_file = tmp_path / "joint_ckpt" / "best_multiplane_model.pt"
    assert ckpt_file.exists()
