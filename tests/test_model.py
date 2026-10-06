import pytest
import torch
import torch.nn as nn

from src.data.dataset import RSNAKneeDataset, create_dataloader
from src.evaluation.metrics import MetricTracker
from src.evaluation.split import DEFAULT_TARGET_COLS
from src.models.model import RSNAKneeModel


def test_model_forward_and_predict_proba():
    """Verify RSNAKneeModel forward pass returns (B, 12) and probabilities in [0.0, 1.0]."""
    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False)
    x = torch.randn(2, 6, 3, 64, 64)

    logits = model(x)
    assert logits.shape == (2, 12)

    probs = model.predict_proba(x)
    assert probs.shape == (2, 12)
    assert probs.min().item() >= 0.0
    assert probs.max().item() <= 1.0
    assert not torch.isnan(probs).any()


def test_model_return_attention():
    """Verify return_attention=True yields both logits and attention weights."""
    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False)
    x = torch.randn(3, 5, 3, 64, 64)

    logits, weights = model(x, return_attention=True)
    assert logits.shape == (3, 12)
    assert weights.shape == (3, 5, 1)
    assert torch.allclose(weights.sum(dim=1).squeeze(-1), torch.ones(3), atol=1e-5)


def test_model_amp_and_backward():
    """Verify model forward and backward pass run under PyTorch AMP without error."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"

    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False).to(device)
    x = torch.randn(2, 4, 3, 64, 64, device=device)
    targets = torch.randint(0, 2, (2, 12), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    # Test autocast
    with torch.autocast(device_type=device_type, dtype=torch.float16 if device == "cuda" else torch.bfloat16):
        logits = model(x)
        loss = criterion(logits, targets)

    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
    assert loss.item() > 0.0


def test_end_to_end_dataloader_to_metric_pipeline(tmp_path):
    """End-to-end integration: Caching -> Dataset -> DataLoader -> Model -> MetricTracker."""
    import numpy as np

    cache_dir = tmp_path / "model_pipeline_cache"
    n_studies = 4
    records = []

    for i in range(n_studies):
        sid = f"study_{i:03d}"
        targets = [float((i + j) % 2) for j in range(12)]
        target_dict = {DEFAULT_TARGET_COLS[j]: targets[j] for j in range(12)}
        records.append({"StudyInstanceUID": sid, **target_dict})

        vols = {
            "sagittal": np.random.uniform(0, 1000, size=(6, 64, 64)).astype(np.float32),
            "coronal": np.random.uniform(0, 1000, size=(6, 64, 64)).astype(np.float32),
            "axial": np.random.uniform(0, 1000, size=(6, 64, 64)).astype(np.float32),
        }
        RSNAKneeDataset.cache_study_slabs(
            study_id=sid,
            series_volumes=vols,
            output_dir=cache_dir,
            num_slabs=3,  # 3 slabs * 3 planes = 9 slabs total
            target_shape=(64, 64),
        )

    dataset = RSNAKneeDataset(
        records=records,
        cache_dir=cache_dir,
        num_slabs_per_plane=3,
        target_shape=(64, 64),
        target_cols=DEFAULT_TARGET_COLS,
    )
    loader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)

    # Instantiate model
    model = RSNAKneeModel(backbone_name="resnet34d", num_classes=12, pretrained=False)
    tracker = MetricTracker(class_names=DEFAULT_TARGET_COLS)

    for batch in loader:
        slabs = batch["slabs"]  # (B, 9, 3, 64, 64)
        batch_targets = batch["targets"]  # (B, 12)
        probs = model.predict_proba(slabs)
        tracker.update(batch_targets, probs)

    results = tracker.compute()
    assert results["num_samples"] == n_studies
    assert "macro_auc" in results
    report = tracker.generate_summary_report()
    assert "VALIDATION EVALUATION REPORT" in report
    print("\n" + report)
