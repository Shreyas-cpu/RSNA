import json
import time
from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.dataset import RSNAKneeDataset, create_dataloader
from src.data.dicom_reader import DICOMReader
from src.data.plane_classifier import classify_plane
from src.data.slab_extractor import SlabExtractor


def test_cache_study_slabs_and_load(tmp_path):
    """Verify caching multi-plane slabs creates valid .npz file and manifest."""
    study_id = "test_study_001"
    cache_dir = tmp_path / "slab_cache"

    # Synthetic volumes for 3 planes
    series_vols = {
        "sagittal": np.random.uniform(0, 1000, size=(10, 128, 128)).astype(np.float32),
        "coronal": np.random.uniform(0, 1000, size=(10, 128, 128)).astype(np.float32),
        "axial": np.random.uniform(0, 1000, size=(10, 128, 128)).astype(np.float32),
    }

    npz_path = RSNAKneeDataset.cache_study_slabs(
        study_id=study_id,
        series_volumes=series_vols,
        output_dir=cache_dir,
        num_slabs=5,
        target_shape=(64, 64),
    )

    assert Path(npz_path).exists()
    assert (cache_dir / "manifest.json").exists()

    with open(cache_dir / "manifest.json", "r") as f:
        manifest = json.load(f)
    assert study_id in manifest
    assert manifest[study_id]["total_slabs"] == 15  # 3 planes * 5 slabs
    assert manifest[study_id]["shape"] == [15, 3, 64, 64]

    # Test loading from dataset
    records = [{"StudyInstanceUID": study_id, "label_1": 1.0, "label_2": 0.0}]
    dataset = RSNAKneeDataset(
        records=records,
        cache_dir=cache_dir,
        num_slabs_per_plane=5,
        target_shape=(64, 64),
        target_cols=["label_1", "label_2"],
    )

    assert len(dataset) == 1
    item = dataset[0]

    assert item["study_id"] == study_id
    assert isinstance(item["slabs"], torch.Tensor)
    assert item["slabs"].shape == (15, 3, 64, 64)
    assert item["slabs"].dtype == torch.float32
    assert isinstance(item["targets"], torch.Tensor)
    assert torch.equal(item["targets"], torch.tensor([1.0, 0.0]))


def test_dataloader_batch_generation(tmp_path):
    """Verify DataLoader yields batched slab tensors with shape (B, S, 3, H, W)."""
    cache_dir = tmp_path / "cache"
    num_studies = 4
    records = []

    for i in range(num_studies):
        sid = f"study_{i:03d}"
        records.append({"StudyInstanceUID": sid, "target": float(i % 2)})
        vols = {
            "sagittal": np.ones((8, 64, 64), dtype=np.float32) * i,
            "coronal": np.ones((8, 64, 64), dtype=np.float32) * (i + 1),
            "axial": np.ones((8, 64, 64), dtype=np.float32) * (i + 2),
        }
        RSNAKneeDataset.cache_study_slabs(
            study_id=sid,
            series_volumes=vols,
            output_dir=cache_dir,
            num_slabs=4,
            target_shape=(64, 64),
        )

    dataset = RSNAKneeDataset(
        records=records,
        cache_dir=cache_dir,
        num_slabs_per_plane=4,
        target_shape=(64, 64),
        target_cols=["target"],
    )

    loader = create_dataloader(dataset, batch_size=2, shuffle=False, num_workers=0)
    batches = list(loader)

    assert len(batches) == 2
    for b in batches:
        assert "study_id" in b
        assert "slabs" in b
        assert "targets" in b
        # Shape: (Batch_size, 12 slabs, 3 channels, 64 H, 64 W)
        assert b["slabs"].shape == (2, 12, 3, 64, 64)
        assert b["targets"].shape == (2, 1)


def test_end_to_end_ingestion_and_streaming(tmp_path, synthetic_dicom_series):
    """End-to-end integration: DICOM -> Reader -> Plane -> Slabs -> Cache -> Dataset -> DataLoader."""
    study_dir = tmp_path / "study_synthetic"
    study_dir.mkdir()

    # Create 3 series: Sagittal, Coronal, Axial
    series_configs = [
        ("series_sag", (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)),  # Sagittal
        ("series_cor", (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)),  # Coronal
        ("series_ax", (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),   # Axial
    ]

    for name, cosines in series_configs:
        s_dir = study_dir / name
        synthetic_dicom_series(s_dir, num_slices=6, orientation_cosines=cosines)

    # 1. Reader ingests series
    reader = DICOMReader()
    series_vols = {}
    for s_dir in study_dir.iterdir():
        vol, meta = reader.read_series(s_dir)
        plane = meta["plane"]
        series_vols[plane] = vol

    assert "sagittal" in series_vols
    assert "coronal" in series_vols
    assert "axial" in series_vols

    # 2. Cache study slabs
    cache_dir = tmp_path / "e2e_cache"
    RSNAKneeDataset.cache_study_slabs(
        study_id="study_e2e",
        series_volumes=series_vols,
        output_dir=cache_dir,
        num_slabs=3,
        target_shape=(64, 64),
    )

    # 3. Stream through Dataset and DataLoader
    records = [{"StudyInstanceUID": "study_e2e", "label": 1.0}]
    dataset = RSNAKneeDataset(
        records=records,
        cache_dir=cache_dir,
        num_slabs_per_plane=3,
        target_shape=(64, 64),
        target_cols=["label"],
    )
    loader = create_dataloader(dataset, batch_size=1, shuffle=False, num_workers=0)

    batch = next(iter(loader))
    assert batch["slabs"].shape == (1, 9, 3, 64, 64)  # 3 planes * 3 slabs = 9
    assert batch["targets"].shape == (1, 1)


def test_throughput_benchmark():
    """Benchmark slab extraction throughput ensuring < 5ms per slice equivalent."""
    extractor = SlabExtractor(target_shape=(256, 256))
    # 50-slice volume
    volume = np.random.uniform(0, 2000, size=(50, 256, 256)).astype(np.float32)

    start = time.perf_counter()
    slabs = extractor.extract_slabs(volume, num_slabs=10)
    elapsed = time.perf_counter() - start

    ms_per_slice = (elapsed / 50.0) * 1000.0
    print(f"\nThroughput benchmark: {ms_per_slice:.2f} ms/slice ({elapsed:.3f}s for 50 slices)")

    # Assert processing is comfortably fast (< 5ms per slice)
    assert ms_per_slice < 5.0, f"Extraction too slow: {ms_per_slice:.2f} ms/slice"
    assert slabs.shape == (10, 3, 256, 256)
