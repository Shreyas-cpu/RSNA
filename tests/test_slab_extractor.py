import numpy as np
import pytest
from src.data.slab_extractor import SlabExtractor


def test_apply_windowing_linear():
    """Verify linear windowing correctly maps window range to [0.0, 1.0]."""
    # Center 500, width 400 -> range [300, 700]
    data = np.array([100.0, 300.0, 500.0, 700.0, 900.0], dtype=np.float32)
    norm = SlabExtractor.apply_windowing(data, window_center=500.0, window_width=400.0)

    assert norm.dtype == np.float32
    assert norm[0] == 0.0  # < 300 clamped to 0
    assert norm[1] == 0.0  # 300 -> 0.0
    assert pytest.approx(norm[2], rel=1e-5) == 0.5  # 500 -> 0.5
    assert norm[3] == 1.0  # 700 -> 1.0
    assert norm[4] == 1.0  # > 700 clamped to 1


def test_apply_windowing_percentile_fallback():
    """Verify dynamic percentile clipping scales intensities cleanly without tags."""
    np.random.seed(42)
    data = np.random.normal(loc=1000.0, scale=200.0, size=(10, 50, 50)).astype(np.float32)
    norm = SlabExtractor.apply_windowing(data, window_center=None, window_width=None)

    assert norm.shape == data.shape
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert not np.isnan(norm).any()
    assert not np.isinf(norm).any()


def test_extract_slabs_shape_and_bounds():
    """Verify extract_slabs extracts requested slab count and 3 channels."""
    extractor = SlabExtractor(target_shape=(128, 128))
    # Synthetic volume: 20 slices of 256x256
    volume = np.random.uniform(0, 4000, size=(20, 256, 256)).astype(np.float32)

    slabs = extractor.extract_slabs(volume, num_slabs=5)

    assert slabs.shape == (5, 3, 128, 128)
    assert slabs.dtype == np.float32
    assert slabs.min() >= 0.0
    assert slabs.max() <= 1.0
    assert not np.isnan(slabs).any()


def test_extract_slabs_single_slice_volume():
    """Verify extract_slabs handles edge case of single-slice MRI sequence."""
    extractor = SlabExtractor(target_shape=(64, 64))
    single_slice = np.random.uniform(100, 500, size=(1, 100, 100)).astype(np.float32)

    slabs = extractor.extract_slabs(single_slice, num_slabs=3)

    assert slabs.shape == (3, 3, 64, 64)
    # In single slice, all 3 channels should be identical duplicate of the slice
    assert np.allclose(slabs[0, 0], slabs[0, 1])
    assert np.allclose(slabs[0, 1], slabs[0, 2])


def test_resize_slab():
    """Verify resize_slab dimensions and channel consistency."""
    slab = np.ones((3, 200, 300), dtype=np.float32) * 0.75
    resized = SlabExtractor.resize_slab(slab, target_shape=(128, 128))

    assert resized.shape == (3, 128, 128)
    assert pytest.approx(resized.mean(), rel=1e-3) == 0.75
