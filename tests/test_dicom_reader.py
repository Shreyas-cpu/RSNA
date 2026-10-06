"""Unit tests for high-throughput DICOM reader."""

from pathlib import Path
import numpy as np
import pytest
from src.data.dicom_reader import DICOMReader
from tests.conftest import create_synthetic_dicom


def test_dicom_reader_read_single_slice(synthetic_dicom_file: Path):
    reader = DICOMReader()
    slice_data = reader.read_slice(synthetic_dicom_file)

    assert slice_data.pixel_array.shape == (64, 64)
    assert slice_data.instance_number == 1
    assert slice_data.photometric_interpretation == "MONOCHROME2"
    assert np.allclose(slice_data.pixel_array, 500.0)


def test_dicom_reader_monochrome1_inversion(tmp_path: Path):
    # For MONOCHROME1, white is low and black is high; reader must invert
    pixels = np.array([[100, 200], [300, 400]], dtype=np.uint16)
    file_path = tmp_path / "mono1.dcm"
    create_synthetic_dicom(file_path, pixels, photometric="MONOCHROME1")

    reader = DICOMReader()
    slice_data = reader.read_slice(file_path)

    # Inversion: 400 - val -> [[300, 200], [100, 0]]
    expected = np.array([[300, 200], [100, 0]], dtype=np.float32)
    assert np.array_equal(slice_data.pixel_array, expected)


def test_dicom_reader_rescale_slope_intercept(tmp_path: Path):
    pixels = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    file_path = tmp_path / "rescaled.dcm"
    create_synthetic_dicom(
        file_path,
        pixels,
        rescale_slope=2.0,
        rescale_intercept=-10.0,
    )

    reader = DICOMReader()
    slice_data = reader.read_slice(file_path)

    # val * 2.0 - 10.0 -> [[10, 30], [50, 70]]
    expected = np.array([[10, 30], [50, 70]], dtype=np.float32)
    assert np.array_equal(slice_data.pixel_array, expected)


def test_dicom_reader_read_series_sorting_and_plane(synthetic_series_dir: Path):
    reader = DICOMReader()
    volume, metadata = reader.read_series(synthetic_series_dir, max_workers=4)

    # 5 slices of 64x64
    assert volume.shape == (5, 64, 64)
    assert metadata["num_slices"] == 5
    assert metadata["plane"] == "sagittal"

    # Verify slices are sorted by InstanceNumber (100, 200, 300, 400, 500)
    for i in range(5):
        assert np.allclose(volume[i], (i + 1) * 100)
