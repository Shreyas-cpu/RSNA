"""Shared pytest fixtures and synthetic DICOM generator for unit testing."""

import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
import pytest


def create_synthetic_dicom(
    file_path: Path,
    pixel_array: np.ndarray,
    instance_number: int = 1,
    orientation_cosines: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    series_description: str = "T2 SAGITTAL",
    photometric: str = "MONOCHROME2",
    rescale_slope: float = 1.0,
    rescale_intercept: float = 0.0,
) -> Path:
    """Creates a valid synthetic DICOM file for testing."""
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.Modality = "MR"
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()

    ds.PatientName = "TEST^KNEE"
    ds.PatientID = "123456"
    ds.InstanceNumber = instance_number
    ds.SliceLocation = float(instance_number * 3.0)
    ds.SeriesDescription = series_description
    ds.PhotometricInterpretation = photometric
    ds.RescaleSlope = rescale_slope
    ds.RescaleIntercept = rescale_intercept
    ds.ImageOrientationPatient = list(orientation_cosines)

    arr = np.asarray(pixel_array, dtype=np.uint16)
    ds.Rows, ds.Columns = arr.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PixelData = arr.tobytes()

    ds.save_as(str(file_path))
    return file_path


@pytest.fixture
def synthetic_dicom_file(tmp_path: Path) -> Path:
    """Fixture providing a single synthetic DICOM file."""
    pixels = np.ones((64, 64), dtype=np.uint16) * 500
    file_path = tmp_path / "slice_01.dcm"
    return create_synthetic_dicom(file_path, pixels, instance_number=1)


@pytest.fixture
def synthetic_series_dir(tmp_path: Path) -> Path:
    """Fixture providing a series directory with 5 sorted slices."""
    series_dir = tmp_path / "series_01"
    for i in range(1, 6):
        pixels = np.full((64, 64), fill_value=i * 100, dtype=np.uint16)
        file_path = series_dir / f"slice_{i:02d}.dcm"
        create_synthetic_dicom(
            file_path=file_path,
            pixel_array=pixels,
            instance_number=i,
            orientation_cosines=(0.0, 1.0, 0.0, 0.0, 0.0, -1.0),  # Sagittal cosines
            series_description="Sagittal PD FatSat",
        )
    return series_dir


@pytest.fixture
def synthetic_dicom_series():
    """Callable fixture to create a series of synthetic DICOM files in a directory."""
    def _create(
        series_dir: Path,
        num_slices: int = 5,
        orientation_cosines: Tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        series_description: str = "Series",
    ) -> Path:
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(1, num_slices + 1):
            pixels = np.full((64, 64), fill_value=i * 100, dtype=np.uint16)
            file_path = series_dir / f"slice_{i:02d}.dcm"
            create_synthetic_dicom(
                file_path=file_path,
                pixel_array=pixels,
                instance_number=i,
                orientation_cosines=orientation_cosines,
                series_description=series_description,
            )
        return series_dir

    return _create
