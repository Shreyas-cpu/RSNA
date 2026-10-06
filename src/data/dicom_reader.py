"""High-throughput DICOM reading and decompression engine.

Provides C++ accelerated decompression via dicomsdl when available,
with automatic fallback to pydicom and pylibjpeg, thread-pool concurrency,
MONOCHROME1 pixel inversion, and physical slice ordering.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pydicom

from src.data.plane_classifier import classify_plane

# Optional dicomsdl import for high-speed C++ decoding
try:
    import dicomsdl
    HAS_DICOMSDL = True
except ImportError:
    dicomsdl = None
    HAS_DICOMSDL = False


@dataclass
class SliceData:
    """Metadata and pixel array for an individual DICOM slice."""
    pixel_array: np.ndarray
    instance_number: int
    slice_location: float
    orientation_cosines: Optional[List[float]]
    series_description: str
    photometric_interpretation: str
    rescale_slope: float
    rescale_intercept: float


class DICOMReader:
    """Fast DICOM reader supporting single slice and concurrent series decompression."""

    def __init__(self, prefer_dicomsdl: bool = True):
        self.prefer_dicomsdl = prefer_dicomsdl and HAS_DICOMSDL

    def read_slice(self, path: Union[str, Path]) -> SliceData:
        """Reads and decompresses an individual DICOM slice.

        Args:
            path: Path to the .dcm file.

        Returns:
            SliceData instance containing normalized pixel array and metadata.
        """
        path = Path(path)
        pixel_array: Optional[np.ndarray] = None
        dcm_header = None

        # 1. Try C++ decompression via dicomsdl if available
        if self.prefer_dicomsdl:
            try:
                ds = dicomsdl.open(str(path))
                pixel_array = ds.pixelData()
                # Extract basic tags from dicomsdl
                info = ds.dumpDict()
                instance_number = int(info.get("InstanceNumber", 0))
                slice_location = float(info.get("SliceLocation", float(instance_number)))
                cosines = info.get("ImageOrientationPatient", None)
                if cosines and isinstance(cosines, (list, tuple)) and len(cosines) == 6:
                    cosines = [float(x) for x in cosines]
                else:
                    cosines = None
                series_desc = str(info.get("SeriesDescription", ""))
                photometric = str(info.get("PhotometricInterpretation", "MONOCHROME2")).strip()
                slope = float(info.get("RescaleSlope", 1.0))
                intercept = float(info.get("RescaleIntercept", 0.0))
            except Exception:
                pixel_array = None

        # 2. Fallback to pydicom
        if pixel_array is None:
            dcm = pydicom.dcmread(str(path))
            pixel_array = dcm.pixel_array
            instance_number = int(getattr(dcm, "InstanceNumber", 0))
            slice_location = float(getattr(dcm, "SliceLocation", float(instance_number)))
            cosines = getattr(dcm, "ImageOrientationPatient", None)
            if cosines is not None and len(cosines) == 6:
                cosines = [float(x) for x in cosines]
            else:
                cosines = None
            series_desc = str(getattr(dcm, "SeriesDescription", ""))
            photometric = str(getattr(dcm, "PhotometricInterpretation", "MONOCHROME2")).strip()
            slope = float(getattr(dcm, "RescaleSlope", 1.0))
            intercept = float(getattr(dcm, "RescaleIntercept", 0.0))

        pixel_array = np.asarray(pixel_array, dtype=np.float32)

        # 3. Apply Rescale Slope / Intercept
        if slope != 1.0 or intercept != 0.0:
            pixel_array = pixel_array * slope + intercept

        # 4. Handle Photometric Interpretation inversion
        if photometric == "MONOCHROME1":
            max_val = np.amax(pixel_array) if pixel_array.size > 0 else 0
            pixel_array = max_val - pixel_array

        return SliceData(
            pixel_array=pixel_array,
            instance_number=instance_number,
            slice_location=slice_location,
            orientation_cosines=cosines,
            series_description=series_desc,
            photometric_interpretation=photometric,
            rescale_slope=slope,
            rescale_intercept=intercept,
        )

    def read_series(
        self,
        series_dir: Union[str, Path],
        max_workers: int = 8,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Concurrently reads all slices in a series directory and sorts them spatially.

        Args:
            series_dir: Directory containing DICOM files for a single series.
            max_workers: Number of threads for concurrent decompression.

        Returns:
            Tuple of (volume_array of shape [D, H, W], series_metadata dict).
        """
        series_dir = Path(series_dir)
        dcm_files = sorted(list(series_dir.glob("*.dcm")))
        if not dcm_files:
            # Also support bare filenames without .dcm extension
            dcm_files = sorted([f for f in series_dir.iterdir() if f.is_file()])

        if not dcm_files:
            raise FileNotFoundError(f"No DICOM slices found in {series_dir}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            slices: List[SliceData] = list(executor.map(self.read_slice, dcm_files))

        # Sort slices by InstanceNumber, falling back to SliceLocation
        slices.sort(key=lambda s: (s.instance_number, s.slice_location))

        # Stack into 3D volume array (Depth, Height, Width)
        volume = np.stack([s.pixel_array for s in slices], axis=0)

        # Derive series plane from the middle slice
        mid_idx = len(slices) // 2
        ref_slice = slices[mid_idx]
        plane = classify_plane(ref_slice.orientation_cosines, ref_slice.series_description)

        metadata = {
            "num_slices": len(slices),
            "plane": plane,
            "series_description": ref_slice.series_description,
            "orientation_cosines": ref_slice.orientation_cosines,
            "shape": volume.shape,
        }

        return volume, metadata
