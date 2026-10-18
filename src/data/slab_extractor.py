"""2.5D slab extraction and tissue contrast windowing engine.

Normalizes DICOM pixel arrays and extracts contiguous 3-slice triplets (2.5D slabs)
to provide out-of-plane anatomical context without the computational cost of full 3D CNNs.
"""

from typing import Optional, Tuple
import cv2
import numpy as np


class SlabExtractor:
    """Extracts normalized 2.5D multi-slice slabs from 3D MRI volume arrays."""

    def __init__(self, target_shape: Tuple[int, int] = (256, 256)):
        """Initialize slab extractor with target spatial resolution (H, W)."""
        self.target_shape = target_shape

    @staticmethod
    def apply_windowing(
        image: np.ndarray,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
    ) -> np.ndarray:
        """Apply linear DICOM window mapping or dynamic percentile clipping.

        Args:
            image: 2D or 3D numpy array of raw intensities.
            window_center: DICOM WindowCenter attribute, if present.
            window_width: DICOM WindowWidth attribute, if present.

        Returns:
            Normalized float32 array with values strictly bounded in [0.0, 1.0].
        """
        arr = image.astype(np.float32)

        if window_center is not None and window_width is not None and window_width > 0:
            lower = float(window_center) - float(window_width) / 2.0
            upper = float(window_center) + float(window_width) / 2.0
            if upper > lower:
                norm = (arr - lower) / (upper - lower)
            else:
                norm = np.zeros_like(arr)
        else:
            # Dynamic percentile clipping (0.5th to 99.5th percentile)
            p_low = float(np.percentile(arr, 0.5))
            p_high = float(np.percentile(arr, 99.5))
            if p_high > p_low:
                norm = (arr - p_low) / (p_high - p_low)
            else:
                max_val = float(np.max(arr))
                min_val = float(np.min(arr))
                if max_val > min_val:
                    norm = (arr - min_val) / (max_val - min_val)
                else:
                    norm = np.zeros_like(arr)

        norm = np.clip(norm, 0.0, 1.0)
        return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def resize_slab(
        slab: np.ndarray,
        target_shape: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """Resize a 3-channel slab of shape (3, H, W) to (3, target_H, target_W).

        Args:
            slab: Shape (3, H, W) float32 numpy array.
            target_shape: Tuple (target_H, target_W).

        Returns:
            Resized shape (3, target_H, target_W) float32 numpy array.
        """
        target_H, target_W = target_shape
        C, H, W = slab.shape
        if H == target_H and W == target_W:
            return slab

        # Transpose to (H, W, C) for OpenCV resize
        hwc = np.transpose(slab, (1, 2, 0))
        # OpenCV resize takes (dsize_width, dsize_height)
        resized_hwc = cv2.resize(hwc, (target_W, target_H), interpolation=cv2.INTER_LINEAR)
        # Ensure 3D shape if cv2 squeezed channels
        if resized_hwc.ndim == 2:
            resized_hwc = resized_hwc[:, :, np.newaxis]
        return np.transpose(resized_hwc, (2, 0, 1)).astype(np.float32)

    def extract_slabs(
        self,
        volume_array: np.ndarray,
        num_slabs: int = 5,
        target_shape: Optional[Tuple[int, int]] = None,
        window_center: Optional[float] = None,
        window_width: Optional[float] = None,
        as_uint8: bool = False,
    ) -> np.ndarray:
        """Extract num_slabs 3-channel contiguous slabs from volume array.

        Args:
            volume_array: 3D volume array of shape (D, H, W), or 2D (H, W).
            num_slabs: Number of equidistant slabs to sample.
            target_shape: Optional override for spatial (H, W) resolution.
            window_center: Optional window center.
            window_width: Optional window width.
            as_uint8: If True, quantizes output to uint8 in [0, 255] for 4x disk compression.

        Returns:
            Array of shape (num_slabs, 3, target_H, target_W) with dtype float32 or uint8.
        """
        if volume_array.ndim == 2:
            volume_array = volume_array[np.newaxis, ...]

        D, H, W = volume_array.shape
        out_shape = target_shape or self.target_shape

        # Normalize volume tissue windowing
        norm_volume = self.apply_windowing(
            volume_array, window_center=window_center, window_width=window_width
        )

        if D == 1:
            indices = [0] * num_slabs
        else:
            indices = np.linspace(0, D - 1, num_slabs, dtype=int)

        slabs = []
        for idx in indices:
            i0 = max(0, idx - 1)
            i1 = idx
            i2 = min(D - 1, idx + 1)

            # Stack contiguous 3-slice slab: (3, H, W)
            slab = np.stack([norm_volume[i0], norm_volume[i1], norm_volume[i2]], axis=0)
            resized_slab = self.resize_slab(slab, out_shape)
            slabs.append(resized_slab)

        stacked = np.stack(slabs, axis=0)
        if as_uint8:
            return np.clip(stacked * 255.0, 0, 255).astype(np.uint8)
        return stacked.astype(np.float32)
