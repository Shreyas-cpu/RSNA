"""Anatomical plane classification for knee MRI series.

Calculates the normal vector from ImageOrientationPatient direction cosines
to classify Sagittal, Coronal, or Axial planes with geometric certainty,
falling back to SeriesDescription heuristics if cosines are missing.
"""

from typing import List, Optional, Sequence, Union
import numpy as np


def classify_plane(
    orientation_cosines: Optional[Union[Sequence[float], np.ndarray]] = None,
    series_description: Optional[str] = None,
) -> str:
    """Classifies the anatomical plane of an MRI sequence.

    Args:
        orientation_cosines: 6-element sequence [rx, ry, rz, cx, cy, cz]
            representing row and column direction cosines (ImageOrientationPatient).
        series_description: Optional DICOM SeriesDescription string.

    Returns:
        One of 'sagittal', 'coronal', 'axial', or 'unknown'.
    """
    if orientation_cosines is not None and len(orientation_cosines) == 6:
        try:
            cosines = np.asarray(orientation_cosines, dtype=np.float64)
            r = cosines[:3]
            c = cosines[3:]

            # Normal vector n = r x c
            normal = np.cross(r, c)
            abs_norm = np.abs(normal)

            # Dominant normal axis dictates orientation
            # Axis 0 (X normal -> Left/Right) = Sagittal
            # Axis 1 (Y normal -> Anterior/Posterior) = Coronal
            # Axis 2 (Z normal -> Head/Foot) = Axial
            max_axis = int(np.argmax(abs_norm))
            if np.max(abs_norm) > 1e-4:
                if max_axis == 0:
                    return "sagittal"
                elif max_axis == 1:
                    return "coronal"
                elif max_axis == 2:
                    return "axial"
        except Exception:
            pass

    # Heuristic fallback based on series description keywords
    if series_description:
        desc = series_description.lower()
        if any(term in desc for term in ["sag", "sg", "sagittal"]):
            return "sagittal"
        if any(term in desc for term in ["cor", "cr", "coronal"]):
            return "coronal"
        if any(term in desc for term in ["ax", "tra", "trans", "axial"]):
            return "axial"

    return "unknown"
