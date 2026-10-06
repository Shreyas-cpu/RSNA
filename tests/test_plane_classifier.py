"""Unit tests for anatomical plane classification."""

import pytest
from src.data.plane_classifier import classify_plane


def test_classify_sagittal_from_cosines():
    # Sagittal plane: normal vector along X axis (e.g. [1, 0, 0])
    # r = [0, 1, 0], c = [0, 0, -1] -> r x c = [-1, 0, 0] (X dominant)
    cosines = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    assert classify_plane(cosines) == "sagittal"


def test_classify_coronal_from_cosines():
    # Coronal plane: normal vector along Y axis (e.g. [0, 1, 0])
    # r = [1, 0, 0], c = [0, 0, -1] -> r x c = [0, 1, 0] (Y dominant)
    cosines = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0]
    assert classify_plane(cosines) == "coronal"


def test_classify_axial_from_cosines():
    # Axial plane: normal vector along Z axis (e.g. [0, 0, 1])
    # r = [1, 0, 0], c = [0, 1, 0] -> r x c = [0, 0, 1] (Z dominant)
    cosines = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert classify_plane(cosines) == "axial"


def test_classify_plane_keyword_fallback():
    assert classify_plane(None, series_description="T2_SAG_FSE") == "sagittal"
    assert classify_plane(None, series_description="COR_PD_SPAIR") == "coronal"
    assert classify_plane(None, series_description="AXIAL_T1_KNEE") == "axial"
    assert classify_plane(None, series_description="LOCALIZER_SCOUT") == "unknown"


def test_classify_plane_empty_or_invalid():
    assert classify_plane([]) == "unknown"
    assert classify_plane([0.0, 0.0]) == "unknown"
    assert classify_plane(None, None) == "unknown"
