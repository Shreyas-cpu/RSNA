"""Unit tests for SubmissionExporter and kernel metadata generator."""

import json
from pathlib import Path
import pytest

from src.evaluation.split import DEFAULT_TARGET_COLS
from src.submission.exporter import SubmissionExporter
from src.submission.metadata import generate_kernel_metadata


def test_generate_kernel_metadata(tmp_path):
    """Verify kernel-metadata.json is generated with required offline and GPU settings."""
    out_file = tmp_path / "kernel-metadata.json"
    meta = generate_kernel_metadata(
        kernel_slug="promethious/rsna-knee-multiplane",
        title="RSNA Knee Multi-Plane Attention",
        output_path=out_file,
    )

    assert out_file.exists()
    assert meta["id"] == "promethious/rsna-knee-multiplane"
    assert meta["enable_internet"] == "false"
    assert meta["enable_gpu"] == "true"
    assert "rsna-knee-abnormality-detection" in meta["competition_sources"]

    # Verify JSON structure
    with open(out_file, "r") as f:
        data = json.load(f)
        assert data["id"] == "promethious/rsna-knee-multiplane"


def test_export_standalone_script(tmp_path):
    """Verify exported submission script is valid, syntactically compilable Python."""
    exporter = SubmissionExporter(target_cols=DEFAULT_TARGET_COLS)
    out_script = tmp_path / "submission.py"
    res_path = exporter.export_standalone_script(out_script)

    assert res_path.exists()
    code = res_path.read_text(encoding="utf-8")

    # Assert key components exist
    assert "classify_plane" in code
    assert "extract_25d_slabs" in code
    assert "MultiPlaneKneeModel" in code
    assert "run_inference" in code
    assert "TARGET_COLS" in code

    # Verify syntax compilation
    compiled = compile(code, str(out_script), "exec")
    assert compiled is not None


def test_export_jupyter_notebook(tmp_path):
    """Verify exported Jupyter Notebook complies with nbformat v4 specification."""
    exporter = SubmissionExporter()
    out_nb = tmp_path / "rsna_knee_submission.ipynb"
    res_path = exporter.export_jupyter_notebook(out_nb)

    assert res_path.exists()
    with open(res_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 2
    assert nb["cells"][0]["cell_type"] == "markdown"
    assert nb["cells"][1]["cell_type"] == "code"
