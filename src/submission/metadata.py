"""Kaggle Kernel Metadata Generator for CLI Deployment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def generate_kernel_metadata(
    kernel_slug: str = "rsna-knee-submission",
    title: str = "RSNA Knee Multi-Plane Attention Submission",
    code_file: str = "submission.py",
    language: str = "python",
    kernel_type: str = "script",
    is_private: bool = True,
    enable_gpu: bool = True,
    enable_internet: bool = False,
    competition_sources: Optional[List[str]] = None,
    dataset_sources: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Generate Kaggle kernel-metadata.json configuration dictionary.

    Args:
        kernel_slug: Kaggle kernel slug / ID.
        title: Display title for the kernel.
        code_file: Entrypoint filename ('submission.py' or 'rsna_knee_submission.ipynb').
        language: Programming language ('python').
        kernel_type: 'script' or 'notebook'.
        is_private: Whether the kernel is private.
        enable_gpu: Whether to attach GPU accelerator (T4/P100).
        enable_internet: Must be False for competition submission scoring.
        competition_sources: List of competition handles.
        dataset_sources: List of external dataset handles (weights, wheels).
        output_path: Optional path to write kernel-metadata.json file.

    Returns:
        Dict representing kernel-metadata.json.
    """
    comp_sources = competition_sources or ["rsna-knee-abnormality-detection"]
    ds_sources = dataset_sources or []

    metadata = {
        "id": kernel_slug,
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": "true" if is_private else "false",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": ds_sources,
        "competition_sources": comp_sources,
    }

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return metadata
