"""PyTorch dataset and high-throughput slab cache streaming engine.

Provides zero-bottleneck data loading by reading pre-extracted 2.5D multi-plane slabs
from compressed .npz archives, with automatic fallback to on-the-fly extraction.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.dicom_reader import DICOMReader
from src.data.plane_classifier import classify_plane
from src.data.slab_extractor import SlabExtractor

PLANES = ("sagittal", "coronal", "axial")


class RSNAKneeDataset(Dataset):
    """PyTorch Dataset streaming multi-plane 2.5D MRI slabs with disk cache support."""

    def __init__(
        self,
        records: Union[List[Dict[str, Any]], Any],
        cache_dir: Optional[Union[str, Path]] = None,
        dicom_dir: Optional[Union[str, Path]] = None,
        plane_volumes: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
        slab_extractor: Optional[SlabExtractor] = None,
        num_slabs_per_plane: int = 5,
        target_shape: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        target_cols: Optional[List[str]] = None,
    ):
        """Initialize the dataset.

        Args:
            records: List of dictionaries or pandas DataFrame containing study metadata.
            cache_dir: Path to directory containing pre-extracted .npz slab files.
            dicom_dir: Path to raw DICOM studies for on-the-fly fallback ingestion.
            plane_volumes: Optional in-memory mapping of study_id -> {plane: volume_array}.
            slab_extractor: SlabExtractor instance.
            num_slabs_per_plane: Number of 2.5D slabs per anatomical plane.
            target_shape: Spatial resolution (H, W).
            transform: Optional transform applied to each 3-channel slab.
            target_cols: List of column names corresponding to target labels.
        """
        # Convert DataFrame to records if pandas DataFrame
        if hasattr(records, "to_dict"):
            self.records = records.to_dict(orient="records")
        else:
            self.records = list(records)

        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dicom_dir = Path(dicom_dir) if dicom_dir else None
        self.plane_volumes = plane_volumes or {}
        self.slab_extractor = slab_extractor or SlabExtractor(target_shape=target_shape)
        self.num_slabs_per_plane = num_slabs_per_plane
        self.target_shape = target_shape
        self.transform = transform
        self.target_cols = target_cols

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def cache_study_slabs(
        cls,
        study_id: str,
        series_volumes: Dict[str, np.ndarray],
        output_dir: Union[str, Path],
        slab_extractor: Optional[SlabExtractor] = None,
        num_slabs: int = 5,
        target_shape: Tuple[int, int] = (256, 256),
    ) -> Path:
        """Extract multi-plane slabs and cache as compressed .npz archive with manifest.

        Args:
            study_id: Unique StudyInstanceUID.
            series_volumes: Dict mapping plane ('sagittal', 'coronal', 'axial') to 3D arrays.
            output_dir: Destination directory.
            slab_extractor: SlabExtractor instance.
            num_slabs: Number of slabs per plane.
            target_shape: Spatial resolution (H, W).

        Returns:
            Path to the saved .npz archive.
        """
        extractor = slab_extractor or SlabExtractor(target_shape=target_shape)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        plane_slabs = {}
        combined_list = []

        for plane in PLANES:
            if plane in series_volumes and series_volumes[plane] is not None:
                slabs = extractor.extract_slabs(
                    series_volumes[plane],
                    num_slabs=num_slabs,
                    target_shape=target_shape,
                )
            else:
                # Fallback to zeros if plane sequence is absent
                slabs = np.zeros(
                    (num_slabs, 3, target_shape[0], target_shape[1]),
                    dtype=np.float32,
                )
            plane_slabs[plane] = slabs
            combined_list.append(slabs)

        # Shape: (num_planes * num_slabs, 3, H, W)
        combined_slabs = np.concatenate(combined_list, axis=0).astype(np.float32)

        npz_path = out_dir / f"{study_id}.npz"
        np.savez_compressed(
            npz_path,
            slabs=combined_slabs,
            sagittal=plane_slabs["sagittal"],
            coronal=plane_slabs["coronal"],
            axial=plane_slabs["axial"],
        )

        # Update manifest.json
        manifest_path = out_dir / "manifest.json"
        manifest_data = {}
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = {}

        manifest_data[study_id] = {
            "file": f"{study_id}.npz",
            "total_slabs": combined_slabs.shape[0],
            "channels": 3,
            "shape": list(combined_slabs.shape),
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=2)

        return npz_path

    def _load_slabs_from_cache(self, study_id: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        cache_path = self.cache_dir / f"{study_id}.npz"
        if not cache_path.exists():
            return None

        with np.load(cache_path) as npz:
            if "slabs" in npz:
                arr = npz["slabs"]
                if arr.dtype == np.uint8:
                    return (arr.astype(np.float32) / 255.0)
                return arr.astype(np.float32)
            # Reconstruct from planes
            combined = []
            for plane in PLANES:
                if plane in npz:
                    p_arr = npz[plane]
                    if p_arr.dtype == np.uint8:
                        p_arr = p_arr.astype(np.float32) / 255.0
                    else:
                        p_arr = p_arr.astype(np.float32)
                    combined.append(p_arr)
                else:
                    combined.append(
                        np.zeros(
                            (
                                self.num_slabs_per_plane,
                                3,
                                self.target_shape[0],
                                self.target_shape[1],
                            ),
                            dtype=np.float32,
                        )
                    )
            return np.concatenate(combined, axis=0).astype(np.float32)

    def _extract_slabs_on_the_fly(self, study_id: str) -> np.ndarray:
        # Check in-memory plane volumes
        if study_id in self.plane_volumes:
            vols = self.plane_volumes[study_id]
            combined = []
            for plane in PLANES:
                if plane in vols and vols[plane] is not None:
                    s = self.slab_extractor.extract_slabs(
                        vols[plane],
                        num_slabs=self.num_slabs_per_plane,
                        target_shape=self.target_shape,
                    )
                else:
                    s = np.zeros(
                        (
                            self.num_slabs_per_plane,
                            3,
                            self.target_shape[0],
                            self.target_shape[1],
                        ),
                        dtype=np.float32,
                    )
                combined.append(s)
            return np.concatenate(combined, axis=0).astype(np.float32)

        # Fallback to reading raw DICOMs from disk if dicom_dir provided
        if self.dicom_dir:
            study_path = self.dicom_dir / study_id
            if study_path.exists():
                reader = DICOMReader()
                vols = {}
                for series_p in study_path.iterdir():
                    if series_p.is_dir():
                        series_vol, meta = reader.read_series(series_p)
                        plane = meta.get("plane", "unknown")
                        if plane in PLANES and plane not in vols:
                            vols[plane] = series_vol

                combined = []
                for plane in PLANES:
                    if plane in vols:
                        s = self.slab_extractor.extract_slabs(
                            vols[plane],
                            num_slabs=self.num_slabs_per_plane,
                            target_shape=self.target_shape,
                        )
                    else:
                        s = np.zeros(
                            (
                                self.num_slabs_per_plane,
                                3,
                                self.target_shape[0],
                                self.target_shape[1],
                            ),
                            dtype=np.float32,
                        )
                    combined.append(s)
                return np.concatenate(combined, axis=0).astype(np.float32)

        # Fallback default zeros
        total_slabs = len(PLANES) * self.num_slabs_per_plane
        return np.zeros(
            (total_slabs, 3, self.target_shape[0], self.target_shape[1]),
            dtype=np.float32,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        study_id = str(record.get("StudyInstanceUID") or record.get("study_id", f"study_{idx}"))

        # Retrieve slabs from cache or on-the-fly
        slabs = self._load_slabs_from_cache(study_id)
        if slabs is None:
            slabs = self._extract_slabs_on_the_fly(study_id)

        # Apply transforms if provided
        if self.transform is not None:
            transformed = []
            for s in slabs:
                # s is (3, H, W)
                s_hwc = np.transpose(s, (1, 2, 0))
                res = self.transform(image=s_hwc)
                trans_img = res["image"] if isinstance(res, dict) else res
                if trans_img.ndim == 2:
                    trans_img = trans_img[:, :, np.newaxis]
                if trans_img.shape[-1] == 3:
                    trans_img = np.transpose(trans_img, (2, 0, 1))
                transformed.append(trans_img)
            slabs = np.stack(transformed, axis=0)

        item: Dict[str, Any] = {
            "study_id": study_id,
            "slabs": torch.from_numpy(slabs).float(),
        }

        # Extract targets if target_cols provided or 'targets' in record
        if self.target_cols:
            targets = [float(record.get(col, 0.0)) for col in self.target_cols]
            item["targets"] = torch.tensor(targets, dtype=torch.float32)
        elif "targets" in record:
            item["targets"] = torch.as_tensor(record["targets"], dtype=torch.float32)

        return item


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs: Any,
) -> DataLoader:
    """Create a high-performance PyTorch DataLoader for RSNAKneeDataset.

    Args:
        dataset: The Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle each epoch.
        num_workers: Number of background worker threads/processes.
        pin_memory: Whether to pin memory for fast GPU page transfers.
        **kwargs: Additional arguments passed to DataLoader.

    Returns:
        Configured DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        **kwargs,
    )
