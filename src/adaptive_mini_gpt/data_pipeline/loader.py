"""Phase 8: Portable Dataset Loading Interface.

Provides uniform, platform-agnostic data resolution across:
- Local development (reads AMG_DATA_ROOT or data/generated/)
- Kaggle GPU runtime (detects attached datasets in /kaggle/input/ or local runtime paths)
- Google Colab runtime (detects /content/data or runtime cache)

Validates dataset manifests and SHA-256 digests prior to data consumption.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Iterator

from adaptive_mini_gpt.data_pipeline.manifest import (
    DatasetManifest,
    verify_dataset_integrity,
)

logger = logging.getLogger(__name__)


def resolve_data_root(custom_path: str | Path | None = None, writable: bool = False) -> Path:
    """Resolve data root directory uniformly across Local, Kaggle, and Colab."""
    if custom_path is not None:
        p = Path(custom_path)
        return p

    # 1. Check AMG_DATA_ROOT environment variable
    env_root = os.environ.get("AMG_DATA_ROOT")
    if env_root:
        return Path(env_root)

    # 2. If caller needs a writable directory for generating data
    if writable:
        if Path("/kaggle/working").exists():
            return Path("/kaggle/working/data/generated")
        if Path("/content").exists():
            return Path("/content/data")
        return Path("data/generated")

    # 3. Check Kaggle input dataset paths (read-only)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates = list(kaggle_input.glob("**/dataset_manifest.json"))
        if candidates:
            return candidates[0].parent

    # 4. Check Google Colab runtime paths
    colab_root = Path("/content/data")
    if colab_root.exists():
        return colab_root

    # 5. Default local development path
    return Path("data/generated")


class PortableDatasetLoader:
    """Cross-platform loader for versioned datasets with cryptographic integrity verification."""

    def __init__(self, base_data_root: Path | None = None) -> None:
        self.data_root = base_data_root or resolve_data_root()

    def get_dataset_dir(self, dataset_id: str, version: str = "1.0", seed: int | None = None) -> Path:
        """Find dataset directory under resolved data root."""
        pattern = f"{dataset_id}_v{version}*" if seed is None else f"{dataset_id}_v{version}_seed{seed}"
        matches = list(self.data_root.glob(pattern))
        if matches and matches[0].is_dir():
            return matches[0]

        direct = self.data_root / f"{dataset_id}_v{version}"
        if direct.exists() and direct.is_dir():
            return direct

        # Fallback to direct data root if dataset_manifest.json exists directly there
        if (self.data_root / "dataset_manifest.json").exists():
            return self.data_root

        raise FileNotFoundError(
            f"Dataset '{dataset_id}' version '{version}' not found under data root: {self.data_root}"
        )

    def load_manifest(self, dataset_dir: Path) -> DatasetManifest:
        """Load manifest from a verified dataset directory."""
        manifest_file = dataset_dir / "dataset_manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"dataset_manifest.json not found in {dataset_dir}")
        return DatasetManifest.load(manifest_file)

    def verify(self, manifest: DatasetManifest, dataset_dir: Path) -> tuple[bool, list[str]]:
        """Verify partition files, SHA-256 digests, and split disjointness."""
        return verify_dataset_integrity(manifest, dataset_dir)

    def load_and_verify(
        self,
        dataset_id: str,
        version: str = "1.0",
        seed: int | None = None,
    ) -> tuple[DatasetManifest, Path]:
        """Convenience method: resolve, load manifest, and cryptographically verify integrity."""
        dataset_dir = self.get_dataset_dir(dataset_id, version, seed)
        manifest = self.load_manifest(dataset_dir)
        is_valid, errors = self.verify(manifest, dataset_dir)
        if not is_valid:
            raise RuntimeError(
                f"Dataset integrity verification failed for '{dataset_id}' at {dataset_dir}: {errors}"
            )
        logger.info("Successfully verified dataset %s v%s at %s", dataset_id, version, dataset_dir)
        return manifest, dataset_dir

    def stream_partition(
        self,
        manifest: DatasetManifest,
        dataset_dir: Path,
        partition_name: str,
    ) -> Iterator[dict[str, Any]]:
        """Stream JSON records from a specific partition file."""
        if partition_name not in manifest.partitions:
            raise KeyError(f"Partition '{partition_name}' not defined in manifest for {manifest.dataset_id}")

        p_info = manifest.partitions[partition_name]
        p_path = dataset_dir / p_info.relative_path
        if not p_path.exists():
            raise FileNotFoundError(f"Partition file missing: {p_path}")

        with open(p_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
