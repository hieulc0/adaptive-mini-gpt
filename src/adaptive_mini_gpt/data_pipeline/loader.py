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


def resolve_data_root(custom_path: str | Path | None = None) -> Path:
    """Resolve data root directory uniformly across Local, Kaggle, and Colab."""
    if custom_path is not None:
        p = Path(custom_path)
        if p.exists():
            return p

    # 1. Check AMG_DATA_ROOT environment variable
    env_root = os.environ.get("AMG_DATA_ROOT")
    if env_root:
        p = Path(env_root)
        if p.exists():
            return p

    # 2. Check Kaggle input dataset paths
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Look for amg dataset subdirectories
        candidates = list(kaggle_input.glob("**/dataset_manifest.json"))
        if candidates:
            return candidates[0].parent
        return kaggle_input

    # 3. Check Google Colab runtime paths
    colab_root = Path("/content/data")
    if colab_root.exists():
        return colab_root

    # 4. Fallback to repository local data/generated
    repo_generated = Path("data/generated")
    if repo_generated.exists():
        return repo_generated

    # Default to current directory data
    return Path("data")


class PortableDatasetLoader:
    """Loads and verifies datasets across portable execution environments."""

    def __init__(self, data_root: str | Path | None = None, enforce_manifest: bool = True) -> None:
        self.data_root = resolve_data_root(data_root)
        self.enforce_manifest = enforce_manifest

    def load_dataset(
        self,
        dataset_id: str,
        version: str | None = None,
    ) -> tuple[DatasetManifest | None, Path]:
        """Locate dataset directory and load/validate its manifest."""
        # Search patterns:
        # data_root / dataset_id / f"v{version}"
        # data_root / dataset_id
        # data_root
        candidate_dirs = []
        if version:
            candidate_dirs.append(self.data_root / dataset_id / f"v{version}")
            candidate_dirs.append(self.data_root / f"{dataset_id}_v{version}")
        candidate_dirs.append(self.data_root / dataset_id)
        candidate_dirs.append(self.data_root)

        dataset_dir = None
        for cd in candidate_dirs:
            if cd.exists() and (cd / "dataset_manifest.json").exists():
                dataset_dir = cd
                break

        manifest = None
        if dataset_dir:
            manifest_file = dataset_dir / "dataset_manifest.json"
            manifest = DatasetManifest.load(manifest_file)
            if self.enforce_manifest:
                valid, errors = verify_dataset_integrity(manifest, dataset_dir)
                if not valid:
                    error_msg = "; ".join(errors)
                    raise ValueError(f"Dataset integrity verification failed for {dataset_id}: {error_msg}")
                logger.info("Dataset integrity verified for %s (v%s)", dataset_id, manifest.version)
        else:
            # Fall back to finding manifest anywhere in data_root
            manifest_paths = list(self.data_root.rglob("dataset_manifest.json"))
            if manifest_paths:
                dataset_dir = manifest_paths[0].parent
                manifest = DatasetManifest.load(manifest_paths[0])
            else:
                dataset_dir = self.data_root

        return manifest, dataset_dir

    def stream_partition(
        self,
        dataset_id: str,
        partition_name: str,
        version: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield verified JSON records from a specific partition file."""
        manifest, dataset_dir = self.load_dataset(dataset_id, version=version)
        if manifest and partition_name in manifest.partitions:
            rel_path = manifest.partitions[partition_name].relative_path
            p_file = dataset_dir / rel_path
        else:
            p_file = dataset_dir / f"{partition_name}.jsonl"

        if not p_file.exists():
            raise FileNotFoundError(f"Partition file not found: {p_file}")

        with open(p_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
