"""Packaging utilities for private Kaggle Datasets and Google Colab integration."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from adaptive_mini_gpt.data_pipeline.manifest import DatasetManifest

logger = logging.getLogger(__name__)


def prepare_kaggle_dataset_package(
    dataset_dir: Path,
    output_package_dir: Path,
    owner_slug: str,
    dataset_title: str | None = None,
    is_private: bool = True,
) -> Path:
    """Prepare a lean, private Kaggle dataset package from a verified dataset directory.

    Audits against secret, code, and checkpoint leakage, and generates `dataset-metadata.json`.

    Args:
        dataset_dir: Directory containing partition files and `dataset_manifest.json`.
        output_package_dir: Base directory where package bundle will be constructed.
        owner_slug: Kaggle username/organization slug (e.g., 'sparklinghydraulics0').
        dataset_title: Human-readable title for Kaggle dataset.
        is_private: Whether dataset must be private on Kaggle (default True).

    Returns:
        Path to constructed package folder ready for `kaggle datasets create -p <path>`.
    """
    manifest_file = dataset_dir / "dataset_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"dataset_manifest.json not found in {dataset_dir}")

    # Audit source directory against secret/code file leakage before creating package
    forbidden_extensions = {".py", ".sh", ".env", ".key", ".pem", ".pt", ".pth", ".bin"}
    for f in dataset_dir.rglob("*"):
        if f.is_file() and f.suffix in forbidden_extensions:
            raise ValueError(f"Security error: unauthorized file {f.name} in source data directory!")

    manifest = DatasetManifest.load(manifest_file)
    pkg_name = f"{manifest.dataset_id}_v{manifest.version}"
    target_dir = output_package_dir / pkg_name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy partition files and manifest only
    shutil.copy2(manifest_file, target_dir / "dataset_manifest.json")
    for p_info in manifest.partitions.values():
        src_file = dataset_dir / p_info.relative_path
        if src_file.exists():
            shutil.copy2(src_file, target_dir / p_info.relative_path)
        else:
            raise FileNotFoundError(f"Partition file missing: {src_file}")

    # 2. Audit target bundle
    for f in target_dir.rglob("*"):
        if f.is_file() and f.suffix in forbidden_extensions:
            raise ValueError(f"Security error: unauthorized file {f.name} in data package!")

    # 3. Create dataset-metadata.json for Kaggle CLI
    title = dataset_title or f"Adaptive Mini-GPT {manifest.dataset_id.replace('_', ' ').title()}"
    dataset_slug = f"{owner_slug}/{manifest.dataset_id.replace('_', '-')}"
    metadata = {
        "title": title,
        "id": dataset_slug,
        "licenses": [{"name": manifest.license.lower()}],
        "isPrivate": is_private,
    }

    metadata_path = target_dir / "dataset-metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as meta_f:
        json.dump(metadata, meta_f, indent=2)

    logger.info("Created private Kaggle dataset package at %s", target_dir)
    return target_dir


def format_colab_download_snippet(
    owner_slug: str,
    dataset_id: str,
    version: str = "1",
) -> str:
    """Format python snippet to download and verify dataset in Google Colab via kagglehub."""
    dataset_handle = f"{owner_slug}/{dataset_id.replace('_', '-')}/versions/{version}"
    snippet = f'''
# --- Google Colab Pinned Dataset Loading ---
import os
from pathlib import Path
import kagglehub

# 1. Download pinned dataset version to Colab runtime-local storage
dataset_path = kagglehub.dataset_download("{dataset_handle}")
print("Downloaded pinned dataset to:", dataset_path)

# 2. Set AMG_DATA_ROOT for portable data loader
os.environ["AMG_DATA_ROOT"] = dataset_path

# 3. Load and verify manifest
from adaptive_mini_gpt.data_pipeline import PortableDatasetLoader
loader = PortableDatasetLoader()
dataset_dir = loader.get_dataset_dir("{dataset_id}", "{version}")
manifest = loader.load_manifest(dataset_dir)
is_valid, errors = loader.verify(manifest, dataset_dir)
if not is_valid:
    raise RuntimeError(f"Dataset integrity verification failed: {{errors}}")
print("Verified dataset integrity successfully:", manifest.dataset_id, manifest.version)
'''
    return snippet.strip()
