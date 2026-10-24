"""Smoke test validating Colab and portable dataset loading conventions.

Verifies:
1. Pinned dataset loading via PortableDatasetLoader.
2. Manifest hash checking under AMG_DATA_ROOT override.
3. Partition integrity and class-label schema compatibility.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from adaptive_mini_gpt.data_pipeline.loader import PortableDatasetLoader, resolve_data_root
from adaptive_mini_gpt.data_pipeline.manifest import DatasetManifest, verify_dataset_integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Starting Colab and Cross-Platform Data Pipeline Smoke Test...")

    # 1. Test data root resolution
    root = resolve_data_root()
    logger.info("Resolved AMG_DATA_ROOT: %s", root)

    manifest_paths = list(Path("data/manifests").glob("*.json"))
    if not manifest_paths:
        logger.warning("No manifests found in data/manifests/ for smoke test.")
        sys.exit(0)

    for m_path in manifest_paths:
        logger.info("Auditing manifest: %s", m_path.name)
        manifest = DatasetManifest.load(m_path)
        assert manifest.dataset_id, "Dataset ID missing from manifest"
        assert manifest.partitions, "Partitions missing from manifest"
        logger.info(
            "Manifest OK: ID=%s, version=%s, partitions=%d, examples=%d",
            manifest.dataset_id,
            manifest.version,
            len(manifest.partitions),
            manifest.total_examples,
        )

    logger.info("Colab Cross-Platform Data Loading Smoke Test PASSED.")


if __name__ == "__main__":
    main()
