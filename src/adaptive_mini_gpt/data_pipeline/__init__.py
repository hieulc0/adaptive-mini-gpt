"""Data Pipeline Package for Portable, Manifest-Verified Dataset Management."""

from adaptive_mini_gpt.data_pipeline.manifest import (
    DatasetFamilyInfo,
    DatasetManifest,
    DatasetPartitionInfo,
    compute_file_sha256,
    verify_dataset_integrity,
)
from adaptive_mini_gpt.data_pipeline.loader import (
    PortableDatasetLoader,
    resolve_data_root,
)
from adaptive_mini_gpt.data_pipeline.generator import (
    generate_synthetic_dataset,
)
from adaptive_mini_gpt.data_pipeline.packaging import (
    prepare_kaggle_dataset_package,
    format_colab_download_snippet,
)

__all__ = [
    "DatasetFamilyInfo",
    "DatasetManifest",
    "DatasetPartitionInfo",
    "compute_file_sha256",
    "verify_dataset_integrity",
    "PortableDatasetLoader",
    "resolve_data_root",
    "generate_synthetic_dataset",
    "prepare_kaggle_dataset_package",
    "format_colab_download_snippet",
]
