"""Dataset manifest schema, hashing, and verification utilities for M70 & M71."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    """Compute standard SHA-256 hex digest for a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass
class DatasetPartitionInfo:
    """Metadata for a single dataset partition file (train, dev, sealed_test)."""
    partition_name: str
    relative_path: str
    sha256: str
    example_count: int
    label_distribution: dict[str, int]


@dataclass
class DatasetFamilyInfo:
    """Metadata describing a relational task family."""
    family_id: str
    rule_name: str = ""
    relation_name: str = ""
    description: str = ""
    entity_universe_size: int = 0


@dataclass
class DatasetManifest:
    """Formal dataset manifest tracking provenance, cryptographic hashes, and integrity."""
    schema_version: str
    dataset_id: str
    version: str
    dataset_type: str  # "synthetic", "external", "mixed"
    git_commit: str
    generator_config_hash: str
    seed: int
    independent_world_ids: list[str]
    task_families: list[DatasetFamilyInfo]
    partitions: dict[str, DatasetPartitionInfo]
    total_examples: int
    created_at: str
    license: str = "CC-BY-4.0"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return asdict(self)

    def save(self, path: Path) -> None:
        """Write manifest formatted as JSON to path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        """Load manifest from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        known_family_fields = {"family_id", "rule_name", "relation_name", "description", "entity_universe_size"}
        families = [
            DatasetFamilyInfo(**{k: v for k, v in fam.items() if k in known_family_fields})
            for fam in data.get("task_families", [])
        ]

        known_partition_fields = {"partition_name", "relative_path", "sha256", "example_count", "label_distribution"}
        partitions = {
            p_name: DatasetPartitionInfo(**{k: v for k, v in p_info.items() if k in known_partition_fields})
            for p_name, p_info in data.get("partitions", {}).items()
        }

        return cls(
            schema_version=data.get("schema_version", "1.0"),
            dataset_id=data["dataset_id"],
            version=data.get("version", "1.0"),
            dataset_type=data.get("dataset_type", "synthetic"),
            git_commit=data.get("git_commit", "unknown"),
            generator_config_hash=data.get("generator_config_hash", ""),
            seed=data.get("seed", 42),
            independent_world_ids=data.get("independent_world_ids", []),
            task_families=families,
            partitions=partitions,
            total_examples=data.get("total_examples", 0),
            created_at=data.get("created_at", ""),
            license=data.get("license", "CC-BY-4.0"),
            notes=data.get("notes", ""),
        )


def verify_dataset_integrity(
    manifest: DatasetManifest,
    dataset_dir: Path,
) -> tuple[bool, list[str]]:
    """Cryptographically verify all partition files, checksums, and check split disjointness.

    Returns:
        (is_valid, list of error messages)
    """
    errors: list[str] = []

    # 1. Verify existence and SHA-256 for each partition
    seen_examples: dict[str, str] = {}
    seen_worlds: dict[str, str] = {}

    for p_name, p_info in manifest.partitions.items():
        file_path = dataset_dir / p_info.relative_path
        if not file_path.exists():
            errors.append(f"Missing partition file for '{p_name}': {file_path}")
            continue

        actual_digest = compute_file_sha256(file_path)
        if actual_digest != p_info.sha256:
            errors.append(
                f"Checksum mismatch for partition '{p_name}' ({file_path}): "
                f"expected {p_info.sha256}, got {actual_digest}"
            )

        # 2. Check example count and audit split disjointness
        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception as e:
                    errors.append(f"Corrupted JSON on line {line_idx} in {file_path}: {e}")
                    continue
                count += 1
                q_text = record.get("question_text", record.get("text", ""))
                if q_text:
                    if q_text in seen_examples:
                        prev_p = seen_examples[q_text]
                        if prev_p != p_name:
                            errors.append(
                                f"Data leakage: example '{q_text}' appears in multiple partitions ('{prev_p}' and '{p_name}')!"
                            )
                        else:
                            errors.append(
                                f"Duplicate example within partition '{p_name}': '{q_text}'"
                            )
                    seen_examples[q_text] = p_name

                # World-split isolation check
                w_id = record.get("world_id")
                if w_id:
                    if p_name.startswith("train") and "sealed" in seen_worlds.get(w_id, ""):
                        errors.append(f"World leakage: world '{w_id}' appears in both training and sealed partitions!")
                    seen_worlds[w_id] = p_name

        if count != p_info.example_count:
            errors.append(
                f"Example count mismatch for partition '{p_name}': "
                f"manifest says {p_info.example_count}, found {count}"
            )

    is_valid = (len(errors) == 0)
    return is_valid, errors
