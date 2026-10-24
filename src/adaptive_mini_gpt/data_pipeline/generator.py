"""Deterministic synthetic dataset generator for Adaptive Mini-GPT benchmarks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from adaptive_mini_gpt.data_pipeline.manifest import (
    DatasetFamilyInfo,
    DatasetManifest,
    DatasetPartitionInfo,
    compute_file_sha256,
)

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    spec_path: Path,
    output_base_dir: Path,
    seed: int = 42,
    git_commit: str = "unknown",
) -> tuple[DatasetManifest, Path]:
    """Generate deterministic synthetic dataset from JSON specification.

    Args:
        spec_path: Path to dataset specification JSON.
        output_base_dir: Directory where generated dataset folder will be created.
        seed: Random seed for deterministic generation.
        git_commit: Git commit hash for provenance tracking.

    Returns:
        Tuple of (generated DatasetManifest, dataset directory Path).
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found at {spec_path}")

    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    dataset_id = spec.get("dataset_id") or spec.get("spec_id") or "synthetic_relational_dataset"
    version = spec.get("version", "1.0")
    dataset_type = spec.get("dataset_type", "synthetic")
    license_str = spec.get("license", "CC-BY-4.0")
    config_hash = compute_file_sha256(spec_path)

    dataset_dir = output_base_dir / f"{dataset_id}_v{version}_seed{seed}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    independent_worlds = spec.get("independent_worlds", ["world_alpha", "world_beta", "world_gamma"])
    families = spec.get("task_families") or spec.get("families") or []
    if not families:
        raise ValueError(f"Specification {spec_path} must define non-empty 'families' or 'task_families'")

    partition_sizes = spec.get("partition_sizes", {"train": 60, "dev": 20, "sealed_test": 20})

    family_infos: list[DatasetFamilyInfo] = []
    for fam in families:
        family_infos.append(
            DatasetFamilyInfo(
                family_id=fam["family_id"],
                rule_name=fam.get("rule_name", fam["family_id"]),
                relation_name=fam.get("relation_name", "relates_to"),
                description=fam.get("description", ""),
            )
        )

    # Deterministic generation loop
    rng = np.random.default_rng(seed)

    partitions: dict[str, DatasetPartitionInfo] = {}
    labels = ["NO", "YES", "UNKNOWN"]
    total_samples = 0

    # Ensure completely disjoint entity namespaces for each partition
    partition_prefixes = {
        "train": "TrEntity",
        "dev": "DevEntity",
        "sealed_test": "SealedTestEntity",
    }

    for p_name, n_samples in partition_sizes.items():
        p_file = dataset_dir / f"{p_name}.jsonl"
        label_dist = {lbl: 0 for lbl in labels}
        prefix = partition_prefixes.get(p_name, f"{p_name}Entity")

        with open(p_file, "w", encoding="utf-8") as out_f:
            for i in range(n_samples):
                fam = families[i % len(families)]
                fam_id = fam["family_id"]
                rel = fam.get("relation_name", "relates_to")

                # Ensure unique entity pair for every sample to avoid duplicate questions
                src_node = (i * 2) % 10000
                tgt_node = (i * 2 + 1) % 10000

                src = f"{prefix}_{fam_id[:4]}_{src_node:04d}"
                tgt = f"{prefix}_{fam_id[:4]}_{tgt_node:04d}"

                # Deterministic label choice based on parity/modulo
                expected = labels[(src_node + tgt_node + i) % 3]
                label_dist[expected] += 1

                evidence = f"Evidence: [{src}->{tgt}, Rule={fam_id}]."
                q_text = f"{evidence} Does {src} {rel} {tgt}? Answer:"

                record = {
                    "sample_id": f"{dataset_id}_{p_name}_{i:04d}",
                    "partition": p_name,
                    "family_id": fam_id,
                    "source_entity": src,
                    "target_entity": tgt,
                    "relation_name": rel,
                    "question_text": q_text,
                    "expected_answer": expected,
                    "seed": seed,
                }
                out_f.write(json.dumps(record) + "\n")

        p_sha = compute_file_sha256(p_file)
        partitions[p_name] = DatasetPartitionInfo(
            partition_name=p_name,
            relative_path=f"{p_name}.jsonl",
            sha256=p_sha,
            example_count=n_samples,
            label_distribution=label_dist,
        )
        total_samples += n_samples

    manifest = DatasetManifest(
        schema_version="1.0",
        dataset_id=dataset_id,
        version=version,
        dataset_type=dataset_type,
        git_commit=git_commit,
        generator_config_hash=config_hash,
        seed=seed,
        independent_world_ids=independent_worlds,
        task_families=family_infos,
        partitions=partitions,
        total_examples=total_samples,
        created_at="2026-10-24T00:00:00Z",
        license=license_str,
    )

    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest.save(manifest_path)
    logger.info("Successfully generated dataset %s with %d samples at %s", dataset_id, total_samples, dataset_dir)
    return manifest, dataset_dir
