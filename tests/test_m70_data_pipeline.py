"""Unit tests for Phase 14: Data Pipeline & Portable Manifest Infrastructure."""

import json
import os
import tempfile
from pathlib import Path
import pytest

from adaptive_mini_gpt.data_pipeline import (
    DatasetManifest,
    DatasetPartitionInfo,
    PortableDatasetLoader,
    compute_file_sha256,
    format_colab_download_snippet,
    generate_synthetic_dataset,
    prepare_kaggle_dataset_package,
    resolve_data_root,
    verify_dataset_integrity,
)


def test_manifest_verification_and_tamper_detection(tmp_path: Path):
    """Verify that dataset manifest accurately detects matching files and catches tampering."""
    # Create sample partition file
    p_file = tmp_path / "train.jsonl"
    sample_data = [
        {"sample_id": "s1", "question_text": "Is A true? Answer:", "expected_answer": "YES"},
        {"sample_id": "s2", "question_text": "Is B true? Answer:", "expected_answer": "NO"},
    ]
    with open(p_file, "w", encoding="utf-8") as f:
        for r in sample_data:
            f.write(json.dumps(r) + "\n")

    digest = compute_file_sha256(p_file)

    manifest = DatasetManifest(
        schema_version="1.0",
        dataset_id="test_dataset",
        version="1.0",
        dataset_type="synthetic",
        git_commit="test_commit",
        generator_config_hash="cfg_hash",
        seed=42,
        independent_world_ids=["w1"],
        task_families=[],
        partitions={
            "train": DatasetPartitionInfo(
                partition_name="train",
                relative_path="train.jsonl",
                sha256=digest,
                example_count=2,
                label_distribution={"YES": 1, "NO": 1},
            )
        },
        total_examples=2,
        created_at="2026-10-24T00:00:00Z",
    )

    # 1. Valid verification
    is_valid, errors = verify_dataset_integrity(manifest, tmp_path)
    assert is_valid is True
    assert len(errors) == 0

    # 2. Tampering test: append corrupt line
    with open(p_file, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": "s3", "question_text": "Is C true?", "expected_answer": "UNKNOWN"}) + "\n")

    is_valid_tampered, errors_tampered = verify_dataset_integrity(manifest, tmp_path)
    assert is_valid_tampered is False
    assert any("Checksum mismatch" in err or "Example count mismatch" in err for err in errors_tampered)


def test_deterministic_synthetic_generation(tmp_path: Path):
    """Verify deterministic dataset generation from spec produces verifiable files and manifest."""
    spec_path = Path("data/specs/m70_relational_benchmark_spec.json")
    if not spec_path.exists():
        pytest.skip("m70_relational_benchmark_spec.json not found")

    manifest, dataset_dir = generate_synthetic_dataset(
        spec_path=spec_path,
        output_base_dir=tmp_path,
        seed=42,
        git_commit="test_commit_hash",
    )

    assert manifest.dataset_id == "m70_relational_benchmark"
    assert manifest.version == "1.0"
    assert (dataset_dir / "train.jsonl").exists()
    assert (dataset_dir / "dev.jsonl").exists()
    assert (dataset_dir / "sealed_test.jsonl").exists()
    assert (dataset_dir / "dataset_manifest.json").exists()

    # Re-verify generated dataset integrity
    is_valid, errors = verify_dataset_integrity(manifest, dataset_dir)
    assert is_valid is True
    assert len(errors) == 0


def test_portable_path_resolution(monkeypatch, tmp_path: Path):
    """Verify resolution of data root via AMG_DATA_ROOT and default fallbacks."""
    # Test environment variable override
    custom_root = tmp_path / "custom_data"
    custom_root.mkdir()
    monkeypatch.setenv("AMG_DATA_ROOT", str(custom_root))

    resolved = resolve_data_root()
    assert resolved == custom_root

    # Test explicit custom path argument
    another_root = tmp_path / "another_data"
    another_root.mkdir()
    resolved_explicit = resolve_data_root(another_root)
    assert resolved_explicit == another_root


def test_kaggle_packaging_security_and_colab_snippet(tmp_path: Path):
    """Verify Kaggle upload package creation, metadata generation, and code leakage rejection."""
    # Create clean mock dataset
    data_dir = tmp_path / "clean_data"
    data_dir.mkdir()
    p_file = data_dir / "train.jsonl"
    with open(p_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"question_text": "q1", "expected_answer": "YES"}) + "\n")

    manifest = DatasetManifest(
        schema_version="1.0",
        dataset_id="secure_dataset",
        version="1.0",
        dataset_type="synthetic",
        git_commit="c1",
        generator_config_hash="h1",
        seed=42,
        independent_world_ids=["w1"],
        task_families=[],
        partitions={
            "train": DatasetPartitionInfo(
                partition_name="train",
                relative_path="train.jsonl",
                sha256=compute_file_sha256(p_file),
                example_count=1,
                label_distribution={"YES": 1},
            )
        },
        total_examples=1,
        created_at="2026-10-24T00:00:00Z",
    )
    manifest.save(data_dir / "dataset_manifest.json")

    pkg_dir = tmp_path / "packages"
    target_pkg = prepare_kaggle_dataset_package(
        dataset_dir=data_dir,
        output_package_dir=pkg_dir,
        owner_slug="sparklinghydraulics0",
        is_private=True,
    )

    assert (target_pkg / "dataset-metadata.json").exists()
    assert (target_pkg / "train.jsonl").exists()
    assert (target_pkg / "dataset_manifest.json").exists()

    with open(target_pkg / "dataset-metadata.json") as f:
        meta = json.load(f)
    assert meta["isPrivate"] is True
    assert "sparklinghydraulics0/secure-dataset" in meta["id"]

    # Test security audit catches forbidden files
    bad_file = data_dir / "secret.env"
    bad_file.write_text("SECRET=123")
    with pytest.raises(ValueError, match="Security error"):
        prepare_kaggle_dataset_package(
            dataset_dir=data_dir,
            output_package_dir=pkg_dir,
            owner_slug="sparklinghydraulics0",
        )

    # Test Colab snippet generator
    snippet = format_colab_download_snippet(
        owner_slug="sparklinghydraulics0",
        dataset_id="secure_dataset",
        version="1",
    )
    assert "kagglehub.dataset_download" in snippet
    assert "AMG_DATA_ROOT" in snippet
