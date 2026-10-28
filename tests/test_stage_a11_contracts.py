"""LOCAL-SAFE Contracts for Stage A.11 (Clause-Sensitive Relational Learning).

Strictly enforces:
- ZERO neural forward/backward passes.
- ZERO checkpoint loading.
- ZERO CPU training.
- 100% Bag-of-Words multiset equality on counterfactual minimal pairs.
- Zero vocabulary & template leakage in Multi-Clause Pretraining corpus.
- Zero overlap with retired sealed_a10 suite ('64daf317...').
- Detached readout probe contract and index arithmetic.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from adaptive_mini_gpt.data.clause_minimal_pairs_a11 import (
    build_semantic_minimal_pairs_suite,
    verify_pair_multiset_equality,
    words_bow,
)
from adaptive_mini_gpt.data.pretraining_a11 import (
    generate_multi_clause_pretraining_corpus,
    verify_multi_clause_pretraining_corpus_isolation,
)
from adaptive_mini_gpt.data.sealed_a11 import (
    RETIRED_SEALED_A10_SHA256,
    generate_master_sealed_suite_a11,
    verify_sealed_a11_isolation,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


def test_minimal_pairs_multiset_equality():
    """Verify bit-exact Bag-of-Words multiset equality across all minimal pairs."""
    suite = build_semantic_minimal_pairs_suite()
    train_pairs = suite["train_pairs"]
    dev_pairs = suite["dev_pairs"]

    assert len(train_pairs) == 96
    assert len(dev_pairs) == 96

    for a, b in train_pairs + dev_pairs:
        assert verify_pair_multiset_equality(a, b)
        # Ensure labels are counterfactual (differ on at least one trait)
        assert a.labels != b.labels, f"Pair {a.pair_id} has identical labels!"


def test_bow_linear_mathematical_bound():
    """Mathematically proves that any Bag-of-Words linear model cannot exceed 50% pair accuracy."""
    suite = build_semantic_minimal_pairs_suite()
    dev_pairs = suite["dev_pairs"]

    # For any deterministic BoW function f(bow(x)) -> y_pred:
    # Since bow(a) == bow(b), f(bow(a)) == f(bow(b)).
    # Since y_true(a) != y_true(b), at least one member of the pair MUST be incorrectly predicted!
    # Therefore, pair_exact (which requires BOTH to be correct) is mathematically bounded to 0.0%
    # for all strictly opposing pairs, and <= 50.0% overall.
    for a, b in dev_pairs:
        bow_a = Counter(words_bow(a.question))
        bow_b = Counter(words_bow(b.question))
        assert bow_a == bow_b
        # An arbitrary single prediction for this bag-of-words
        hypothetical_pred = a.labels
        correct_a = (hypothetical_pred == a.labels)
        correct_b = (hypothetical_pred == b.labels)
        # Both CANNOT be simultaneously correct
        assert not (correct_a and correct_b)


def test_multi_clause_pretraining_isolation():
    """Verify multi-clause pretraining corpus has zero task words and zero leaks."""
    corpus = generate_multi_clause_pretraining_corpus(2000, seed=42)
    assert len(corpus) == 2000

    info = verify_multi_clause_pretraining_corpus_isolation(corpus)
    assert info["isolation_verified"]
    assert not info["forbidden_words_found"]
    assert info["corpus_sha256"] == "4cc0652f11e6476e873fac8c5cd8d4c875687a567f30383162d60785c0231205"

    # Verify presence of multi-clause contrast structures
    joined = " ".join(corpus).lower()
    for marker in (";", "whereas", "although", "while", "however"):
        assert marker in joined, f"Pretraining corpus missing relational marker: {marker}"


def test_sealed_a11_pristine_isolation_and_retirement():
    """Verify sealed_a11 retirement of sealed_a10 and complete template/entity isolation."""
    suite = generate_master_sealed_suite_a11()
    assert len(suite) == 360

    iso = verify_sealed_a11_isolation(suite)
    assert iso["isolation_verified"]
    assert iso["sealed_suite_sha256"] == "21e6ff6cd6c39289d70962c4f436e49626554d0ad32c4c732ec26ccd3b1e031b"
    assert iso["sealed_suite_sha256"] != RETIRED_SEALED_A10_SHA256

    # Verify seeds and entity names
    assert iso["seeds"] == [42, 43, 44]
    assert set(iso["names"]) == {"Drav", "Sorn", "Kael", "Vond", "Lira", "Meko"}

    # Verify 90 minimal pairs in sealed suite
    assert iso["num_counterfactual_pairs"] == 90


def test_detached_probe_feature_extraction_logic():
    """Tests h_last_question_token index arithmetic on mock sequence tensor (LOCAL-SAFE, no weights)."""
    B = 4
    L = 32
    D = 16

    tok = ByteTextTokenizerM103()
    q_id = tok.QUESTION_ID
    a_id = tok.ANSWER_ID
    pad_id = tok.PAD_ID

    mock_tokens = torch.full((B, L), pad_id, dtype=torch.long)
    # Row 0: Question at 2, Answer at 10
    mock_tokens[0, 2] = q_id
    mock_tokens[0, 3:10] = torch.arange(10, 17)
    mock_tokens[0, 10] = a_id

    # Row 1: Question at 1, Answer at 15
    mock_tokens[1, 1] = q_id
    mock_tokens[1, 2:15] = torch.arange(20, 33)
    mock_tokens[1, 15] = a_id

    # Mock hidden states: h_seq[b, l, :] = l
    mock_h_seq = torch.zeros((B, L, D), dtype=torch.float32)
    for b in range(B):
        for seq_idx in range(L):
            mock_h_seq[b, seq_idx, :] = float(seq_idx)

    # Compute h_last_question_token
    global_mask = mock_tokens != pad_id
    last_indices = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        row = mock_tokens[b]
        q_locs = (row == q_id).nonzero(as_tuple=True)[0]
        a_locs = (row == a_id).nonzero(as_tuple=True)[0]
        if len(q_locs) > 0 and len(a_locs) > 0 and a_locs[0] > q_locs[0]:
            last_indices[b] = a_locs[0] - 1
        elif len(q_locs) > 0:
            last_indices[b] = (global_mask[b].nonzero(as_tuple=True)[0])[-1]
        else:
            last_indices[b] = L - 1

    batch_idx = torch.arange(B)
    h_last_token = mock_h_seq[batch_idx, last_indices]

    # Verify row 0 took index 9 (right before answer at 10)
    assert last_indices[0].item() == 9
    assert (h_last_token[0] == 9.0).all()

    # Verify row 1 took index 14 (right before answer at 15)
    assert last_indices[1].item() == 14
    assert (h_last_token[1] == 14.0).all()


def test_provenance_reconciliation_manifest():
    """Verify reconciliation manifest maps digests and documents seed differences."""
    p = Path("data/manifests/reconciliation_a9_a10_v1.json")
    assert p.exists()
    recon = json.loads(p.read_text())

    assert recon["parent_digest_reconciliation"]["root_file_sha256"] == "3e93d7238bfdb706ddfd49ee9903fed8b1e6f577529537f1e21cae12e905b262"
    assert recon["parent_digest_reconciliation"]["root_tensor_digest_code"] == "aed7480606a1a60e38ee7a95d22b5bcf14141f034f76cb45fb59680de277a33e"
    assert len(recon["seed_42_trajectory_reconciliation"]["code_and_rng_differences"]) >= 3
