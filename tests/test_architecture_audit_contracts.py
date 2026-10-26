"""LOCAL-SAFE: pure token fixtures and mocked methods; no neural execution."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adaptive_mini_gpt.model.transformer import MiniGPT
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import TextStudentM103
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import ByteTextTokenizerM103


def test_prompt_has_real_boundaries_and_complete_question():
    tok = ByteTextTokenizerM103()
    ids = tok.encode_prompt(
        question="A wooden tool.", explanation="Fire burns wood.", max_length=64
    )
    assert ids.count(tok.QUESTION_ID) == 1
    assert ids.count(tok.RULE_ID) == 1
    assert ids[-1] == tok.ANSWER_ID
    q = ids.index(tok.QUESTION_ID)
    assert tok.decode(ids[q + 1 : -1]) == "A wooden tool."


def test_prompt_overflow_is_explicit():
    with pytest.raises(ValueError, match="exceeds"):
        ByteTextTokenizerM103().encode_prompt(
            question="entity", explanation="x" * 300, max_length=256
        )


def test_literal_special_token_text_remains_bytes():
    # Preserve raw text API and existing checkpoint/tokenizer meaning.
    tok = ByteTextTokenizerM103()
    assert tok.QUESTION_ID not in tok.encode("<question>")


def test_exact_match_rejects_substring_answer():
    fake = SimpleNamespace(
        generate=lambda **kwargs: list(b"wrong 4 energy extra"), tokenizer=ByteTextTokenizerM103()
    )
    item = SimpleNamespace(prompt_len=1, full_tokens=[257], answer="4 energy")
    assert TextStudentM103.evaluate_exact_match(fake, [item]) == 0.0


def test_optimizer_excludes_all_embeddings_and_deduplicates_tied_weight():
    class Parameter:
        requires_grad = True

        def __init__(self, dim):
            self.ndim = dim

        def dim(self):
            return self.ndim

    token, position, dense, bias = Parameter(2), Parameter(2), Parameter(2), Parameter(1)
    # Real module type checks, fake parameters; no tensors or model construction.
    from torch import nn

    embedding = object.__new__(nn.Embedding)
    object.__setattr__(embedding, "weight", token)
    position_embedding = object.__new__(nn.Embedding)
    object.__setattr__(position_embedding, "weight", position)
    fake = SimpleNamespace(
        _embedding_grad_mask=None,
        named_parameters=lambda: iter(
            [
                ("wte.weight", token),
                ("wpe.weight", position),
                ("dense.weight", dense),
                ("dense.bias", bias),
            ]
        ),
        modules=lambda: iter([embedding, position_embedding]),
    )
    with patch(
        "adaptive_mini_gpt.model.transformer.AdamW", side_effect=lambda groups, **kw: groups
    ):
        groups = MiniGPT.configure_optimizers(fake, 0.001, 0.1)
    assert groups[0]["params"] == [dense]
    assert groups[1]["params"] == [token, position, bias]


def test_unknown_transfer_label_rejected_before_tensor_encoding():
    from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m104_e import (
        ConceptLearnerM104_E,
    )

    fake = SimpleNamespace(
        action_vocab={"Apply_Fire": 0}, state_vocab={"Dormant": 0}, tag_vocab={}, device="cpu"
    )
    item = SimpleNamespace(
        action_name="Kindle",
        expected_state="Molten",
        rule_explanation="rule",
        question_text="question",
        object_name="obj",
    )
    with pytest.raises(ValueError, match="Unknown"):
        ConceptLearnerM104_E._encode_batch_tensors(fake, [item], {}, ByteTextTokenizerM103())


def test_paraphrase_split_uses_paraphrase_text():
    from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1 import (
        CausalItemM103_G1,
    )
    from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
        ProceduralCausalWorldM104_B,
    )

    item = CausalItemM103_G1(
        "fixture",
        "novel_entity",
        "obj",
        "Fire",
        None,
        "Dormant",
        "Ignited",
        "rule",
        "original",
        "Ignited",
        "family",
    )
    fake = SimpleNamespace(
        objects={"obj": SimpleNamespace(is_novel=True, description="a tool")},
        rng=SimpleNamespace(randint=lambda *args: 0, shuffle=lambda x: None),
        PARAPHRASE_TEMPLATES=["Alternate {desc} {act}"],
        ACTIONS=[],
        world_id="fixture",
    )
    world = object.__new__(ProceduralCausalWorldM104_B)
    world.__dict__.update(fake.__dict__)
    with patch(
        "adaptive_mini_gpt.natural_language_learning_m103.causal_world_m103_g1.ProceduralCausalWorldM103_G1.generate_datasets",
        return_value={"novel_entity": [item]},
    ):
        result = ProceduralCausalWorldM104_B.generate_datasets_m104b(world, candidate_pool_size=0)
    assert result["paraphrase_test"][0].question_text == "Alternate a tool Fire"
    assert result["novel_entity"][0].question_text == "original"


def test_remote_guard_rejects_local_cuda_and_remote_cpu():
    from adaptive_mini_gpt.training.remote_runtime import require_remote_cuda

    with (
        patch("adaptive_mini_gpt.training.remote_runtime.Path.is_dir", return_value=False),
        patch("torch.cuda.is_available", return_value=True),
    ):
        with pytest.raises(RuntimeError, match="refused"):
            require_remote_cuda()
    with (
        patch("adaptive_mini_gpt.training.remote_runtime.Path.is_dir", return_value=True),
        patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Batch"}),
        patch("torch.cuda.is_available", return_value=True),
    ):
        assert require_remote_cuda() == "kaggle"
        with pytest.raises(RuntimeError, match="refused"):
            require_remote_cuda("cpu")


def test_transaction_evaluation_error_returns_rejected_result():
    from adaptive_mini_gpt.natural_language_learning_m103.self_modification_m105_a import (
        ModificationOperatorTypeM105_A,
        NeuralModificationProposalM105_A,
        StructuralRollbackControllerM105_A,
    )

    ctrl = StructuralRollbackControllerM105_A(SimpleNamespace(parameters=lambda: []))
    ctrl.create_snapshot = lambda: None
    ctrl.restore_snapshot = lambda: None
    values = iter([1.0])
    proposal = NeuralModificationProposalM105_A(
        "fixture",
        ModificationOperatorTypeM105_A.SELECTIVE_BACKBONE_UPDATE,
        "fixture",
        {},
        "fixture",
    )
    result = ctrl.apply_and_evaluate(proposal, val_eval_fn=lambda: next(values))
    assert not result.accepted and result.rollback_executed
    assert result.validation_status.startswith("ERROR_ROLLED_BACK")


def test_unsupported_structural_operator_cannot_commit_noop():
    from adaptive_mini_gpt.natural_language_learning_m103.self_modification_m105_a import (
        ModificationOperatorTypeM105_A,
        NeuralModificationProposalM105_A,
        StructuralRollbackControllerM105_A,
    )

    ctrl = StructuralRollbackControllerM105_A(SimpleNamespace(parameters=lambda: []))
    ctrl.create_snapshot = lambda: None
    ctrl.restore_snapshot = lambda: None
    proposal = NeuralModificationProposalM105_A(
        "fixture", ModificationOperatorTypeM105_A.DYNAMIC_WIDTH_EXPANSION, "fixture", {}, "fixture"
    )
    result = ctrl.apply_and_evaluate(proposal, val_eval_fn=lambda: 1.0)
    assert not result.accepted


def test_padding_must_not_discard_observed_tokens():
    with pytest.raises(ValueError, match="exceeds"):
        ByteTextTokenizerM103().pad_batch([[257, 261, 65, 262]], max_length=3)


def test_prompt_exact_boundary_and_extensible_budget():
    tok = ByteTextTokenizerM103()
    full, n = tok.format_qa_sequence("é" * 140, "", "rule")
    assert tok.encode_prompt("é" * 140, "rule", max_length=n) == full[:n]
    with pytest.raises(ValueError, match="exceeds"):
        tok.encode_prompt("é" * 140, "rule", max_length=n - 1)


def test_independent_rng_stream_restores_ambient_state():
    import random

    from adaptive_mini_gpt.training.reproducibility import isolated_rng

    with patch("torch.cuda.is_available", return_value=False):
        before = random.getstate()
        with isolated_rng(42, "fixture"):
            first = random.random()
        assert random.getstate() == before
        random.random()
        with isolated_rng(42, "fixture"):
            assert random.random() == first


def test_optimizer_membership_reports_dynamic_missing_parameter():
    from adaptive_mini_gpt.training.optimizer_contract import optimizer_membership

    # Parameters are fake objects; no neural initialization or optimizer steps.
    class Param:
        requires_grad = True

    old, new = Param(), Param()
    model = SimpleNamespace(
        parameters=lambda: iter([old, new]),
        named_parameters=lambda: iter([("old", old), ("new", new)]),
    )
    optimizer = SimpleNamespace(param_groups=[{"params": [old]}])
    assert optimizer_membership(model, optimizer)["missing_trainable"] == ["new"]


def test_complete_protocol_retains_whitespace_and_rejects_missing_evidence():
    from adaptive_mini_gpt.natural_language_learning_m103.input_protocol import (
        ObservedCase,
        encode_observation,
        validate_cases,
    )

    case = ObservedCase("x", "  wood\n", " rules ", "fire", ("wood",), "hot", {"wood": "wood"})
    ids = encode_observation(case, ByteTextTokenizerM103(), 512)
    assert 261 in ids and 260 in ids
    validate_cases([case], {"wood": 0}, {"fire": 0}, {"hot": 0})
    from dataclasses import replace

    with pytest.raises(ValueError, match="evidence"):
        validate_cases([replace(case, question="metal")], {"wood": 0}, {"fire": 0}, {"hot": 0})
    with pytest.raises(ValueError, match="Contradictory"):
        validate_cases([replace(case, question="not wood")], {"wood": 0}, {"fire": 0}, {"hot": 0})
