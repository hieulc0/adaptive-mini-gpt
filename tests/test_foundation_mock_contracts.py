"""LOCAL-SAFE: inspected policy/manifest fixtures, no tensors or neural execution."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from adaptive_mini_gpt.training.legacy_entrypoints import require_historical_reproduction


def test_unsafe_historical_cli_requires_explicit_remote_override():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(RuntimeError, match="unsupported"):
            require_historical_reproduction("historical")
    with patch.dict("os.environ", {"AMG_ALLOW_UNSUPPORTED_HISTORICAL": "1"}):
        with patch(
            "adaptive_mini_gpt.training.legacy_entrypoints.require_remote_cuda",
            side_effect=RuntimeError("remote only"),
        ):
            with pytest.raises(RuntimeError, match="remote only"):
                require_historical_reproduction("historical")


def test_all_listed_quarantined_clis_enforce_the_boundary():
    manifest = json.loads(Path("data/manifests/legacy_foundation_entrypoints.json").read_text())
    assert len(manifest["quarantined"]) == 12
    for name in manifest["quarantined"]:
        assert f'require_historical_reproduction("{Path(name).name}")' in Path(name).read_text()


def test_binding_counterfactual_changes_answer_with_identical_words():
    from adaptive_mini_gpt.data.concept_binding import bag, pair_cases
    from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
        ByteTextTokenizerM103,
    )

    rows = pair_cases("tiny", ((False, False), (False, True)), 0, 0, ("spark",), 2, 0)
    assert rows[0].state == 0 and rows[1].state == 1
    assert bag(rows[0].question) == bag(rows[1].question)
    assert rows[0].question != rows[1].question
    tok = ByteTextTokenizerM103()
    for row in rows:
        ids = tok.encode_prompt(row.question, row.rules, max_length=256)
        assert tok.decode(ids[ids.index(tok.QUESTION_ID) + 1 : -1]) == row.question
    negated = pair_cases("tiny", ((False, False), (True, True)), 0, 0, ("fire", "spark"), 0, 0)
    assert negated[0].state == negated[1].state == 0
