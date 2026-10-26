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
