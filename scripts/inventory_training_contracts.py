"""LOCAL-SAFE AST inventory; no imported project code or neural execution."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path


def inventory(root=Path("src")):
    rows = []
    names = {
        "cross_entropy",
        "binary_cross_entropy_with_logits",
        "nll_loss",
        "CrossEntropyLoss",
        "AdamW",
        "scaled_dot_product_attention",
        "add_param_group",
    }
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in names:
                rows.append(
                    {
                        "path": str(path),
                        "line": node.lineno,
                        "operation": name,
                        "arguments": ast.unparse(node)[:600],
                    }
                )
    return {
        "counts": dict(Counter(r["operation"] for r in rows)),
        "sites": rows,
        "scope": "Source inventory, not automatic semantic qualification",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(inventory(), indent=2))
