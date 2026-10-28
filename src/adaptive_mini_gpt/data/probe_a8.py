"""Coverage-matched diagnostic data selection and verification for Stage A.8.

Pure index/string operations; zero neural inference.
"""

from __future__ import annotations

from typing import Any


def analyze_row_factors(row: Any) -> tuple[tuple[str, ...], str, str, int, str]:
    """Extract (statuses, negator, order, frame, prefix) for exact factor balancing."""
    content = getattr(row, "content", "")
    statuses = tuple(getattr(row, "statuses", ()))
    frame = getattr(row, "frame", 0)
    prefix = getattr(row, "prefix", "")

    has_not = "not" in content
    has_without = "without" in content
    negator = "not" if has_not else ("without" if has_without else "none")

    if "flammable" in content and "conductive" in content:
        order = (
            "flam_first"
            if content.index("flammable") < content.index("conductive")
            else "cond_first"
        )
    elif "flammable" in content:
        order = "only_flam"
    elif "conductive" in content:
        order = "only_cond"
    else:
        order = "none"

    return statuses, negator, order, frame, prefix


def coverage_matched_probe_indices(
    rows: list[Any], max_coverage: int = 756
) -> tuple[list[int], list[int]]:
    """Select 378 balanced Rax rows crossing all factors, and 378 Vim rows for familiar-val.

    Args:
        rows: All training rows (where the first max_coverage rows are coverage descriptions).
        max_coverage: Number of non-rehearsal coverage descriptions (756).

    Returns:
        probe_train_indices: 378 indices of Rax coverage descriptions.
        familiar_val_indices: 378 indices of Vim coverage descriptions.
    """
    coverage_rows = rows[:max_coverage]
    rax_indices = [i for i, r in enumerate(coverage_rows) if getattr(r, "name", "") == "Rax"]
    vim_indices = [i for i, r in enumerate(coverage_rows) if getattr(r, "name", "") == "Vim"]

    if len(rax_indices) != 378 or len(vim_indices) != 378:
        raise ValueError(
            f"Expected exactly 378 Rax and 378 Vim rows, got {len(rax_indices)} Rax and {len(vim_indices)} Vim"
        )

    # Verify that all 378 factor cells are present in rax_indices
    cells = {analyze_row_factors(coverage_rows[i]) for i in rax_indices}
    if len(cells) != 378:
        raise ValueError(f"Incomplete factor coverage in probe training: {len(cells)}/378 cells")

    # Verify both negators are represented
    negators = {cell[1] for cell in cells}
    if not {"not", "without"}.issubset(negators):
        raise ValueError(f"Missing negator operators in probe training: {negators}")

    # Verify both orders are represented
    orders = {cell[2] for cell in cells}
    if not {"flam_first", "cond_first"}.issubset(orders):
        raise ValueError(f"Missing property presentation orders in probe training: {orders}")

    # Verify all 6 frames
    frames = {cell[3] for cell in cells}
    if frames != set(range(6)):
        raise ValueError(f"Missing frames in probe training: {frames}")

    # Verify all 3 prefixes
    prefixes = {cell[4] for cell in cells}
    if prefixes != {"", "Log. ", "An entry. "}:
        raise ValueError(f"Missing prefixes in probe training: {prefixes}")

    return rax_indices, vim_indices


def extract_question_token_bounds(
    tokens: list[int],
    question_text: str,
    q_id: int = 261,
    a_id: int = 262,
) -> tuple[int, int]:
    """Finds exact [q_pos, a_pos] bounds for question tokens."""
    try:
        q_pos = tokens.index(q_id)
        a_pos = tokens.index(a_id)
        if a_pos > q_pos:
            return q_pos, a_pos
        return q_pos, len(tokens)
    except ValueError:
        return 0, len(tokens)
