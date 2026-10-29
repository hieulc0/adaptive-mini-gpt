"""Continuous query-conditioned logit and representation sensitivity evaluation.

Measures fine-grained continuous response to query swaps (Case A vs Case B)
sharing identical descriptive prefixes:
1. Continuous Logit Shift: Delta_logit = E[||logits_A - logits_B||_1]
2. Signed Target-Direction Logit Margin: E[(y_A - y_B) * (logits_A - logits_B)]
   capturing sub-threshold movement in the correct direction even when binary predictions
   do not cross the 0.0 threshold.
3. Hidden State Shift: Delta_h = E[||h_last(A) - h_last(B)||_2] and cosine distance.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def compute_continuous_sensitivity(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    h_last_a: torch.Tensor,
    h_last_b: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
) -> dict[str, Any]:
    """Computes fine-grained continuous logit and representation sensitivity metrics.

    Args:
        logits_a: (N, C) logits for case A.
        logits_b: (N, C) logits for case B (with swapped query target).
        h_last_a: (N, D) hidden representation at last question token for case A.
        h_last_b: (N, D) hidden representation at last question token for case B.
        y_a: (N, C) ground-truth binary targets for case A.
        y_b: (N, C) ground-truth binary targets for case B.

    Returns:
        dict containing continuous logit shift, signed target margin, representation shifts,
        and discrete flip rates.
    """
    n_pairs = logits_a.size(0)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "delta_logit_mean": 0.0,
            "delta_h_l2_mean": 0.0,
            "cosine_distance_mean": 0.0,
            "signed_target_margin_mean": 0.0,
            "signed_margin_positive_rate": 0.0,
            "discrete_prediction_flip_rate": 0.0,
            "sub_threshold_sensitivity_count": 0,
        }

    # 1. Continuous Logit Shift: L1 distance per pair, averaged
    delta_logit = (logits_a - logits_b).abs().sum(dim=-1)  # (N,)
    delta_logit_mean = float(delta_logit.mean().item())

    # 2. Signed Target-Direction Logit Margin:
    # delta_y = y_a - y_b in {-1, 0, 1}
    # delta_l = logits_a - logits_b
    # When delta_y != 0, does delta_l have the same sign?
    delta_y = (y_a - y_b).float()  # (N, C)
    delta_l = (logits_a - logits_b).float()  # (N, C)

    active_mask = delta_y != 0  # (N, C)
    if active_mask.any():
        signed_margins = delta_y[active_mask] * delta_l[active_mask]
        signed_target_margin_mean = float(signed_margins.mean().item())
        signed_margin_positive_rate = float((signed_margins > 0).float().mean().item())
    else:
        signed_target_margin_mean = 0.0
        signed_margin_positive_rate = 0.0

    # 3. Hidden State Shift: L2 norm and Cosine Distance
    delta_h = (h_last_a - h_last_b).norm(p=2, dim=-1)  # (N,)
    delta_h_l2_mean = float(delta_h.mean().item())

    cos_sim = F.cosine_similarity(h_last_a, h_last_b, dim=-1)  # (N,)
    cos_dist = 1.0 - cos_sim
    cosine_distance_mean = float(cos_dist.mean().item())

    # 4. Discrete Prediction Flip Rate (standard binary threshold 0.0)
    pred_a = logits_a >= 0.0
    pred_b = logits_b >= 0.0
    discrete_flips = (pred_a != pred_b).any(dim=-1)  # (N,)
    discrete_prediction_flip_rate = float(discrete_flips.float().mean().item())

    # 5. Sub-threshold sensitivity: pairs with delta_logit > 1e-4 but ZERO discrete prediction flips
    sub_threshold_mask = (delta_logit > 1e-4) & (~discrete_flips)
    sub_threshold_sensitivity_count = int(sub_threshold_mask.sum().item())

    return {
        "n_pairs": n_pairs,
        "delta_logit_mean": delta_logit_mean,
        "delta_h_l2_mean": delta_h_l2_mean,
        "cosine_distance_mean": cosine_distance_mean,
        "signed_target_margin_mean": signed_target_margin_mean,
        "signed_margin_positive_rate": signed_margin_positive_rate,
        "discrete_prediction_flip_rate": discrete_prediction_flip_rate,
        "sub_threshold_sensitivity_count": sub_threshold_sensitivity_count,
    }
