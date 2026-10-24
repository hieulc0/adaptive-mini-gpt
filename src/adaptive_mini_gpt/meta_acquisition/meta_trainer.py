"""Milestone M82 Phase 1 & 2: Meta-Policy Training and Episode Execution Engine.

Implements:
1. Extraction of hidden features and diagnostic learning signals without leaking test data.
2. Episodic execution of adaptation tasks under controller supervision.
3. Policy gradient meta-training (REINFORCE with moving-average baseline & entropy regularization).
4. Meta-validation tracking to guard against overfitting.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.learning_to_learn.capacity_adaptation import (
    BottleneckSequenceAdapter,
)
from adaptive_mini_gpt.meta_acquisition.controller import (
    AcquisitionAction,
    AcquisitionState,
    InitChoice,
    LearnedAcquisitionController,
    LRChoice,
    RankChoice,
    StoppingChoice,
)
from adaptive_mini_gpt.meta_acquisition.environment import (
    MetaTaskDataset,
    TaskExample,
)
from adaptive_mini_gpt.model import MiniGPT
from adaptive_mini_gpt.neural_weight_learning.pathway_instrumentation import (
    NeuralPathwayInstrumenter,
)
from adaptive_mini_gpt.neural_weight_learning.procedural_tasks import ProceduralExample

logger = logging.getLogger(__name__)

QUALIFICATION_THRESHOLD: float = 0.85


@dataclass
class AdaptationTrajectoryPoint:
    """Diagnostic state and metrics recorded during an adaptation episode."""

    step: int
    train_loss: float
    qual_em: float
    qual_loss: float
    token_acc: float
    state: AcquisitionState
    action: AcquisitionAction


@dataclass
class EpisodeOutcome:
    """Complete results from a single task adaptation episode."""

    task_id: str
    steps_to_threshold: int
    reached_threshold: bool
    final_step: int
    qualification_em: float
    sealed_test_em: float
    sealed_test_loss: float
    auc_qual_em: float
    selected_rank: int
    allocated_parameters: int
    init_choice: InitChoice
    selected_lr: float
    reward: float
    elapsed_time: float
    adapter_hash: str
    trajectory: list[AdaptationTrajectoryPoint] = field(default_factory=list)


def extract_features(
    model: MiniGPT,
    examples: list[TaskExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract symbol hidden states [B, L, d_model] and target tokens [B, L] from frozen base model."""
    compat_examples = [
        ProceduralExample(
            example_id=ex.example_id,
            rule_family=str(ex.family.value),
            input_tokens=ex.input_tokens,
            target_tokens=ex.target_tokens,
            input_text=ex.input_text,
            target_text=ex.target_text,
        )
        for ex in examples
    ]
    return NeuralPathwayInstrumenter.extract_symbol_hidden_states(model, compat_examples, device)


def compute_diagnostic_state(
    adapter: nn.Module,
    dev_h: torch.Tensor,
    dev_y: torch.Tensor,
    prev_loss: float,
    current_step: int,
    max_steps: int,
    current_rank: int,
    anchor_h: torch.Tensor | None = None,
    anchor_y: torch.Tensor | None = None,
) -> AcquisitionState:
    """Compute non-leaking diagnostic state features from dev probe batch."""
    adapter.eval()
    with torch.no_grad():
        logits = adapter(dev_h)
        loss = float(F.cross_entropy(logits.view(-1, 16), dev_y.view(-1)).item())
        preds = torch.argmax(logits, dim=-1)
        token_acc = float((preds == dev_y).float().mean().item())

        probs = F.softmax(logits, dim=-1)
        # Avoid log(0)
        entropy = float(-(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item())

        replay_loss = 0.0
        if anchor_h is not None and anchor_y is not None:
            a_logits = adapter(anchor_h)
            replay_loss = float(F.cross_entropy(a_logits.view(-1, 16), anchor_y.view(-1)).item())

    loss_slope = (prev_loss - loss) if prev_loss > 0 else 0.0
    grad_norm = min(loss * 0.5, 2.0)  # surrogate early gradient magnitude

    return AcquisitionState(
        probe_loss=loss,
        loss_slope=loss_slope,
        prediction_entropy=entropy,
        grad_norm=grad_norm,
        replay_loss=replay_loss,
        progress_ratio=current_step / max(1, max_steps),
        candidate_capacity_ratio=current_rank / 16.0,
        token_acc=token_acc,
    )


class AdaptationEpisodeEngine:
    """Executes single task adaptation episodes under controller or baseline policies."""

    @classmethod
    def run_episode(
        cls,
        task: MetaTaskDataset,
        train_h: torch.Tensor,
        train_y: torch.Tensor,
        dev_h: torch.Tensor,
        dev_y: torch.Tensor,
        qual_h: torch.Tensor,
        qual_y: torch.Tensor,
        test_h: torch.Tensor,
        test_y: torch.Tensor,
        d_model: int,
        device: torch.device,
        controller: LearnedAcquisitionController | None = None,
        fixed_lr: float | None = None,
        fixed_rank: int | None = None,
        fixed_init: InitChoice | None = None,
        force_random_policy: bool = False,
        experience_priors: dict[int, dict[str, torch.Tensor]] | None = None,
        anchor_h: torch.Tensor | None = None,
        anchor_y: torch.Tensor | None = None,
        max_steps: int = 60,
        eval_interval: int = 5,
        deterministic_actions: bool = False,
    ) -> EpisodeOutcome:
        """Execute one complete adaptation episode."""
        start_time = time.perf_counter()

        # Step 0: Initial probe probe state using a dummy rank-8 probe
        probe_adapter = BottleneckSequenceAdapter(d_model=d_model, bottleneck_dim=8, num_symbols=16).to(device)
        state_0 = compute_diagnostic_state(
            adapter=probe_adapter,
            dev_h=dev_h,
            dev_y=dev_y,
            prev_loss=3.0,
            current_step=0,
            max_steps=max_steps,
            current_rank=8,
            anchor_h=anchor_h,
            anchor_y=anchor_y,
        )

        # Decide initialization, learning rate, and adapter rank
        if force_random_policy:
            import random
            rng = random.Random()
            act_0 = AcquisitionAction(
                init_choice=InitChoice(rng.randint(0, 1)),
                lr_choice=LRChoice(rng.randint(0, 3)),
                rank_choice=RankChoice(rng.randint(0, 2)),
                stopping_choice=StoppingChoice.CONTINUE,
            )
        elif controller is not None:
            act_0 = controller.select_action(state_0, deterministic=deterministic_actions, qual_em_so_far=0.0)
        else:
            # Fixed baseline defaults
            act_0 = AcquisitionAction(
                init_choice=fixed_init or InitChoice.FRESH_RANDOM,
                lr_choice=LRChoice.LR_3E_3,
                rank_choice=RankChoice.RANK_8,
                stopping_choice=StoppingChoice.CONTINUE,
            )

        # Apply overrides if explicitly given
        init_choice = fixed_init if fixed_init is not None else act_0.init_choice
        lr_val = fixed_lr if fixed_lr is not None else act_0.lr_choice.to_float()
        rank_val = fixed_rank if fixed_rank is not None else act_0.rank_choice.to_rank()

        # Instantiate target adapter
        adapter = BottleneckSequenceAdapter(d_model=d_model, bottleneck_dim=rank_val, num_symbols=16).to(device)

        # Apply warm-start if selected and available
        if init_choice == InitChoice.EXPERIENCE_PRIOR and experience_priors is not None:
            if rank_val in experience_priors:
                adapter.load_state_dict(experience_priors[rank_val], strict=False)

        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr_val, weight_decay=1e-4)

        trajectory: list[AdaptationTrajectoryPoint] = []
        steps_to_thresh = max_steps + 10
        reached_thresh = False
        prev_loss = state_0.probe_loss
        final_step = max_steps
        current_qual_em = 0.0

        for step in range(1, max_steps + 1):
            adapter.train()
            optimizer.zero_grad()
            logits = adapter(train_h)
            loss = F.cross_entropy(logits.view(-1, 16), train_y.view(-1))
            loss.backward()
            optimizer.step()

            if step % eval_interval == 0:
                adapter.eval()
                with torch.no_grad():
                    q_logits = adapter(qual_h)
                    q_preds = torch.argmax(q_logits, dim=-1)
                    q_em = float((q_preds == qual_y).all(dim=-1).float().mean().item())
                    q_loss = float(F.cross_entropy(q_logits.view(-1, 16), qual_y.view(-1)).item())
                    q_acc = float((q_preds == qual_y).float().mean().item())

                current_qual_em = q_em
                state_t = compute_diagnostic_state(
                    adapter=adapter,
                    dev_h=dev_h,
                    dev_y=dev_y,
                    prev_loss=prev_loss,
                    current_step=step,
                    max_steps=max_steps,
                    current_rank=rank_val,
                    anchor_h=anchor_h,
                    anchor_y=anchor_y,
                )
                prev_loss = state_t.probe_loss

                if q_em >= QUALIFICATION_THRESHOLD and not reached_thresh:
                    steps_to_thresh = step
                    reached_thresh = True

                # Controller stopping query
                if controller is not None and not force_random_policy:
                    act_t = controller.select_action(
                        state_t,
                        deterministic=deterministic_actions,
                        qual_em_so_far=current_qual_em,
                        admission_threshold=QUALIFICATION_THRESHOLD,
                    )
                elif force_random_policy:
                    import random
                    act_t = AcquisitionAction(
                        init_choice=init_choice,
                        lr_choice=LRChoice.LR_3E_3,
                        rank_choice=RankChoice.RANK_8,
                        stopping_choice=StoppingChoice.STOP_AND_QUALIFY if (q_em >= QUALIFICATION_THRESHOLD and random.random() < 0.5) else StoppingChoice.CONTINUE,
                    )
                else:
                    act_t = AcquisitionAction(
                        init_choice=init_choice,
                        lr_choice=LRChoice.LR_3E_3,
                        rank_choice=RankChoice.RANK_8,
                        stopping_choice=StoppingChoice.STOP_AND_QUALIFY if q_em >= QUALIFICATION_THRESHOLD else StoppingChoice.CONTINUE,
                    )

                trajectory.append(
                    AdaptationTrajectoryPoint(
                        step=step,
                        train_loss=round(float(loss.item()), 4),
                        qual_em=round(q_em, 4),
                        qual_loss=round(q_loss, 4),
                        token_acc=round(q_acc, 4),
                        state=state_t,
                        action=act_t,
                    )
                )

                # Early-stopping if controller triggers stop when threshold is satisfied
                if act_t.stopping_choice == StoppingChoice.STOP_AND_QUALIFY and q_em >= QUALIFICATION_THRESHOLD:
                    final_step = step
                    break

        auc = sum(pt.qual_em for pt in trajectory) / max(1, len(trajectory))

        # Single final evaluation on sealed test set
        adapter.eval()
        with torch.no_grad():
            t_logits = adapter(test_h)
            t_preds = torch.argmax(t_logits, dim=-1)
            test_em = float((t_preds == test_y).all(dim=-1).float().mean().item())
            test_loss = float(F.cross_entropy(t_logits.view(-1, 16), test_y.view(-1)).item())

        elapsed = time.perf_counter() - start_time
        param_count = adapter.count_parameters()

        # Scientific Reward Formulation:
        # High accuracy, low steps, small parameter count, severe penalty for failing admission
        step_cost = final_step if reached_thresh else (max_steps + 10)
        reward = (
            current_qual_em * 100.0
            - 0.5 * step_cost
            - 0.005 * param_count
            - (50.0 if not reached_thresh else 0.0)
        )

        return EpisodeOutcome(
            task_id=task.task_id,
            steps_to_threshold=steps_to_thresh,
            reached_threshold=reached_thresh,
            final_step=final_step,
            qualification_em=round(current_qual_em, 4),
            sealed_test_em=round(test_em, 4),
            sealed_test_loss=round(test_loss, 4),
            auc_qual_em=round(auc, 4),
            selected_rank=rank_val,
            allocated_parameters=param_count,
            init_choice=init_choice,
            selected_lr=lr_val,
            reward=round(reward, 2),
            elapsed_time=round(elapsed, 2),
            adapter_hash=adapter.compute_parameter_hash(),
            trajectory=trajectory,
        )


class MetaPolicyTrainer:
    """Trains LearnedAcquisitionController using policy gradient with moving baseline."""

    @classmethod
    def train_controller(
        cls,
        controller: LearnedAcquisitionController,
        model: MiniGPT,
        meta_train_tasks: list[MetaTaskDataset],
        meta_val_tasks: list[MetaTaskDataset],
        device: torch.device,
        experience_priors: dict[int, dict[str, torch.Tensor]] | None = None,
        anchor_h: torch.Tensor | None = None,
        anchor_y: torch.Tensor | None = None,
        episodes: int = 15,
        meta_lr: float = 3e-3,
        entropy_coef: float = 0.01,
        max_steps: int = 60,
    ) -> dict[str, Any]:
        """Train controller across meta-training task episodes."""
        logger.info("Commencing Meta-Acquisition Policy Training across %d episodes...", episodes)
        optimizer = torch.optim.AdamW(controller.parameters(), lr=meta_lr, weight_decay=1e-4)

        moving_baseline = 0.0
        alpha = 0.8  # baseline smoothing
        training_history: list[dict[str, Any]] = []

        d_model = model.config.d_model

        # Pre-extract task features for efficiency
        task_features: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for t in (meta_train_tasks + meta_val_tasks):
            tr_h, tr_y = extract_features(model, t.train_examples, device)
            dv_h, dv_y = extract_features(model, t.dev_examples, device)
            ql_h, ql_y = extract_features(model, t.qualification_examples, device)
            ts_h, ts_y = extract_features(model, t.sealed_test_examples, device)
            task_features[t.task_id] = (tr_h, tr_y, dv_h, dv_y, ql_h, ql_y, ts_h, ts_y)

        best_val_em = 0.0
        best_state_dict = copy.deepcopy(controller.state_dict())

        for ep in range(episodes):
            task = meta_train_tasks[ep % len(meta_train_tasks)]
            tr_h, tr_y, dv_h, dv_y, ql_h, ql_y, ts_h, ts_y = task_features[task.task_id]

            controller.train()
            outcome = AdaptationEpisodeEngine.run_episode(
                task=task,
                train_h=tr_h,
                train_y=tr_y,
                dev_h=dv_h,
                dev_y=dv_y,
                qual_h=ql_h,
                qual_y=ql_y,
                test_h=ts_h,
                test_y=ts_y,
                d_model=d_model,
                device=device,
                controller=controller,
                experience_priors=experience_priors,
                anchor_h=anchor_h,
                anchor_y=anchor_y,
                max_steps=max_steps,
                deterministic_actions=False,
            )

            # Collect episode transitions
            states = [pt.state for pt in outcome.trajectory]
            actions = [pt.action for pt in outcome.trajectory]

            if states and actions:
                log_probs, entropy = controller.evaluate_episode(states, actions)
                advantage = outcome.reward - moving_baseline

                # REINFORCE loss: - log_prob * advantage - beta * entropy
                policy_loss = -(log_probs * advantage).mean() - entropy_coef * entropy.mean()

                optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(controller.parameters(), max_norm=1.0)
                optimizer.step()

                moving_baseline = alpha * moving_baseline + (1 - alpha) * outcome.reward

            # Evaluate on meta-validation set every 5 episodes
            val_em_avg = 0.0
            if (ep + 1) % 5 == 0 or ep == episodes - 1:
                controller.eval()
                val_ems = []
                for v_task in meta_val_tasks:
                    v_tr_h, v_tr_y, v_dv_h, v_dv_y, v_ql_h, v_ql_y, v_ts_h, v_ts_y = task_features[v_task.task_id]
                    v_outcome = AdaptationEpisodeEngine.run_episode(
                        task=v_task,
                        train_h=v_tr_h,
                        train_y=v_tr_y,
                        dev_h=v_dv_h,
                        dev_y=v_dv_y,
                        qual_h=v_ql_h,
                        qual_y=v_ql_y,
                        test_h=v_ts_h,
                        test_y=v_ts_y,
                        d_model=d_model,
                        device=device,
                        controller=controller,
                        experience_priors=experience_priors,
                        anchor_h=anchor_h,
                        anchor_y=anchor_y,
                        max_steps=max_steps,
                        deterministic_actions=True,
                    )
                    val_ems.append(v_outcome.qualification_em)
                val_em_avg = sum(val_ems) / max(1, len(val_ems))
                if val_em_avg > best_val_em:
                    best_val_em = val_em_avg
                    best_state_dict = copy.deepcopy(controller.state_dict())

            training_history.append(
                {
                    "episode": ep + 1,
                    "task_id": task.task_id,
                    "reward": outcome.reward,
                    "steps_to_thresh": outcome.steps_to_threshold,
                    "qual_em": outcome.qualification_em,
                    "selected_rank": outcome.selected_rank,
                    "selected_lr": outcome.selected_lr,
                    "val_em_avg": round(val_em_avg, 4),
                }
            )

        controller.load_state_dict(best_state_dict)
        logger.info("Meta-Training complete. Best Validation EM: %.4f", best_val_em)

        return {
            "episodes_trained": episodes,
            "best_val_em": round(best_val_em, 4),
            "final_moving_baseline": round(moving_baseline, 2),
            "history": training_history,
        }
