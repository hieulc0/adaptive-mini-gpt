"""Milestone M87: Lifelong Knowledge Evaluation Matrix (R[i, j]).

Formally measures continual learning dynamics across long-horizon task streams:
- Evaluation Matrix: R[i, j] = performance on task j after training through episode i.
- Average Accuracy (ACC): Mean performance across all encountered tasks.
- Backward Transfer (BWT): Influence of subsequent learning on past task competencies.
- Forward Transfer (FWT): Zero-shot capability on new tasks before explicit training.
- Average Forgetting (F): Peak past performance minus final performance across all tasks.
- Long-Delayed Revisit Retention: Retention on skills revisited after >= 40 intervening episodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContinualEvaluationMetrics:
    """Formal lifelong continual learning metrics computed from R[i, j]."""

    num_tasks: int
    final_average_accuracy: float
    backward_transfer: float
    forward_transfer: float
    average_forgetting: float
    long_delayed_revisit_retention: float
    per_task_initial_em: dict[str, float] = field(default_factory=dict)
    per_task_final_em: dict[str, float] = field(default_factory=dict)
    per_task_forgetting: dict[str, float] = field(default_factory=dict)


class LifelongEvaluationMatrix:
    """Tracks and computes the lifelong task evaluation matrix R[i, j]."""

    def __init__(self) -> None:
        # Maps (episode_idx, task_id) -> exact_match
        self.r_matrix: dict[int, dict[str, float]] = {}
        # Task introduction episode: task_id -> first_seen_episode_idx
        self.task_intro_episodes: dict[str, int] = {}
        # Zero-shot baseline performance before any training on task
        self.zero_shot_baselines: dict[str, float] = {}

    def record_zero_shot_probe(self, task_id: str, em: float) -> None:
        """Record zero-shot capability on task before training."""
        if task_id not in self.zero_shot_baselines:
            self.zero_shot_baselines[task_id] = float(em)

    def record_episode_evaluation(
        self,
        episode_idx: int,
        task_id: str,
        eval_results: dict[str, float],
    ) -> None:
        """Record evaluation exact match for all active tasks after episode episode_idx.

        eval_results maps task_id -> exact_match on that task's held-out qualification set.
        """
        if task_id not in self.task_intro_episodes:
            self.task_intro_episodes[task_id] = episode_idx

        self.r_matrix[episode_idx] = {k: float(v) for k, v in eval_results.items()}

    def compute_metrics(self) -> ContinualEvaluationMetrics:
        """Compute formal continual learning metrics: ACC, BWT, FWT, F."""
        if not self.r_matrix:
            return ContinualEvaluationMetrics(
                num_tasks=0,
                final_average_accuracy=0.0,
                backward_transfer=0.0,
                forward_transfer=0.0,
                average_forgetting=0.0,
                long_delayed_revisit_retention=1.0,
            )

        episodes = sorted(self.r_matrix.keys())
        final_ep = episodes[-1]
        unique_tasks = list(self.task_intro_episodes.keys())
        t_count = len(unique_tasks)

        if t_count == 0:
            return ContinualEvaluationMetrics(
                num_tasks=0,
                final_average_accuracy=0.0,
                backward_transfer=0.0,
                forward_transfer=0.0,
                average_forgetting=0.0,
                long_delayed_revisit_retention=1.0,
            )

        # 1. Final Average Accuracy (ACC): mean R[T-1, j] across all unique tasks j
        final_evals = self.r_matrix[final_ep]
        per_task_final = {t: final_evals.get(t, 0.0) for t in unique_tasks}
        acc = sum(per_task_final.values()) / max(1, t_count)

        # 2. Backward Transfer (BWT) & Forgetting (F)
        per_task_initial = {}
        per_task_forgetting = {}
        bwt_diffs = []
        forgetting_vals = []
        delayed_retentions = []

        for task_id in unique_tasks:
            intro_ep = self.task_intro_episodes[task_id]
            # Initial qualification EM after the task was first acquired: R[intro_ep, task_id]
            r_initial = self.r_matrix.get(intro_ep, {}).get(task_id, 0.0)
            r_final = per_task_final[task_id]
            per_task_initial[task_id] = r_initial

            # Only compute BWT and forgetting for tasks learned before the final episode
            if intro_ep < final_ep:
                # BWT component: R[final, j] - R[intro, j]
                bwt_diffs.append(r_final - r_initial)

                # Forgetting: max past performance minus final performance
                past_ems = [
                    self.r_matrix[ep].get(task_id, 0.0)
                    for ep in episodes
                    if intro_ep <= ep < final_ep and task_id in self.r_matrix[ep]
                ]
                peak_past = max(past_ems) if past_ems else r_initial
                f_j = max(0.0, peak_past - r_final)
                per_task_forgetting[task_id] = f_j
                forgetting_vals.append(f_j)

                # Long-delayed revisit retention (tasks learned >= 40 episodes before stream end)
                if (final_ep - intro_ep) >= 40:
                    ret_delay = r_final / max(0.01, r_initial)
                    delayed_retentions.append(ret_delay)
            else:
                per_task_forgetting[task_id] = 0.0

        bwt = sum(bwt_diffs) / max(1, len(bwt_diffs)) if bwt_diffs else 0.0
        avg_f = sum(forgetting_vals) / max(1, len(forgetting_vals)) if forgetting_vals else 0.0
        delayed_ret = (
            sum(delayed_retentions) / len(delayed_retentions)
            if delayed_retentions
            else 1.0
        )

        # 3. Forward Transfer (FWT)
        fwt_diffs = []
        for task_id in unique_tasks:
            intro_ep = self.task_intro_episodes[task_id]
            if intro_ep > 0:
                # Performance on task_id just before it was trained: R[intro_ep - 1, task_id]
                prior_ep = intro_ep - 1
                r_prior = self.r_matrix.get(prior_ep, {}).get(
                    task_id, self.zero_shot_baselines.get(task_id, 0.0)
                )
                b_j = self.zero_shot_baselines.get(task_id, 0.0)
                fwt_diffs.append(r_prior - b_j)

        fwt = sum(fwt_diffs) / max(1, len(fwt_diffs)) if fwt_diffs else 0.0

        return ContinualEvaluationMetrics(
            num_tasks=t_count,
            final_average_accuracy=round(acc, 4),
            backward_transfer=round(bwt, 4),
            forward_transfer=round(fwt, 4),
            average_forgetting=round(avg_f, 4),
            long_delayed_revisit_retention=round(delayed_ret, 4),
            per_task_initial_em={k: round(v, 4) for k, v in per_task_initial.items()},
            per_task_final_em={k: round(v, 4) for k, v in per_task_final.items()},
            per_task_forgetting={k: round(v, 4) for k, v in per_task_forgetting.items()},
        )
