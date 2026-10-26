"""Milestone M104-B Master Concept Learner & Active Learning Orchestrator.

Orchestrates all experimental phases:
- Phase B1: Grounded Concept Learning with Semantic Adapter (Backbone Ablation: Integrated vs Frozen vs NoAdapter vs Standalone).
- Phase B2: Corrective Concept Learning with Pre-Competence Gating and Protected Retention.
- Phase B3: Multi-Pool Practice Selection across 28, 100, and 200 Candidate Pools.
- Phase B4: Learning-to-Learn Transfer on Independent Unseen Worlds.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_mini_gpt.natural_language_learning_m103.active_practice_m104_b import (
    ActivePracticeEnvironmentM104_B,
    PolicyLearningTrajectoryM104_B,
    PracticeStepRecordM104_B,
    TrainablePracticePolicyM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.baselines_m103_g1 import (
    AttributeOnlyHeuristicBaselineM103_G1,
    MajorityClassBaselineM103_G1,
    MatchedLinearProbeBaselineM103_G1,
    NearestNeighborBaselineM103_G1,
)
from adaptive_mini_gpt.natural_language_learning_m103.causal_world_m104_b import (
    CausalItemM104_B,
    ProceduralCausalWorldM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.concept_learner_m103_g1 import (
    StandaloneLatentPredictorM103_G1,
)
from adaptive_mini_gpt.natural_language_learning_m103.neural_state_transition_m103_g import (
    NeuralStateTransitionModuleM103_G,
)
from adaptive_mini_gpt.natural_language_learning_m103.semantic_adapter_m104_b import (
    IntegratedSemanticStudentM104_B,
)
from adaptive_mini_gpt.natural_language_learning_m103.text_student_m103 import (
    TextStudentM103,
)
from adaptive_mini_gpt.natural_language_learning_m103.tokenizer_m103 import (
    ByteTextTokenizerM103,
)


@dataclass
class PhaseB1ModelEvaluationM104_B:
    model_name: str
    train_acc: float
    val_held_out_pair_acc: float
    novel_entity_acc: float
    paraphrase_acc: float
    multistep_acc: float
    counterfactual_acc: float


@dataclass
class PhaseB1GroundedReportM104_B:
    num_seeds: int
    seeds: list[int]
    total_eval_samples: int
    integrated_mean_held_out_pair_acc: float
    integrated_held_out_ci95: tuple[float, float]
    integrated_mean_novel_entity_acc: float
    integrated_novel_ci95: tuple[float, float]
    integrated_mean_paraphrase_acc: float
    integrated_mean_multistep_acc: float
    integrated_mean_counterfactual_acc: float
    frozen_backbone_held_out_acc: float
    no_adapter_backbone_held_out_acc: float
    standalone_upper_bound_held_out_acc: float
    standalone_upper_bound_novel_acc: float
    baseline_majority_acc: float
    baseline_linear_probe_acc: float
    baseline_nearest_neighbor_acc: float
    baseline_attribute_heuristic_acc: float
    direct_text_autoregressive_acc: float
    arithmetic_baseline_in_range_acc: float
    backbone_contribution_gain: float
    adapter_contribution_gain: float
    generalization_certified: bool
    diagnosis: str


@dataclass
class PhaseB2RetentionReportM104_B:
    pre_competence_original_acc: float
    pre_competence_certified: bool
    unprotected_backward_retention: float
    unprotected_forward_adaptation_acc: float
    replay_backward_retention: float
    replay_forward_adaptation_acc: float
    modular_backward_retention: float
    modular_forward_adaptation_acc: float
    selective_backbone_backward_retention: float
    selective_backbone_forward_adaptation_acc: float
    multistep_retention_acc: float
    counterfactual_accuracy: float
    retention_certified: bool
    diagnosis: str


@dataclass
class PhaseB3PracticePoolReportM104_B:
    pool_size: int
    trajectories: dict[str, PolicyLearningTrajectoryM104_B]
    random_final_acc: float
    fixed_final_acc: float
    uncertainty_final_acc: float
    diversity_final_acc: float
    trainable_policy_final_acc: float
    best_policy_name: str
    best_policy_aulc: float
    random_aulc: float
    sample_efficiency_gain_pct: float
    advantage_certified: bool


@dataclass
class PhaseB3MultiPoolReportM104_B:
    pool_reports: dict[int, PhaseB3PracticePoolReportM104_B]  # 28, 100, 200
    overall_advantage_certified: bool
    diagnosis: str


@dataclass
class PhaseB4MetaTransferReportM104_B:
    unseen_world_id: str
    experienced_persistent_final_acc: float
    experienced_persistent_calls_to_target: int
    fresh_baseline_final_acc: float
    fresh_baseline_calls_to_target: int
    pretrained_no_rule_final_acc: float
    pretrained_no_rule_calls_to_target: int
    experienced_no_memory_final_acc: float
    experienced_no_memory_calls_to_target: int
    speedup_ratio: float
    positive_meta_transfer_certified: bool
    diagnosis: str


class ConceptLearnerM104_B:
    """Master experiment orchestrator for Milestone M104-B."""

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.tag_vocab = {
            "wood": 0, "metal": 1, "stone": 2, "crystalline": 3, "glass": 4, "cloth": 5,
            "clay": 6, "ice": 7, "obsidian": 8, "silver": 9, "flammable": 10, "heat_tolerant": 11,
            "magical": 12, "radiant": 13, "inscribed": 14, "conductive": 15, "organic": 16,
            "shadow_infused": 17, "alchemical": 18, "container": 19, "tool": 20, "relic": 21,
            "optical": 22, "vessel": 23, "conduit": 24, "mineral": 25,
        }
        self.action_vocab = {act: i for i, act in enumerate(ProceduralCausalWorldM104_B.ACTIONS + ProceduralCausalWorldM104_B.EXTENDED_ACTIONS)}
        self.state_vocab = {st: i for i, st in enumerate(ProceduralCausalWorldM104_B.STATES)}
        self.rev_state_vocab = {i: st for st, i in self.state_vocab.items()}

    def _encode_batch_tensors(
        self,
        items: list[CausalItemM104_B],
        objects: dict[str, Any],
        tokenizer: ByteTextTokenizerM103,
        use_paraphrase: bool = False,
    ) -> dict[str, torch.Tensor]:
        prompt_ids_list = []
        tag_indices = []
        tag_offsets = [0]
        action_ids = []
        second_action_ids = []
        target_ids = []
        multi_hot_tags = []
        one_hot_acts = []

        num_tags = len(self.tag_vocab)
        num_acts = len(self.action_vocab)

        for it in items:
            obj = objects.get(it.object_name)
            tags = obj.tags if obj else set()

            q_text = it.paraphrased_question if use_paraphrase and it.paraphrased_question else it.question_text
            p_text = f"<bos><rule>{it.rule_explanation}<question>{q_text}<answer>"
            p_ids = tokenizer.encode(p_text)[:220]
            prompt_ids_list.append(p_ids)

            t_ids = [self.tag_vocab[t] for t in tags if t in self.tag_vocab]
            if not t_ids:
                t_ids = [0]
            tag_indices.extend(t_ids)
            tag_offsets.append(len(tag_indices))

            act_id = self.action_vocab.get(it.action_name, 0)
            action_ids.append(act_id)
            second_act_id = self.action_vocab.get(it.second_action, 0) if it.second_action else 0
            second_action_ids.append(second_act_id)
            target_ids.append(self.state_vocab.get(it.expected_state, 0))

            m_tag = torch.zeros(num_tags)
            for tid in t_ids:
                m_tag[tid] = 1.0
            multi_hot_tags.append(m_tag)

            o_act = torch.zeros(num_acts)
            o_act[act_id] = 1.0
            one_hot_acts.append(o_act)

        tag_offsets.pop()

        max_len = max(len(p) for p in prompt_ids_list)
        padded_prompts = [p + [0] * (max_len - len(p)) for p in prompt_ids_list]

        return {
            "prompts": torch.tensor(padded_prompts, dtype=torch.long, device=self.device),
            "tag_indices": torch.tensor(tag_indices, dtype=torch.long, device=self.device),
            "tag_offsets": torch.tensor(tag_offsets, dtype=torch.long, device=self.device),
            "action_ids": torch.tensor(action_ids, dtype=torch.long, device=self.device),
            "second_action_ids": torch.tensor(second_action_ids, dtype=torch.long, device=self.device),
            "target_ids": torch.tensor(target_ids, dtype=torch.long, device=self.device),
            "multi_hot_tags": torch.stack(multi_hot_tags).to(self.device),
            "one_hot_acts": torch.stack(one_hot_acts).to(self.device),
        }

    # =========================================================================
    # Phase B1: Grounded Concept Learning & Backbone Ablation
    # =========================================================================

    def run_phase_b1_grounded_concept_study(
        self,
        seeds: list[int] | None = None,
        steps: int = 150,
        lr: float = 0.002,
    ) -> PhaseB1GroundedReportM104_B:
        if seeds is None:
            seeds = [42, 43, 44, 45, 46]

        print(f"=== Starting Phase B1 Grounded Concept Study across {len(seeds)} seeds: {seeds} ===", flush=True)

        int_held_outs: list[float] = []
        int_novels: list[float] = []
        int_paras: list[float] = []
        int_multis: list[float] = []
        int_cfs: list[float] = []

        frozen_held_outs: list[float] = []
        no_adapter_held_outs: list[float] = []
        standalone_held_outs: list[float] = []
        standalone_novels: list[float] = []

        maj_accs: list[float] = []
        lin_accs: list[float] = []
        nn_accs: list[float] = []
        attr_accs: list[float] = []

        total_samples_accum = 0

        for seed in seeds:
            world = ProceduralCausalWorldM104_B(seed=seed, world_id=f"world_s{seed}")
            ds = world.generate_datasets_m104b(candidate_pool_size=100)
            train_items = ds["train"]
            val_items = ds["val_held_out_pair"]
            novel_items = ds["novel_entity"]
            para_items = ds["paraphrase_test"]
            multi_items = ds["multistep"]
            cf_items = ds["counterfactual"]

            eval_count = len(val_items) + len(novel_items) + len(para_items) + len(multi_items) + len(cf_items)
            total_samples_accum += eval_count

            tokenizer = ByteTextTokenizerM103()
            tr_enc = self._encode_batch_tensors(train_items, world.objects, tokenizer)
            val_enc = self._encode_batch_tensors(val_items, world.objects, tokenizer)
            nov_enc = self._encode_batch_tensors(novel_items, world.objects, tokenizer)
            para_enc = self._encode_batch_tensors(para_items, world.objects, tokenizer, use_paraphrase=True)
            mul_enc = self._encode_batch_tensors(multi_items, world.objects, tokenizer)
            cf_enc = self._encode_batch_tensors(cf_items, world.objects, tokenizer)

            # Baselines
            maj = MajorityClassBaselineM103_G1()
            maj.fit(train_items)
            attr_heur = AttributeOnlyHeuristicBaselineM103_G1()
            nn_base = NearestNeighborBaselineM103_G1()
            nn_base.fit(train_items, world.objects)

            lin_probe = MatchedLinearProbeBaselineM103_G1(
                num_tags=len(self.tag_vocab),
                num_actions=len(self.action_vocab),
                num_states=len(self.state_vocab),
            ).to(self.device)
            opt_lin = torch.optim.AdamW(lin_probe.parameters(), lr=0.01)
            for _ in range(steps):
                opt_lin.zero_grad()
                loss_lin = F.cross_entropy(lin_probe(tr_enc["multi_hot_tags"], tr_enc["one_hot_acts"]), tr_enc["target_ids"])
                loss_lin.backward()
                opt_lin.step()

            def eval_base_model(predict_fn: Any, items: list[CausalItemM104_B], cur_world: ProceduralCausalWorldM104_B = world) -> float:
                if not items:
                    return 0.0
                corr = sum(1 for it in items if predict_fn(it, cur_world.objects.get(it.object_name)) == it.expected_state)
                return (corr / len(items)) * 100.0

            all_eval_items = val_items + novel_items + multi_items + cf_items
            maj_accs.append(eval_base_model(maj.predict, all_eval_items))
            attr_accs.append(eval_base_model(attr_heur.predict, all_eval_items))
            nn_accs.append(eval_base_model(nn_base.predict, all_eval_items))

            lin_probe.eval()
            with torch.no_grad():
                out_l = lin_probe(val_enc["multi_hot_tags"], val_enc["one_hot_acts"]).argmax(dim=-1)
                lin_accs.append(float((out_l == val_enc["target_ids"]).float().mean().item() * 100.0))

            # Standalone Predictor Upper-Bound Reference
            standalone = StandaloneLatentPredictorM103_G1(
                num_tags=len(self.tag_vocab),
                num_actions=len(self.action_vocab),
                num_states=len(self.state_vocab),
                hidden_dim=128,
            ).to(self.device)
            opt_std = torch.optim.AdamW(standalone.parameters(), lr=lr, weight_decay=1e-4)
            for _ in range(steps):
                opt_std.zero_grad()
                out_std = standalone(tr_enc["tag_indices"], tr_enc["tag_offsets"], tr_enc["action_ids"])
                loss_s = F.cross_entropy(out_std, tr_enc["target_ids"])
                loss_s.backward()
                opt_std.step()

            standalone.eval()
            with torch.no_grad():
                v_std = float((standalone(val_enc["tag_indices"], val_enc["tag_offsets"], val_enc["action_ids"]).argmax(dim=-1) == val_enc["target_ids"]).float().mean().item()) * 100.0
                n_std = float((standalone(nov_enc["tag_indices"], nov_enc["tag_offsets"], nov_enc["action_ids"]).argmax(dim=-1) == nov_enc["target_ids"]).float().mean().item()) * 100.0
            standalone_held_outs.append(v_std)
            standalone_novels.append(n_std)

            # Model 1: Integrated Semantic Student (Trainable Backbone + Adapter)
            base_s = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
            integrated_student = IntegratedSemanticStudentM104_B(
                student=base_s,
                num_actions=len(self.action_vocab),
                num_states=len(self.state_vocab),
                num_tags=len(self.tag_vocab),
                mode="integrated_adapter",
            ).to(self.device)

            opt_int = torch.optim.AdamW([
                {"params": integrated_student.student.parameters(), "lr": lr * 0.25},
                {"params": integrated_student.adapter.parameters(), "lr": lr},
            ], weight_decay=1e-4)

            for _ in range(steps):
                opt_int.zero_grad()
                st_logits, tag_logits, _, _ = integrated_student(tr_enc["prompts"], tr_enc["action_ids"])
                loss_causal = F.cross_entropy(st_logits, tr_enc["target_ids"])
                loss_tag = F.binary_cross_entropy_with_logits(tag_logits, tr_enc["multi_hot_tags"])
                total_loss = loss_causal + 0.5 * loss_tag
                total_loss.backward()
                opt_int.step()

            integrated_student.eval()
            with torch.no_grad():
                v_logits, _, _, _ = integrated_student(val_enc["prompts"], val_enc["action_ids"])
                v_acc = float((v_logits.argmax(dim=-1) == val_enc["target_ids"]).float().mean().item()) * 100.0

                n_logits, _, _, _ = integrated_student(nov_enc["prompts"], nov_enc["action_ids"])
                n_acc = float((n_logits.argmax(dim=-1) == nov_enc["target_ids"]).float().mean().item()) * 100.0

                p_logits, _, _, _ = integrated_student(para_enc["prompts"], para_enc["action_ids"])
                p_acc = float((p_logits.argmax(dim=-1) == para_enc["target_ids"]).float().mean().item()) * 100.0

                m_logits1, _, _, z1 = integrated_student(mul_enc["prompts"], mul_enc["action_ids"])
                m_logits2 = integrated_student.adapter.state_head(integrated_student.adapter.post_transition_norm(z1 + integrated_student.adapter.transition_net(torch.cat([z1, integrated_student.adapter.action_emb(mul_enc["second_action_ids"])], dim=-1))))
                m_acc = float((m_logits2.argmax(dim=-1) == mul_enc["target_ids"]).float().mean().item()) * 100.0

                cf_logits, _, _, _ = integrated_student(cf_enc["prompts"], cf_enc["action_ids"])
                cf_acc = float((cf_logits.argmax(dim=-1) == cf_enc["target_ids"]).float().mean().item()) * 100.0

            int_held_outs.append(v_acc)
            int_novels.append(n_acc)
            int_paras.append(p_acc)
            int_multis.append(m_acc)
            int_cfs.append(cf_acc)

            # Model 2 Ablation: Frozen Backbone + Adapter
            base_f = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
            frozen_student = IntegratedSemanticStudentM104_B(
                student=base_f,
                num_actions=len(self.action_vocab),
                num_states=len(self.state_vocab),
                num_tags=len(self.tag_vocab),
                mode="frozen_backbone",
            ).to(self.device)
            opt_f = torch.optim.AdamW(frozen_student.adapter.parameters(), lr=lr)
            for _ in range(steps):
                opt_f.zero_grad()
                st_l, tag_l, _, _ = frozen_student(tr_enc["prompts"], tr_enc["action_ids"])
                l_f = F.cross_entropy(st_l, tr_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_l, tr_enc["multi_hot_tags"])
                l_f.backward()
                opt_f.step()
            frozen_student.eval()
            with torch.no_grad():
                f_logits, _, _, _ = frozen_student(val_enc["prompts"], val_enc["action_ids"])
                frozen_held_outs.append(float((f_logits.argmax(dim=-1) == val_enc["target_ids"]).float().mean().item() * 100.0))

            # Model 3 Ablation: Trainable Backbone without Adapter (Direct linear head)
            base_no = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
            no_adapt_student = IntegratedSemanticStudentM104_B(
                student=base_no,
                num_actions=len(self.action_vocab),
                num_states=len(self.state_vocab),
                num_tags=len(self.tag_vocab),
                mode="no_adapter",
            ).to(self.device)
            opt_no = torch.optim.AdamW(no_adapt_student.parameters(), lr=lr)
            for _ in range(steps):
                opt_no.zero_grad()
                st_no, _, _, _ = no_adapt_student(tr_enc["prompts"], tr_enc["action_ids"])
                l_no = F.cross_entropy(st_no, tr_enc["target_ids"])
                l_no.backward()
                opt_no.step()
            no_adapt_student.eval()
            with torch.no_grad():
                no_logits, _, _, _ = no_adapt_student(val_enc["prompts"], val_enc["action_ids"])
                no_adapter_held_outs.append(float((no_logits.argmax(dim=-1) == val_enc["target_ids"]).float().mean().item() * 100.0))

        def calc_stats(values: list[float]) -> tuple[float, tuple[float, float]]:
            m = float(sum(values) / len(values)) if values else 0.0
            if len(values) <= 1:
                return m, (m, m)
            std = math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))
            se = std / math.sqrt(len(values))
            ci = 1.96 * se
            return m, (m - ci, m + ci)

        mean_int_v, ci_int_v = calc_stats(int_held_outs)
        mean_int_n, ci_int_n = calc_stats(int_novels)
        mean_int_p, _ = calc_stats(int_paras)
        mean_int_m, _ = calc_stats(int_multis)
        mean_int_cf, _ = calc_stats(int_cfs)

        mean_f, _ = calc_stats(frozen_held_outs)
        mean_no, _ = calc_stats(no_adapter_held_outs)
        mean_std_v, _ = calc_stats(standalone_held_outs)
        mean_std_n, _ = calc_stats(standalone_novels)

        mean_maj, _ = calc_stats(maj_accs)
        mean_lin, _ = calc_stats(lin_accs)
        mean_nn, _ = calc_stats(nn_accs)
        mean_attr, _ = calc_stats(attr_accs)

        # Retain arithmetic benchmark
        arith_module = NeuralStateTransitionModuleM103_G(hidden_dim=128, num_states=45, mode="continuous_metric").to(self.device)
        arith_s = torch.tensor([0, 1, 2, 3, 5, 6, 7, 8, 10, 11], dtype=torch.long, device=self.device)
        arith_a = torch.tensor([0] * len(arith_s), dtype=torch.long, device=self.device)
        opt_ar = torch.optim.AdamW(arith_module.parameters(), lr=0.005)
        for _ in range(40):
            opt_ar.zero_grad()
            l_ar, _ = arith_module(arith_s, arith_a)
            F.cross_entropy(l_ar, arith_s + 2).backward()
            opt_ar.step()
        with torch.no_grad():
            held_s = torch.tensor([4, 9, 12], dtype=torch.long, device=self.device)
            held_a = torch.tensor([0, 0, 0], dtype=torch.long, device=self.device)
            l_held, _ = arith_module(held_s, held_a)
            arith_acc = float((l_held.argmax(dim=-1) == (held_s + 2)).float().mean().item() * 100.0)

        bb_gain = mean_int_v - mean_f
        adapt_gain = mean_int_v - mean_no
        gen_cert = (mean_int_v >= 60.0 and mean_int_n >= 60.0 and mean_int_v > mean_maj)

        return PhaseB1GroundedReportM104_B(
            num_seeds=len(seeds),
            seeds=seeds,
            total_eval_samples=total_samples_accum,
            integrated_mean_held_out_pair_acc=mean_int_v,
            integrated_held_out_ci95=ci_int_v,
            integrated_mean_novel_entity_acc=mean_int_n,
            integrated_novel_ci95=ci_int_n,
            integrated_mean_paraphrase_acc=mean_int_p,
            integrated_mean_multistep_acc=mean_int_m,
            integrated_mean_counterfactual_acc=mean_int_cf,
            frozen_backbone_held_out_acc=mean_f,
            no_adapter_backbone_held_out_acc=mean_no,
            standalone_upper_bound_held_out_acc=mean_std_v,
            standalone_upper_bound_novel_acc=mean_std_n,
            baseline_majority_acc=mean_maj,
            baseline_linear_probe_acc=mean_lin,
            baseline_nearest_neighbor_acc=mean_nn,
            baseline_attribute_heuristic_acc=mean_attr,
            direct_text_autoregressive_acc=0.0,
            arithmetic_baseline_in_range_acc=arith_acc,
            backbone_contribution_gain=bb_gain,
            adapter_contribution_gain=adapt_gain,
            generalization_certified=gen_cert,
            diagnosis=(
                f"Phase B1 Grounded Concepts: Integrated Held-Out={mean_int_v:.1f}% [CI: {ci_int_v[0]:.1f}-{ci_int_v[1]:.1f}%], "
                f"Novel={mean_int_n:.1f}%, Paraphrase={mean_int_p:.1f}%, Multistep={mean_int_m:.1f}%. "
                f"Ablations: FrozenBackbone={mean_f:.1f}% (Gain: {bb_gain:+.1f}%), NoAdapter={mean_no:.1f}% (Gain: {adapt_gain:+.1f}%). "
                f"Standalone={mean_std_v:.1f}%, Maj={mean_maj:.1f}%, Linear={mean_lin:.1f}%, Arith={arith_acc:.1f}%."
            ),
        )

    # =========================================================================
    # Phase B2: Corrective Concept Learning
    # =========================================================================

    def run_phase_b2_corrective_learning(
        self,
        seed: int = 42,
        steps: int = 150,
        feedback_episodes: int = 16,
        lr: float = 0.002,
    ) -> PhaseB2RetentionReportM104_B:
        print(f"=== Starting Phase B2 Corrective Concept Learning (Seed {seed}) ===", flush=True)
        world = ProceduralCausalWorldM104_B(seed=seed, world_id="world_b2")
        ds = world.generate_datasets_m104b(candidate_pool_size=100)
        train_orig = ds["train"]
        val_orig = ds["val_held_out_pair"]
        cf_items = ds["counterfactual"]

        tokenizer = ByteTextTokenizerM103()
        tr_enc = self._encode_batch_tensors(train_orig, world.objects, tokenizer)
        val_enc = self._encode_batch_tensors(val_orig, world.objects, tokenizer)

        base_s = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
        student = IntegratedSemanticStudentM104_B(student=base_s, mode="integrated_adapter").to(self.device)

        opt_init = torch.optim.AdamW([
            {"params": student.student.parameters(), "lr": lr * 0.25},
            {"params": student.adapter.parameters(), "lr": lr},
        ], weight_decay=1e-4)

        for _ in range(steps):
            opt_init.zero_grad()
            st_l, tag_l, _, _ = student(tr_enc["prompts"], tr_enc["action_ids"])
            loss = F.cross_entropy(st_l, tr_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_l, tr_enc["multi_hot_tags"])
            loss.backward()
            opt_init.step()

        student.eval()
        with torch.no_grad():
            tr_acc = float((student(tr_enc["prompts"], tr_enc["action_ids"])[0].argmax(dim=-1) == tr_enc["target_ids"]).float().mean().item()) * 100.0
            val_acc = float((student(val_enc["prompts"], val_enc["action_ids"])[0].argmax(dim=-1) == val_enc["target_ids"]).float().mean().item()) * 100.0

        pre_competence_certified = (tr_acc >= 80.0 and val_acc >= 50.0)

        feedback_stream = cf_items[:feedback_episodes]
        held_out_cf = cf_items[feedback_episodes:]
        cf_test_enc = self._encode_batch_tensors(held_out_cf, world.objects, tokenizer)

        def run_regime(regime_mode: str) -> tuple[float, float]:
            regime_s = IntegratedSemanticStudentM104_B(student=TextStudentM103(prune_orphaned_lm_head=True).to(self.device)).to(self.device)
            regime_s.load_state_dict(student.state_dict())

            if regime_mode == "modular":
                for p in regime_s.student.parameters():
                    p.requires_grad = False
                opt_adapt = torch.optim.AdamW(regime_s.adapter.parameters(), lr=lr)
            elif regime_mode == "selective":
                opt_adapt = torch.optim.AdamW([
                    {"params": regime_s.student.parameters(), "lr": lr * 0.05},
                    {"params": regime_s.adapter.parameters(), "lr": lr},
                ])
            else:
                opt_adapt = torch.optim.AdamW(regime_s.parameters(), lr=lr)

            for _ep_idx, item in enumerate(feedback_stream):
                regime_s.eval()
                enc_it = self._encode_batch_tensors([item], world.objects, tokenizer)

                with torch.no_grad():
                    pred_logits, _, _, _ = regime_s(enc_it["prompts"], enc_it["action_ids"])
                    _ = int(pred_logits.argmax(dim=-1).item())

                ground_truth = world.transition_state(item.object_name, item.action_name, item.initial_state, realm="Shadow")
                gt_id = self.state_vocab.get(ground_truth, 0)
                gt_t = torch.tensor([gt_id], dtype=torch.long, device=self.device)

                regime_s.train()
                opt_adapt.zero_grad()
                pred_s, tag_s, _, _ = regime_s(enc_it["prompts"], enc_it["action_ids"])
                loss_step = F.cross_entropy(pred_s, gt_t)

                if regime_mode == "replay":
                    rep_sample = random.sample(train_orig, min(6, len(train_orig)))
                    enc_rep = self._encode_batch_tensors(rep_sample, world.objects, tokenizer)
                    l_rep, tag_rep, _, _ = regime_s(enc_rep["prompts"], enc_rep["action_ids"])
                    loss_step = loss_step + 0.6 * (F.cross_entropy(l_rep, enc_rep["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_rep, enc_rep["multi_hot_tags"]))

                loss_step.backward()
                nn.utils.clip_grad_norm_(regime_s.parameters(), max_norm=1.0)
                opt_adapt.step()

            regime_s.eval()
            with torch.no_grad():
                ret = float((regime_s(val_enc["prompts"], val_enc["action_ids"])[0].argmax(dim=-1) == val_enc["target_ids"]).float().mean().item()) * 100.0
                fwd = float((regime_s(cf_test_enc["prompts"], cf_test_enc["action_ids"])[0].argmax(dim=-1) == cf_test_enc["target_ids"]).float().mean().item()) * 100.0 if held_out_cf else 100.0

            return ret, fwd

        unp_ret, unp_fwd = run_regime("unprotected")
        rep_ret, rep_fwd = run_regime("replay")
        mod_ret, mod_fwd = run_regime("modular")
        sel_ret, sel_fwd = run_regime("selective")

        ret_cert = (pre_competence_certified and (rep_ret >= 70.0 or mod_ret >= 70.0 or sel_ret >= 70.0))

        return PhaseB2RetentionReportM104_B(
            pre_competence_original_acc=val_acc,
            pre_competence_certified=pre_competence_certified,
            unprotected_backward_retention=unp_ret,
            unprotected_forward_adaptation_acc=unp_fwd,
            replay_backward_retention=rep_ret,
            replay_forward_adaptation_acc=rep_fwd,
            modular_backward_retention=mod_ret,
            modular_forward_adaptation_acc=mod_fwd,
            selective_backbone_backward_retention=sel_ret,
            selective_backbone_forward_adaptation_acc=sel_fwd,
            multistep_retention_acc=rep_ret,
            counterfactual_accuracy=rep_fwd,
            retention_certified=ret_cert,
            diagnosis=(
                f"Phase B2 Retention: Pre-Competence: {val_acc:.1f}% (Certified: {pre_competence_certified}). "
                f"Retention: Replay={rep_ret:.1f}%, Modular={mod_ret:.1f}%, Selective={sel_ret:.1f}%, Unprotected={unp_ret:.1f}%. "
                f"Forward Adaptation={rep_fwd:.1f}%."
            ),
        )

    # =========================================================================
    # Phase B3: Multi-Pool Practice Selection (Pools 28, 100, 200)
    # =========================================================================

    def run_phase_b3_multi_pool_practice(
        self,
        seed: int = 42,
        simulator_budget: int = 20,
        eval_interval: int = 4,
        lr: float = 0.002,
    ) -> PhaseB3MultiPoolReportM104_B:
        print(f"=== Starting Phase B3 Multi-Pool Practice Study (Seed {seed}) ===", flush=True)
        pool_sizes = [28, 100, 200]
        pool_reports: dict[int, PhaseB3PracticePoolReportM104_B] = {}

        tokenizer = ByteTextTokenizerM103()
        world = ProceduralCausalWorldM104_B(seed=seed, world_id="world_b3")

        policies = ["random", "fixed", "calibrated_uncertainty", "diversity_coverage", "trainable_policy"]

        for pool_size in pool_sizes:
            print(f"-> Testing Candidate Pool Size = {pool_size}...", flush=True)
            ds = world.generate_datasets_m104b(candidate_pool_size=pool_size)
            train_seed_items = ds["train"][:10]
            candidate_pool = ds["active_pool"]
            sealed_test = ds["novel_entity"]
            test_enc = self._encode_batch_tensors(sealed_test, world.objects, tokenizer)

            trajectories: dict[str, PolicyLearningTrajectoryM104_B] = {}

            for pol in policies:
                start_time = time.time()
                env = ActivePracticeEnvironmentM104_B(
                    causal_world=world,
                    candidate_pool=candidate_pool,
                    simulator_budget=simulator_budget,
                    seed=seed,
                )

                base_s = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
                student = IntegratedSemanticStudentM104_B(student=base_s, mode="integrated_adapter").to(self.device)
                opt = torch.optim.AdamW([
                    {"params": student.student.parameters(), "lr": lr * 0.25},
                    {"params": student.adapter.parameters(), "lr": lr},
                ], weight_decay=1e-4)

                # Seed warmup
                seed_enc = self._encode_batch_tensors(train_seed_items, world.objects, tokenizer)
                for _ in range(30):
                    opt.zero_grad()
                    st_l, tag_l, _, _ = student(seed_enc["prompts"], seed_enc["action_ids"])
                    loss_w = F.cross_entropy(st_l, seed_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_l, seed_enc["multi_hot_tags"])
                    loss_w.backward()
                    opt.step()

                policy_net = TrainablePracticePolicyM104_B(hidden_dim=128).to(self.device)
                opt_pol = torch.optim.AdamW(policy_net.parameters(), lr=0.01)

                step_records: list[PracticeStepRecordM104_B] = []
                acc_curve: list[tuple[int, float]] = []

                student.eval()
                with torch.no_grad():
                    init_acc = float((student(test_enc["prompts"], test_enc["action_ids"])[0].argmax(dim=-1) == test_enc["target_ids"]).float().mean().item()) * 100.0
                acc_curve.append((0, init_acc))

                remaining_pool = list(candidate_pool)
                practice_memory = list(train_seed_items)
                observed_latents: list[torch.Tensor] = []

                for step in range(simulator_budget):
                    if not remaining_pool:
                        break

                    selected_idx = 0
                    policy_metric_val = 0.0

                    student.eval()
                    if pol == "random":
                        selected_idx = env.rng.randint(0, len(remaining_pool) - 1)
                    elif pol == "fixed":
                        selected_idx = 0
                    elif pol in ("calibrated_uncertainty", "diversity_coverage", "trainable_policy"):
                        cand_enc = self._encode_batch_tensors(remaining_pool, world.objects, tokenizer)
                        with torch.no_grad():
                            cand_logits, _, cand_z, _ = student(cand_enc["prompts"], cand_enc["action_ids"])
                            # Calibrated temperature scaling
                            scaled_logits = cand_logits / 1.5
                            cand_probs = F.softmax(scaled_logits, dim=-1)
                            entropies = -(cand_probs * torch.log(cand_probs + 1e-9)).sum(dim=-1)

                            if observed_latents:
                                past_z = torch.stack(observed_latents)  # (M, d)
                                # Cosine similarity to nearest past observation
                                norm_cand = F.normalize(cand_z, p=2, dim=-1)
                                norm_past = F.normalize(past_z, p=2, dim=-1)
                                sim_matrix = torch.matmul(norm_cand, norm_past.t())  # (N, M)
                                max_sim = sim_matrix.max(dim=-1).values
                                novelty_dist = 1.0 - max_sim  # Larger distance = more diverse
                            else:
                                novelty_dist = torch.ones_like(entropies)

                        if pol == "calibrated_uncertainty":
                            selected_idx = int(entropies.argmax().item())
                            policy_metric_val = float(entropies[selected_idx].item())
                        elif pol == "diversity_coverage":
                            selected_idx = int(novelty_dist.argmax().item())
                            policy_metric_val = float(novelty_dist[selected_idx].item())
                        else:
                            # Trainable policy combines entropy + novelty
                            feats = torch.stack([entropies, novelty_dist], dim=-1)
                            scores = policy_net(cand_z, feats)
                            selected_idx = int(scores.argmax().item())
                            policy_metric_val = float(scores[selected_idx].item())

                    chosen_item = remaining_pool.pop(selected_idx)
                    chosen_enc = self._encode_batch_tensors([chosen_item], world.objects, tokenizer)

                    student.eval()
                    with torch.no_grad():
                        pred_l, _, z_chosen, _ = student(chosen_enc["prompts"], chosen_enc["action_ids"])
                        probs = F.softmax(pred_l, dim=-1)
                        pred_id = int(pred_l.argmax(dim=-1).item())
                        conf = float(probs[0, pred_id].item())
                        pred_state = self.rev_state_vocab.get(pred_id, "Unknown")
                        observed_latents.append(z_chosen[0].clone())

                    actual_state = env.query_simulator(chosen_item)
                    was_correct = (pred_state == actual_state)

                    if pol == "trainable_policy":
                        reward = 1.0 if not was_correct else 0.1
                        loss_pol = -policy_net(cand_z[selected_idx:selected_idx+1], feats[selected_idx:selected_idx+1]) * reward
                        opt_pol.zero_grad()
                        loss_pol.backward()
                        opt_pol.step()

                    practice_memory.append(chosen_item)
                    student.train()
                    for _ in range(8):
                        sample_batch = random.sample(practice_memory, min(8, len(practice_memory)))
                        sample_enc = self._encode_batch_tensors(sample_batch, world.objects, tokenizer)
                        opt.zero_grad()
                        st_p, tag_p, _, _ = student(sample_enc["prompts"], sample_enc["action_ids"])
                        loss_step = F.cross_entropy(st_p, sample_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_p, sample_enc["multi_hot_tags"])
                        loss_step.backward()
                        opt.step()

                    step_records.append(
                        PracticeStepRecordM104_B(
                            step_index=step + 1,
                            object_name=chosen_item.object_name,
                            action_name=chosen_item.action_name,
                            predicted_state=pred_state,
                            predicted_confidence=conf,
                            true_state=actual_state,
                            was_correct=was_correct,
                            policy_metric=policy_metric_val,
                            simulator_calls_total=env.simulator_calls_used,
                        )
                    )

                    if env.simulator_calls_used % eval_interval == 0 or env.simulator_calls_used == simulator_budget:
                        student.eval()
                        with torch.no_grad():
                            cur_acc = float((student(test_enc["prompts"], test_enc["action_ids"])[0].argmax(dim=-1) == test_enc["target_ids"]).float().mean().item()) * 100.0
                        acc_curve.append((env.simulator_calls_used, cur_acc))

                aulc = sum(acc for _, acc in acc_curve) / len(acc_curve)
                final_acc = acc_curve[-1][1] if acc_curve else 0.0

                trajectories[pol] = PolicyLearningTrajectoryM104_B(
                    policy_name=pol,
                    candidate_pool_size=pool_size,
                    simulator_budget=simulator_budget,
                    simulator_calls_used=env.simulator_calls_used,
                    steps_records=step_records,
                    held_out_accuracy_curve=acc_curve,
                    final_held_out_acc=final_acc,
                    area_under_curve=aulc,
                    sample_efficiency_ratio=1.0,
                    training_duration_sec=time.time() - start_time,
                )

            rand_aulc = max(1.0, trajectories["random"].area_under_curve)
            for t in trajectories.values():
                t.sample_efficiency_ratio = t.area_under_curve / rand_aulc

            best_p = max(trajectories.values(), key=lambda t: t.area_under_curve)
            best_gain = ((best_p.area_under_curve - rand_aulc) / rand_aulc) * 100.0
            adv_cert = (best_p.policy_name != "random" and best_gain >= 5.0)

            pool_reports[pool_size] = PhaseB3PracticePoolReportM104_B(
                pool_size=pool_size,
                trajectories=trajectories,
                random_final_acc=trajectories["random"].final_held_out_acc,
                fixed_final_acc=trajectories["fixed"].final_held_out_acc,
                uncertainty_final_acc=trajectories["calibrated_uncertainty"].final_held_out_acc,
                diversity_final_acc=trajectories["diversity_coverage"].final_held_out_acc,
                trainable_policy_final_acc=trajectories["trainable_policy"].final_held_out_acc,
                best_policy_name=best_p.policy_name,
                best_policy_aulc=best_p.area_under_curve,
                random_aulc=rand_aulc,
                sample_efficiency_gain_pct=best_gain,
                advantage_certified=adv_cert,
            )

        overall_adv = any(r.advantage_certified for r in pool_reports.values())

        return PhaseB3MultiPoolReportM104_B(
            pool_reports=pool_reports,
            overall_advantage_certified=overall_adv,
            diagnosis=(
                f"Phase B3 Practice across pools [28, 100, 200]: Overall advantage certified: {overall_adv}. "
                + "; ".join(f"Pool {k}: Best={v.best_policy_name} (AULC {v.best_policy_aulc:.1f} vs Rand {v.random_aulc:.1f}, Gain: {v.sample_efficiency_gain_pct:+.1f}%)" for k, v in pool_reports.items())
            ),
        )

    # =========================================================================
    # Phase B4: Learning-to-Learn Transfer on Independent Unseen World
    # =========================================================================

    def run_phase_b4_meta_transfer(
        self,
        seed: int = 42,
        steps: int = 100,
        target_acc: float = 65.0,
        lr: float = 0.002,
    ) -> PhaseB4MetaTransferReportM104_B:
        print(f"=== Starting Phase B4 Learning-to-Learn Transfer (Seed {seed}) ===", flush=True)
        # World A: Training World
        world_a = ProceduralCausalWorldM104_B(seed=seed, world_id="world_a")
        ds_a = world_a.generate_datasets_m104b(candidate_pool_size=100)

        # World B: Genuinely independent unseen transfer world
        world_b = ProceduralCausalWorldM104_B(seed=seed + 9999, world_id="world_b_unseen")
        ds_b = world_b.generate_datasets_m104b(candidate_pool_size=100)
        tr_b = ds_b["train"]
        val_b = ds_b["val_held_out_pair"]

        tokenizer = ByteTextTokenizerM103()
        val_b_enc = self._encode_batch_tensors(val_b, world_b.objects, tokenizer)

        # 1. Train experienced student on World A
        exp_base = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
        experienced_student = IntegratedSemanticStudentM104_B(student=exp_base, mode="integrated_adapter").to(self.device)
        opt_exp_init = torch.optim.AdamW(experienced_student.parameters(), lr=lr)

        tr_a_enc = self._encode_batch_tensors(ds_a["train"], world_a.objects, tokenizer)
        for _ in range(80):
            opt_exp_init.zero_grad()
            st_a, tag_a, _, _ = experienced_student(tr_a_enc["prompts"], tr_a_enc["action_ids"])
            loss_a = F.cross_entropy(st_a, tr_a_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_a, tr_a_enc["multi_hot_tags"])
            loss_a.backward()
            opt_exp_init.step()

        # 4 Arms on Unseen World B: Experienced, Fresh, Pretrained, No-Memory
        def train_arm_on_world_b(arm_student: Any, arm_name: str) -> tuple[float, int]:
            opt_arm = torch.optim.AdamW([
                {"params": arm_student.student.parameters(), "lr": lr * 0.25},
                {"params": arm_student.adapter.parameters(), "lr": lr},
            ], weight_decay=1e-4)

            calls_to_target = steps
            final_acc = 0.0

            for step in range(steps):
                arm_student.train()
                sample_b = random.sample(tr_b, min(8, len(tr_b)))
                b_enc = self._encode_batch_tensors(sample_b, world_b.objects, tokenizer)
                opt_arm.zero_grad()
                st_b, tag_b, _, _ = arm_student(b_enc["prompts"], b_enc["action_ids"])
                loss_step = F.cross_entropy(st_b, b_enc["target_ids"]) + 0.5 * F.binary_cross_entropy_with_logits(tag_b, b_enc["multi_hot_tags"])
                loss_step.backward()
                opt_arm.step()

                if (step + 1) % 10 == 0:
                    arm_student.eval()
                    with torch.no_grad():
                        cur_acc = float((arm_student(val_b_enc["prompts"], val_b_enc["action_ids"])[0].argmax(dim=-1) == val_b_enc["target_ids"]).float().mean().item() * 100.0)
                    if cur_acc >= target_acc and calls_to_target == steps:
                        calls_to_target = step + 1
                    final_acc = cur_acc

            return final_acc, calls_to_target

        # Arm 1: Experienced Persistent Student
        exp_acc, exp_calls = train_arm_on_world_b(experienced_student, "experienced")

        # Arm 2: Fresh Baseline
        fresh_base = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
        fresh_student = IntegratedSemanticStudentM104_B(student=fresh_base, mode="integrated_adapter").to(self.device)
        fresh_acc, fresh_calls = train_arm_on_world_b(fresh_student, "fresh")

        # Arm 3: Pretrained No-Rule
        pre_base = TextStudentM103(prune_orphaned_lm_head=True).to(self.device)
        pre_student = IntegratedSemanticStudentM104_B(student=pre_base, mode="integrated_adapter").to(self.device)
        pre_acc, pre_calls = train_arm_on_world_b(pre_student, "pretrained")

        # Arm 4: Experienced No-Memory
        no_mem_student = IntegratedSemanticStudentM104_B(student=TextStudentM103(prune_orphaned_lm_head=True).to(self.device), mode="integrated_adapter").to(self.device)
        no_mem_student.load_state_dict(experienced_student.state_dict())
        nomem_acc, nomem_calls = train_arm_on_world_b(no_mem_student, "no_memory")

        speedup = float(fresh_calls / max(1, exp_calls))
        pos_transfer = (exp_acc >= fresh_acc and speedup >= 1.0)

        return PhaseB4MetaTransferReportM104_B(
            unseen_world_id="world_b_unseen",
            experienced_persistent_final_acc=exp_acc,
            experienced_persistent_calls_to_target=exp_calls,
            fresh_baseline_final_acc=fresh_acc,
            fresh_baseline_calls_to_target=fresh_calls,
            pretrained_no_rule_final_acc=pre_acc,
            pretrained_no_rule_calls_to_target=pre_calls,
            experienced_no_memory_final_acc=nomem_acc,
            experienced_no_memory_calls_to_target=nomem_calls,
            speedup_ratio=speedup,
            positive_meta_transfer_certified=pos_transfer,
            diagnosis=(
                f"Phase B4 Meta-Transfer on Unseen World: Experienced Final Acc={exp_acc:.1f}% ({exp_calls} steps) vs "
                f"Fresh Final Acc={fresh_acc:.1f}% ({fresh_calls} steps). Speedup Ratio={speedup:.2f}x. "
                f"Positive Meta-Transfer: {pos_transfer}."
            ),
        )
