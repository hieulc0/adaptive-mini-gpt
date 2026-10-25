"""Milestone M94 Phase 3: Program Curriculum & Generalization Benchmarks.

Generates controlled program tasks across the 8 proposed learning stages:
Stage 1: Variables and simple assignments (unseen values and variable names).
Stage 2: Arithmetic expressions (new operand combinations).
Stage 3: Conditions and branches (unseen comparisons and branches).
Stage 4: Bounded loops (unseen iteration counts and longer loops).
Stage 5: Stateful computation (accumulators and counters).
Stage 6: Nested operations (loops containing conditions).
Stage 7: Program execution traces (intermediate state transitions).
Stage 8: Novel unfamiliar algorithms composed from learned concepts.
Additional: Unsupported/invalid programs requiring epistemic abstention.

Enforces strictly disjoint training, validation, and held-out evaluation splits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from adaptive_mini_gpt.neural_program_learning.mini_language import (
    AssignStmt,
    BinOpExpr,
    CompareExpr,
    ExecutionResult,
    ForStmt,
    IfStmt,
    LiteralExpr,
    PrintStmt,
    Program,
    ProgramTokenizer,
    ReferenceInterpreter,
    VarExpr,
)


@dataclass(frozen=True)
class ProgramExample:
    """A single program instance with ground-truth reference execution."""

    program_id: str
    stage_id: int
    category: str
    program: Program
    code_text: str
    token_ids: list[int]
    reference_result: ExecutionResult
    is_supported: bool
    requires_abstention: bool


@dataclass(frozen=True)
class ProgramStageSuite:
    """Dataset split for a specific learning stage."""

    stage_id: int
    stage_name: str
    train_examples: list[ProgramExample]
    dev_examples: list[ProgramExample]
    test_examples: list[ProgramExample]
    held_out_examples: list[ProgramExample]


@dataclass(frozen=True)
class M94ProgramCurriculum:
    """Complete multi-stage benchmark suite for Milestone M94."""

    stages: dict[int, ProgramStageSuite]
    unsupported_examples: list[ProgramExample]


class ProgramCurriculumGenerator:
    """Generator for the 8 M94 program learning stages and generalization splits."""

    TRAIN_VARS = ["a", "b", "c"]
    HELD_OUT_VARS = ["x", "y", "z", "total", "acc", "val", "ctr"]

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = random.Random(seed)

    def generate_full_curriculum(self) -> M94ProgramCurriculum:
        """Construct evaluation suites across all 8 stages and held-out partitions."""
        stages: dict[int, ProgramStageSuite] = {}

        # Stage 1: Variables & Simple Assignments
        stages[1] = self._generate_stage_1(seed=self.seed + 100)

        # Stage 2: Arithmetic Expressions
        stages[2] = self._generate_stage_2(seed=self.seed + 200)

        # Stage 3: Conditions and Branches
        stages[3] = self._generate_stage_3(seed=self.seed + 300)

        # Stage 4: Loops & Length Generalization
        stages[4] = self._generate_stage_4(seed=self.seed + 400)

        # Stage 5: Stateful Accumulation & Counters
        stages[5] = self._generate_stage_5(seed=self.seed + 500)

        # Stage 6: Nested Operations (Loops containing Conditions)
        stages[6] = self._generate_stage_6(seed=self.seed + 600)

        # Stage 7: Program Execution Traces (Intermediate State Transitions)
        stages[7] = self._generate_stage_7(seed=self.seed + 700)

        # Stage 8: Novel Unfamiliar Algorithms Composed from Concepts
        stages[8] = self._generate_stage_8(seed=self.seed + 800)

        # Unsupported & Contradictory tasks
        unsupported = self._generate_unsupported_examples(seed=self.seed + 900)

        return M94ProgramCurriculum(stages=stages, unsupported_examples=unsupported)

    def _make_example(
        self,
        program_id: str,
        stage_id: int,
        category: str,
        program: Program,
        is_supported: bool = True,
        requires_abstention: bool = False,
    ) -> ProgramExample:
        code = program.to_code()
        tokens = ProgramTokenizer.tokenize_code(code)
        ref_res = ReferenceInterpreter.execute(program)
        return ProgramExample(
            program_id=program_id,
            stage_id=stage_id,
            category=category,
            program=program,
            code_text=code,
            token_ids=tokens,
            reference_result=ref_res,
            is_supported=is_supported,
            requires_abstention=requires_abstention,
        )

    # -----------------------------------------------------------------------
    # Stage 1: Variables and Assignments
    # -----------------------------------------------------------------------
    def _generate_stage_1(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Train: seen variables a, b with values 0..9
        for i in range(40):
            val1 = rng.randint(0, 9)
            val2 = rng.randint(0, 9)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(val1)),
                    AssignStmt("b", LiteralExpr(val2)),
                    PrintStmt("b"),
                ]
            )
            train_ex.append(self._make_example(f"s1_tr_{i}", 1, "standard_assign", prog))

        # Dev: seen variables, disjoint values
        for i in range(15):
            val = rng.randint(10, 15)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(val)),
                    PrintStmt("a"),
                ]
            )
            dev_ex.append(self._make_example(f"s1_dev_{i}", 1, "standard_assign", prog))

        # Test: seen variables
        for i in range(25):
            v1 = rng.randint(0, 15)
            v2 = rng.randint(0, 15)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(v1)),
                    AssignStmt("b", VarExpr("a")),
                    AssignStmt("c", LiteralExpr(v2)),
                    PrintStmt("b"),
                ]
            )
            test_ex.append(self._make_example(f"s1_te_{i}", 1, "chained_assign", prog))

        # Held-out: Unseen variable names (x, y, val, acc) and values
        for i in range(25):
            v_name = rng.choice(self.HELD_OUT_VARS)
            val = rng.randint(0, 15)
            prog = Program(
                [
                    AssignStmt(v_name, LiteralExpr(val)),
                    PrintStmt(v_name),
                ]
            )
            held_out_ex.append(self._make_example(f"s1_held_{i}", 1, "unseen_var_name", prog))

        return ProgramStageSuite(
            1, "Variables & Assignments", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Stage 2: Arithmetic Expressions
    # -----------------------------------------------------------------------
    def _generate_stage_2(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Train: Binary arithmetic on seen vars and operands in {1, 2, 3, 5}
        seen_ops = [1, 2, 3, 5]
        for i in range(40):
            v = rng.randint(0, 15)
            k = rng.choice(seen_ops)
            op = rng.choice(["+", "-"])
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(v)),
                    AssignStmt("b", BinOpExpr(VarExpr("a"), op, LiteralExpr(k))),
                    PrintStmt("b"),
                ]
            )
            train_ex.append(self._make_example(f"s2_tr_{i}", 2, "binary_arithmetic", prog))

        for i in range(15):
            v = rng.randint(0, 15)
            k = rng.choice(seen_ops)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(v)),
                    AssignStmt("a", BinOpExpr(VarExpr("a"), "+", LiteralExpr(k))),
                    PrintStmt("a"),
                ]
            )
            dev_ex.append(self._make_example(f"s2_dev_{i}", 2, "self_update_arith", prog))

        for i in range(25):
            v1 = rng.randint(0, 15)
            v2 = rng.randint(0, 15)
            op = rng.choice(["+", "-"])
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(v1)),
                    AssignStmt("b", LiteralExpr(v2)),
                    AssignStmt("c", BinOpExpr(VarExpr("a"), op, VarExpr("b"))),
                    PrintStmt("c"),
                ]
            )
            test_ex.append(self._make_example(f"s2_te_{i}", 2, "var_var_arith", prog))

        # Held-out: Unseen operands in {4, 7, 9} and unseen variable names
        held_ops = [4, 7, 9]
        for i in range(25):
            v_name = rng.choice(self.HELD_OUT_VARS)
            v = rng.randint(0, 15)
            k = rng.choice(held_ops)
            op = rng.choice(["+", "-"])
            prog = Program(
                [
                    AssignStmt(v_name, LiteralExpr(v)),
                    AssignStmt(v_name, BinOpExpr(VarExpr(v_name), op, LiteralExpr(k))),
                    PrintStmt(v_name),
                ]
            )
            held_out_ex.append(self._make_example(f"s2_held_{i}", 2, "unseen_operand_arith", prog))

        return ProgramStageSuite(
            2, "Arithmetic Expressions", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Stage 3: Conditions & Branches
    # -----------------------------------------------------------------------
    def _generate_stage_3(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Train: if-else with threshold in {3, 7}
        seen_thresh = [3, 7]
        for i in range(40):
            val = rng.randint(0, 15)
            thresh = rng.choice(seen_thresh)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(val)),
                    AssignStmt("b", LiteralExpr(0)),
                    IfStmt(
                        cond=CompareExpr(VarExpr("a"), ">", LiteralExpr(thresh)),
                        body=[AssignStmt("b", BinOpExpr(VarExpr("a"), "+", LiteralExpr(2)))],
                        else_body=[AssignStmt("b", BinOpExpr(VarExpr("a"), "-", LiteralExpr(1)))],
                    ),
                    PrintStmt("b"),
                ]
            )
            train_ex.append(self._make_example(f"s3_tr_{i}", 3, "conditional_branch", prog))

        for i in range(15):
            val = rng.randint(0, 15)
            thresh = rng.choice(seen_thresh)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(val)),
                    AssignStmt("b", LiteralExpr(1)),
                    IfStmt(
                        cond=CompareExpr(VarExpr("a"), "==", LiteralExpr(thresh)),
                        body=[AssignStmt("b", LiteralExpr(8))],
                    ),
                    PrintStmt("b"),
                ]
            )
            dev_ex.append(self._make_example(f"s3_dev_{i}", 3, "equality_branch", prog))

        for i in range(25):
            val = rng.randint(0, 15)
            thresh = rng.choice(seen_thresh)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(val)),
                    AssignStmt("b", LiteralExpr(5)),
                    IfStmt(
                        cond=CompareExpr(VarExpr("a"), "<", LiteralExpr(thresh)),
                        body=[AssignStmt("b", BinOpExpr(VarExpr("b"), "+", LiteralExpr(3)))],
                        else_body=[AssignStmt("b", BinOpExpr(VarExpr("b"), "-", LiteralExpr(2)))],
                    ),
                    PrintStmt("b"),
                ]
            )
            test_ex.append(self._make_example(f"s3_te_{i}", 3, "less_than_branch", prog))

        # Held-out: Unseen thresholds in {4, 6, 10} and unseen variable names
        held_thresh = [4, 6, 10]
        for i in range(25):
            v_name = rng.choice(self.HELD_OUT_VARS)
            val = rng.randint(0, 15)
            thresh = rng.choice(held_thresh)
            prog = Program(
                [
                    AssignStmt(v_name, LiteralExpr(val)),
                    AssignStmt("res", LiteralExpr(0)),
                    IfStmt(
                        cond=CompareExpr(VarExpr(v_name), ">", LiteralExpr(thresh)),
                        body=[AssignStmt("res", LiteralExpr(10))],
                        else_body=[AssignStmt("res", LiteralExpr(2))],
                    ),
                    PrintStmt("res"),
                ]
            )
            held_out_ex.append(self._make_example(f"s3_held_{i}", 3, "unseen_thresh_branch", prog))

        return ProgramStageSuite(3, "Conditions & Branches", train_ex, dev_ex, test_ex, held_out_ex)

    # -----------------------------------------------------------------------
    # Stage 4: Loops & Length Generalization
    # -----------------------------------------------------------------------
    def _generate_stage_4(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Train: bounded loops with 2 or 3 iterations
        for i in range(40):
            n_iter = rng.choice([2, 3])
            items = tuple(rng.randint(1, 6) for _ in range(n_iter))
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[AssignStmt("a", BinOpExpr(VarExpr("a"), "+", VarExpr("x")))],
                    ),
                    PrintStmt("a"),
                ]
            )
            train_ex.append(self._make_example(f"s4_tr_{i}", 4, "short_loop", prog))

        for i in range(15):
            items = tuple(rng.randint(1, 4) for _ in range(2))
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(10)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[AssignStmt("a", BinOpExpr(VarExpr("a"), "-", VarExpr("x")))],
                    ),
                    PrintStmt("a"),
                ]
            )
            dev_ex.append(self._make_example(f"s4_dev_{i}", 4, "decrement_loop", prog))

        for i in range(25):
            items = tuple(rng.randint(1, 5) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("b", LiteralExpr(2)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[AssignStmt("b", BinOpExpr(VarExpr("b"), "+", VarExpr("x")))],
                    ),
                    PrintStmt("b"),
                ]
            )
            test_ex.append(self._make_example(f"s4_te_{i}", 4, "standard_loop_te", prog))

        # Held-out Length Generalization: 4 and 5 iterations (never seen during training!)
        for i in range(25):
            n_iter = rng.choice([4, 5])
            items = tuple(rng.randint(1, 4) for _ in range(n_iter))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(0)),
                    ForStmt(
                        loop_var="k",
                        items=items,
                        body=[AssignStmt("total", BinOpExpr(VarExpr("total"), "+", VarExpr("k")))],
                    ),
                    PrintStmt("total"),
                ]
            )
            held_out_ex.append(
                self._make_example(f"s4_held_{i}", 4, "length_generalization_loop", prog)
            )

        return ProgramStageSuite(
            4, "Loops & Length Generalization", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Stage 5: Stateful Accumulation & Counters
    # -----------------------------------------------------------------------
    def _generate_stage_5(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        for i in range(40):
            init_val = rng.randint(0, 5)
            items = tuple(rng.randint(1, 5) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(init_val)),
                    AssignStmt("ctr", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            AssignStmt("total", BinOpExpr(VarExpr("total"), "+", VarExpr("x"))),
                            AssignStmt("ctr", BinOpExpr(VarExpr("ctr"), "+", LiteralExpr(1))),
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            train_ex.append(self._make_example(f"s5_tr_{i}", 5, "accumulator_counter", prog))

        for i in range(15):
            items = tuple(rng.randint(1, 4) for _ in range(2))
            prog = Program(
                [
                    AssignStmt("acc", LiteralExpr(15)),
                    ForStmt(
                        loop_var="val",
                        items=items,
                        body=[AssignStmt("acc", BinOpExpr(VarExpr("acc"), "-", VarExpr("val")))],
                    ),
                    PrintStmt("acc"),
                ]
            )
            dev_ex.append(self._make_example(f"s5_dev_{i}", 5, "subtractive_accumulator", prog))

        for i in range(25):
            items = tuple(rng.randint(1, 5) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(3)),
                    AssignStmt("b", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            AssignStmt("a", BinOpExpr(VarExpr("a"), "+", VarExpr("x"))),
                            AssignStmt("b", BinOpExpr(VarExpr("b"), "+", VarExpr("a"))),
                        ],
                    ),
                    PrintStmt("b"),
                ]
            )
            test_ex.append(self._make_example(f"s5_te_{i}", 5, "dual_state_coupling", prog))

        # Held-out: multi-variable persistent memory across 4 iterations
        for i in range(25):
            items = tuple(rng.randint(1, 4) for _ in range(4))
            prog = Program(
                [
                    AssignStmt("acc", LiteralExpr(1)),
                    AssignStmt("ctr", LiteralExpr(0)),
                    ForStmt(
                        loop_var="k",
                        items=items,
                        body=[
                            AssignStmt("acc", BinOpExpr(VarExpr("acc"), "+", VarExpr("k"))),
                            AssignStmt("ctr", BinOpExpr(VarExpr("ctr"), "+", LiteralExpr(2))),
                        ],
                    ),
                    PrintStmt("acc"),
                ]
            )
            held_out_ex.append(
                self._make_example(f"s5_held_{i}", 5, "held_out_state_persistence", prog)
            )

        return ProgramStageSuite(
            5, "Stateful Accumulation & Counters", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Stage 6: Nested Operations (Loops containing Conditions)
    # -----------------------------------------------------------------------
    def _generate_stage_6(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Example from user prompt:
        # total = 0
        # for x in [3, 5, 2]:
        #     if x > 2:
        #         total = total + x
        # print(total)
        for i in range(40):
            thresh = rng.choice([2, 3])
            items = tuple(rng.randint(1, 6) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("x"), ">", LiteralExpr(thresh)),
                                body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "+", VarExpr("x"))
                                    )
                                ],
                            )
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            train_ex.append(self._make_example(f"s6_tr_{i}", 6, "loop_filter_accumulate", prog))

        for i in range(15):
            thresh = 2
            items = tuple(rng.randint(1, 5) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(10)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("x"), "<", LiteralExpr(thresh)),
                                body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "-", VarExpr("x"))
                                    )
                                ],
                            )
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            dev_ex.append(self._make_example(f"s6_dev_{i}", 6, "loop_filter_sub", prog))

        for i in range(25):
            items = tuple(rng.randint(1, 6) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("x"), ">", LiteralExpr(2)),
                                body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "+", VarExpr("x"))
                                    )
                                ],
                                else_body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "+", LiteralExpr(1))
                                    )
                                ],
                            )
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            test_ex.append(self._make_example(f"s6_te_{i}", 6, "loop_branch_else", prog))

        # Held-out: Unseen thresholds (4, 5) and longer lists (4 items)
        for i in range(25):
            thresh = rng.choice([4, 5])
            items = tuple(rng.randint(1, 8) for _ in range(4))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(0)),
                    ForStmt(
                        loop_var="val",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("val"), ">", LiteralExpr(thresh)),
                                body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "+", VarExpr("val"))
                                    )
                                ],
                            )
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            held_out_ex.append(
                self._make_example(f"s6_held_{i}", 6, "held_out_nested_loop_condition", prog)
            )

        return ProgramStageSuite(6, "Nested Operations", train_ex, dev_ex, test_ex, held_out_ex)

    # -----------------------------------------------------------------------
    # Stage 7: Program Execution Traces
    # -----------------------------------------------------------------------
    def _generate_stage_7(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Example from prompt:
        # a = 2
        # a = a + 3
        # a = a + 5
        # states: a=2 -> a=5 -> a=10
        for i in range(40):
            init_v = rng.randint(0, 5)
            step1 = rng.randint(1, 5)
            step2 = rng.randint(1, 5)
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(init_v)),
                    AssignStmt("a", BinOpExpr(VarExpr("a"), "+", LiteralExpr(step1))),
                    AssignStmt("a", BinOpExpr(VarExpr("a"), "+", LiteralExpr(step2))),
                    PrintStmt("a"),
                ]
            )
            train_ex.append(self._make_example(f"s7_tr_{i}", 7, "sequential_trace", prog))

        for i in range(15):
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(3)),
                    AssignStmt("b", LiteralExpr(4)),
                    AssignStmt("a", BinOpExpr(VarExpr("a"), "+", VarExpr("b"))),
                    PrintStmt("a"),
                ]
            )
            dev_ex.append(self._make_example(f"s7_dev_{i}", 7, "two_var_trace", prog))

        for i in range(25):
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(1)),
                    AssignStmt("b", LiteralExpr(2)),
                    AssignStmt("c", BinOpExpr(VarExpr("a"), "+", VarExpr("b"))),
                    AssignStmt("a", BinOpExpr(VarExpr("c"), "+", LiteralExpr(4))),
                    PrintStmt("a"),
                ]
            )
            test_ex.append(self._make_example(f"s7_te_{i}", 7, "multi_step_trace", prog))

        # Held-out: 4-step trace with unseen variables
        for i in range(25):
            init_v = rng.randint(0, 5)
            prog = Program(
                [
                    AssignStmt("x", LiteralExpr(init_v)),
                    AssignStmt("y", LiteralExpr(2)),
                    AssignStmt("x", BinOpExpr(VarExpr("x"), "+", VarExpr("y"))),
                    AssignStmt("x", BinOpExpr(VarExpr("x"), "-", LiteralExpr(1))),
                    PrintStmt("x"),
                ]
            )
            held_out_ex.append(
                self._make_example(f"s7_held_{i}", 7, "held_out_multi_step_trace", prog)
            )

        return ProgramStageSuite(
            7, "Program Execution Traces", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Stage 8: Novel Unfamiliar Algorithms Composed from Concepts
    # -----------------------------------------------------------------------
    def _generate_stage_8(self, seed: int) -> ProgramStageSuite:
        rng = random.Random(seed)
        train_ex, dev_ex, test_ex, held_out_ex = [], [], [], []

        # Novel algorithms:
        # e.g. Count elements exceeding threshold, Alternating accumulator, Min-capped accumulator
        for i in range(40):
            thresh = 2
            items = tuple(rng.randint(1, 6) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("ctr", LiteralExpr(0)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("x"), ">", LiteralExpr(thresh)),
                                body=[
                                    AssignStmt(
                                        "ctr", BinOpExpr(VarExpr("ctr"), "+", LiteralExpr(1))
                                    )
                                ],
                            )
                        ],
                    ),
                    PrintStmt("ctr"),
                ]
            )
            train_ex.append(self._make_example(f"s8_tr_{i}", 8, "count_greater", prog))

        for i in range(15):
            items = tuple(rng.randint(1, 5) for _ in range(2))
            prog = Program(
                [
                    AssignStmt("acc", LiteralExpr(0)),
                    AssignStmt("k", LiteralExpr(3)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            AssignStmt("acc", BinOpExpr(VarExpr("acc"), "+", VarExpr("x"))),
                            AssignStmt("acc", BinOpExpr(VarExpr("acc"), "+", VarExpr("k"))),
                        ],
                    ),
                    PrintStmt("acc"),
                ]
            )
            dev_ex.append(self._make_example(f"s8_dev_{i}", 8, "step_increment_algorithm", prog))

        for i in range(25):
            items = tuple(rng.randint(1, 5) for _ in range(3))
            prog = Program(
                [
                    AssignStmt("total", LiteralExpr(5)),
                    AssignStmt("sign", LiteralExpr(1)),
                    ForStmt(
                        loop_var="x",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("sign"), "==", LiteralExpr(1)),
                                body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "+", VarExpr("x"))
                                    ),
                                    AssignStmt("sign", LiteralExpr(0)),
                                ],
                                else_body=[
                                    AssignStmt(
                                        "total", BinOpExpr(VarExpr("total"), "-", VarExpr("x"))
                                    ),
                                    AssignStmt("sign", LiteralExpr(1)),
                                ],
                            )
                        ],
                    ),
                    PrintStmt("total"),
                ]
            )
            test_ex.append(self._make_example(f"s8_te_{i}", 8, "alternating_algorithm", prog))

        # Held-out Novel Algorithm: Bounded conditional accumulator with parity toggle on 4 elements
        for i in range(25):
            items = tuple(rng.randint(1, 6) for _ in range(4))
            prog = Program(
                [
                    AssignStmt("res", LiteralExpr(0)),
                    AssignStmt("mode", LiteralExpr(0)),
                    ForStmt(
                        loop_var="val",
                        items=items,
                        body=[
                            IfStmt(
                                cond=CompareExpr(VarExpr("val"), ">", LiteralExpr(3)),
                                body=[
                                    AssignStmt(
                                        "res", BinOpExpr(VarExpr("res"), "+", VarExpr("val"))
                                    ),
                                    AssignStmt("mode", LiteralExpr(1)),
                                ],
                                else_body=[
                                    AssignStmt(
                                        "res", BinOpExpr(VarExpr("res"), "+", LiteralExpr(1))
                                    ),
                                ],
                            )
                        ],
                    ),
                    PrintStmt("res"),
                ]
            )
            held_out_ex.append(
                self._make_example(f"s8_held_{i}", 8, "novel_unseen_algorithm", prog)
            )

        return ProgramStageSuite(
            8, "Novel Unfamiliar Algorithms", train_ex, dev_ex, test_ex, held_out_ex
        )

    # -----------------------------------------------------------------------
    # Unsupported & Contradictory Tasks (Abstention Testing)
    # -----------------------------------------------------------------------
    def _generate_unsupported_examples(self, seed: int) -> list[ProgramExample]:
        """Programs with invalid syntax or undeclared variables that require abstention."""
        random.Random(seed)
        unsupported: list[ProgramExample] = []

        # 1. Undeclared variable access
        for i in range(10):
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(2)),
                    AssignStmt("b", BinOpExpr(VarExpr("a"), "+", VarExpr("unseen_ghost_var"))),
                    PrintStmt("b"),
                ]
            )
            unsupported.append(
                self._make_example(
                    f"unsup_undef_{i}",
                    0,
                    "undefined_variable",
                    prog,
                    is_supported=False,
                    requires_abstention=True,
                )
            )

        # 2. Syntax/Unknown token programs
        for i in range(10):
            # Program trying to print an undefined variable
            prog = Program(
                [
                    AssignStmt("a", LiteralExpr(5)),
                    PrintStmt("z_unassigned"),
                ]
            )
            unsupported.append(
                self._make_example(
                    f"unsup_unassigned_print_{i}",
                    0,
                    "unassigned_print",
                    prog,
                    is_supported=False,
                    requires_abstention=True,
                )
            )

        return unsupported
