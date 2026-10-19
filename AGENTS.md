# Agent Instructions — Adaptive Mini-GPT

Read all `docs/*.md` and `configs/tiny.yaml` before modifying source code. Treat milestone gates and mathematical invariants as binding.

## Project Status & Active Scope
- **M0, M1, and M2 are complete and verified.** Do not regress existing unit or integration tests in `tests/`.
- **Active Scope (M3 & M4):** Modular residual experts (`adaptive/experts.py`), task-conditioned & learned top-$k$ routing with load balancing (`adaptive/router.py`), and checkpoint-boundary structural growth (`adaptive/growth.py`).
- **Tooling:** Python 3.11+, PyTorch 2.4+, `uv`, `pytest`, `ruff`, `pyright`. Standard library `argparse` for `amg` CLI.

## Architectural & Mathematical Guardrails
1. **Fused Causal Attention:** Use `F.scaled_dot_product_attention` with `is_causal=True` during standard causal passes, while retaining explicit mask support for custom inspection tests.
2. **Decoupled Weight Decay:** Never apply weight decay to 1D tensors (`LayerNorm`, biases) or embeddings. Partition optimizer parameter groups into `decay` ($\text{dim} \ge 2$) and `no_decay` ($\text{dim} < 2$).
3. **Function-Preserving Growth Invariant:** When adding new experts or expanding capacity at a checkpoint boundary, initialize the output projection $W_{\text{down}} = \mathbf{0}$ so that forward-pass logits before and after growth are bit-for-bit identical ($f_{\text{grown}}(x) \equiv f_{\text{base}}(x)$ at step $t_0$).
4. **Auxiliary Loss Accounting:** When learned routing is active, track `ce_loss`, `aux_load_balance_loss`, and `router_z_loss` separately in telemetry while optimizing their weighted sum:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha \mathcal{L}_{\text{balance}} + \beta \mathcal{L}_{z}$$
5. **Safe Checkpoint Transitions:** Support strict restoration for interrupted runs (`strict=True`) and explicit partial/backbone restoration (`allow_growth=True`) when migrating a dense checkpoint into an expanded modular architecture.
