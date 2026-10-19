# Agent Instructions — Adaptive Mini-GPT

Read all `docs/*.md` and `configs/tiny.yaml` before modifying source code. Treat milestone gates and mathematical invariants as binding.

## Project Status & Active Scope
- **M0, M1, M2, M3, M4, and M5 are complete and verified on NVIDIA Tesla T4.** Do not regress existing unit or integration tests in `tests/`.
- **Core Modules:**
  - `model/`: Pre-norm causal SDPA Transformer (`transformer.py`, `attention.py`, `config.py`).
  - `adaptive/`: Bottleneck residual experts (`experts.py`), load-balanced & task routers (`router.py`), function-preserving growth (`growth.py`), and causal Mixture-of-Depths (`mod.py`).
  - `data/`: Base-ASCII & special-token `CharacterTokenizer`, vectorized answer masking, and disjoint holdout arithmetic generator.
  - `evaluation/`: Perplexity, BPC, and per-operation exact-match skill verifiers.
- **Tooling:** Python 3.11+, PyTorch 2.4+, `uv`, `pytest`, `ruff`, `pyright`, and `google-colab-cli` (via `uvx`) for headless T4 execution.

## Architectural & Mathematical Guardrails
1. **Fused Causal Attention:** Use `F.scaled_dot_product_attention` with `is_causal=True` during standard causal passes, while retaining explicit mask support for custom inspection tests.
2. **Decoupled Weight Decay:** Never apply weight decay to 1D tensors (`LayerNorm`, biases) or embeddings. Partition optimizer parameter groups into `decay` ($\text{dim} \ge 2$) and `no_decay` ($\text{dim} < 2$).
3. **Function-Preserving Growth Invariant:** When adding new experts or expanding depth at a checkpoint boundary, initialize residual output projections to $\mathbf{0}$ so that forward-pass logits before and after growth are bit-for-bit identical ($f_{\text{grown}}(x) \equiv f_{\text{base}}(x)$ at step $t_0$).
4. **Zero-Forgetting Selective Token Adaptation:** When adapting unseen domain tokens (`trainable_token_ids`), apply gradient masking on `wte` while keeping `lm_head` frozen so prior tasks experience strictly $0.000000$ softmax denominator drift when `bypass_moe=True`.
5. **Causal Mixture-of-Depths (M5):** Physically gather active tokens before FFN/MoE execution for real FLOP reduction, scale outputs by continuous gate $\sigma(z_{b,t})$ for router differentiability, and use token-local thresholding ($z_{b,t} \ge 0.0$) in `eval()` mode to preserve strict causal independence.
6. **Headless Colab Discipline:** Always use non-interactive `uvx --from google-colab-cli colab exec -s <session> -f ...` commands and immediately release GPU instances with `colab stop -s <session>` after runs finish.
