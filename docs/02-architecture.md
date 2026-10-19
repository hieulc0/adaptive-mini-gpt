# Architecture and Mathematical Specification

## 1. Backbone: Pre-Norm Decoder-Only Transformer (M0–M2)

The base model maps token indices $x \in \{0, \dots, V-1\}^T$ to next-token logits:

$$h^{(0)} = \text{Dropout}(W_e[x] + W_p[0 \dots T-1])$$

For each layer $l \in \{1, \dots, L\}$:
$$\tilde{h}^{(l)} = h^{(l-1)} + \text{CausalSDPA}(\text{LN}_1(h^{(l-1)}))$$
$$h^{(l)} = \tilde{h}^{(l)} + \text{FFN}(\text{LN}_2(\tilde{h}^{(l)})) + \text{AdaptiveMoE}(\text{LN}_2(\tilde{h}^{(l)}))$$

where $\text{AdaptiveMoE}(\cdot) = 0$ when `adaptive.enabled = false`.
- **Attention:** Multi-head causal self-attention using PyTorch `F.scaled_dot_product_attention` (enabling FlashAttention / memory-efficient kernels on T4 GPUs).
- **Initialization:** Normal $\mathcal{N}(0, 0.02)$ for linear/embedding weights; GPT-2 depth-scaled initialization $\mathcal{N}(0, 0.02 / \sqrt{2L})$ on residual output projections (`c_proj`).

---

## 2. Modular Residual Experts (M3)

When `adaptive.enabled = true`, each Transformer block (or designated adaptive layers) augments the dense FFN with $E$ parallel bottleneck experts $\{E_1, \dots, E_E\}$.

### Bottleneck Expert Structure
Each expert $E_i: \mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{model}}}$ uses a low-rank bottleneck dimension $d_{\text{expert}}$ (e.g., $d_{\text{expert}} = d_{\text{ff}} / 4$):

$$E_i(u) = W_{\text{down}, i} \left( \text{GELU}(W_{\text{up}, i} u) \right)$$

### Routing Mechanisms
We support two explicit routing regimes:

1. **Task-Conditioned Routing (`routing_mode: "task"`):**
   Given a discrete skill/domain identifier $c \in \{0, \dots, E-1\}$, token states route deterministically to expert $E_c$ alongside the shared dense FFN. Freezing the backbone and experts $\{E_0, \dots, E_{c-1}\}$ guarantees **zero interference** on previously mastered tasks.
2. **Learned Top-$k$ Routing (`routing_mode: "learned_topk"`):**
   A linear router $W_r \in \mathbb{R}^{E \times d_{\text{model}}}$ produces router logits $z(u) = W_r u$. We select the top-$k$ indices $\mathcal{K} = \text{TopK}(z(u), k)$ and normalize probabilities over the selected set:
   $$p_i(u) = \frac{\exp(z_i(u))}{\sum_{j \in \mathcal{K}} \exp(z_j(u))}, \quad i \in \mathcal{K}$$
   The residual update is:
   $$\text{AdaptiveMoE}(u) = \sum_{i \in \mathcal{K}} p_i(u) E_i(u)$$

### Load Balancing & Router Stability Objectives
To prevent expert collapse and numerical instability in FP16:
- **Load-Balancing Loss ($\mathcal{L}_{\text{balance}}$):** Let $f_i$ be the fraction of tokens dispatched to expert $i$, and $P_i$ be the mean router probability allocated to expert $i$ across the batch:
  $$\mathcal{L}_{\text{balance}} = E \sum_{i=1}^{E} f_i \cdot P_i$$
- **Router Z-Loss ($\mathcal{L}_{z}$):** Penalizes large router logits for FP16 stability:
  $$\mathcal{L}_{z} = \frac{1}{B \cdot T} \sum_{b, t} \left( \log \sum_{i=1}^{E} \exp(z_i(u_{b,t})) \right)^2$$

---

## 3. Function-Preserving Structural Growth (M4)

Structural growth is executed as a **verified checkpoint transformation** $\mathcal{G}(\theta_t) \to \theta'_t$:

1. **Adding Residual Experts (`grow_experts`):**
   To grow from $E$ to $E + \Delta E$ experts, copy all existing parameters $\theta_t$. For each new expert $j \in \{E+1, \dots, E+\Delta E\}$:
   - Initialize $W_{\text{up}, j} \sim \mathcal{N}(0, 0.02)$.
   - Initialize $W_{\text{down}, j} = \mathbf{0}$ (exact zeros).
   - Expand router $W_r$ by appending $\Delta E$ rows initialized to $\mathbf{0}$.
   **Invariant:** Because $W_{\text{down}, j} = \mathbf{0}$, $E_j(u) = \mathbf{0}$ at initialization, preserving exact forward trajectories while permitting non-zero gradients $\nabla_{W_{\text{down}, j}} \mathcal{L} \neq 0$ immediately on step $t+1$.
2. **Adding Depth (`grow_layers`):**
   Insert a new `TransformerBlock` where both `attn.c_proj` and `mlp.c_proj` weights and biases are initialized to $\mathbf{0}$, making the new block an exact identity operator $x + 0 = x$ at step $t_0$.

---

## 4. Mixture-of-Depths (M5): Dynamic Compute Allocation

Following Raposo et al. (2024), **Mixture-of-Depths (MoD)** dynamically allocates sub-layer FLOPs across sequence positions by routing only a fraction of tokens through the heavy $\text{FFN}(\cdot)$ and $\text{AdaptiveMoE}(\cdot)$ sub-layers, allowing easy tokens to bypass computation via the residual stream.

### Capacity Factor & True FLOP Reduction
A scalar capacity factor $C \in (0.0, 1.0]$ defines the total token quota per sequence:
$$k = \max(1, \lfloor C \cdot T \rceil)$$

Rather than using wasteful post-compute masking (`torch.where` after full FFN evaluation), the implementation **physically gathers** only the $N_{\text{active}}$ selected token vectors before dispatch:
$$u_{\text{active}} = u[\mathcal{I}_{\text{active}}] \in \mathbb{R}^{N_{\text{active}} \times 1 \times d_{\text{model}}}$$
This reduces actual forward and backward FLOPs through FFN and MoE by exactly $(1 - C)$, saving $50\%$ of parameter activations when $C = 0.5$.

### Differentiable Router & Gating
A linear router $w_g \in \mathbb{R}^{d_{\text{model}}}$ computes scalar routing logits:
$$z_{b,t} = w_g^\top u_{b,t}, \quad g_{b,t} = \sigma(z_{b,t})$$

The residual sub-layer output is scaled by the continuous gate weight $g_{b,t}$ before scatter-adding into the residual stream:
$$\Delta h_{b,t} = g_{b,t} \cdot \text{SubLayer}(u_{b,t})$$
This ensures $\nabla_{w_g} \mathcal{L} \neq 0$, allowing task loss gradients to flow directly into the gating parameters.

### Auxiliary Predictor Loss & Causal Inference
1. **Training Mode (Top-$k$ + Auxiliary BCE):**
   Tokens are ranked per-sequence via $\text{TopK}(z_{b, \cdot}, k)$. An auxiliary binary cross-entropy loss trains the router logits to predict top-$k$ set membership:
   $$\mathcal{L}_{\text{mod}} = \text{BCEWithLogits}(z_{b,t}, y_{b,t}^{\text{top-}k})$$
2. **Causal Inference Mode (Token-Local Thresholding):**
   During autoregressive evaluation and token generation, top-$k$ selection across time dimension $T$ is disabled because it would violate causality (future tokens altering past routing decisions). Instead, tokens route conditionally based on the learned local threshold:
   $$\mathcal{I}_{\text{active}} = \{ (b, t) \mid z_{b,t} \ge 0.0 \}$$
   Because $z_{b,t}$ depends strictly on causal attention outputs at positions $\le t$, causal mask mathematical independence is strictly preserved.
