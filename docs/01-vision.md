# Vision and problem statement

## Purpose
Study **adaptive capacity allocation** in a small decoder-only language model trained from scratch. The long-term system may increase width, depth, or specialist capacity as justified by data, while deactivating or freezing components that should not update.

## Clarify the terminology
- **Width (`d_model`)**: dimension of token states; widening affects embeddings, attention, FFNs, and attached modules.
- **Depth (`n_layers`)**: sequential Transformer transformations.
- **Attention heads (`n_heads`)**: parallel attention subspaces; changing head count alone at fixed width doesn't necessarily add parameters.
- **FFN width (`d_ff`)**: intermediate hidden representation; can be expanded within a module.
- **Expert count / active experts**: total and per-token active specialist modules.
- **Context length**: sequence window; changes compute/memory without necessarily changing parameter count.

Capacity is not knowledge: more weights don't automatically improve generalization. A selected expert is not guaranteed to be skilled at its named task. Routing must be learned and evaluated.

## Success criteria
1. Fixed small Transformer learns on a permitted text corpus and generalizes measurably beyond training examples.
2. Sessions resume correctly from fully validated checkpoints.
3. Modular specialist experiments retain prior task performance, with measurable forgetting.
4. Adaptive growth beats fixed baselines **at matched compute/data/evaluation** or is rejected transparently.

## Constraints
One developer, small budget, local Arch/Linux development, intermittent interactive Google Colab T4 when available, external durable checkpoints. GPU/session lengths are not guaranteed. No automation designed to bypass free-tier service restrictions.

## Research questions
- When does new capacity help more than additional updates at fixed capacity?
- Does keeping old experts frozen reduce catastrophic forgetting?
- Can routers learn useful specialization without expert collapse?
- How do training GPU-hours, tokens seen, active parameters and total memory trade off?
