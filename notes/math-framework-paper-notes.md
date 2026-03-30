# Reading Notes — A Mathematical Framework for Transformer Circuits

Elhage et al., 2021 — transformer-circuits.pub/2021/framework/index.html

*Read through the one-layer attention-only transformers section (March 2026).*
*Two-layer section is Week 3 material.*

---

## Reading Experience

### What Made It Hard

**Notation density.** The paper assumes fluency with linear algebra notation that isn't spelled out. Early on, the QK/OV decomposition, einsum indexing, and tensor shape conventions created significant friction — it was easy to follow the prose but lose the thread when equations appeared. I was able to struggle through the paper and get the key concepts with some AI-assisted explanations, but I definitely need to do more work on the fundamentals to read these types of papers.

**Vocabulary without definitions.** Terms like *logits*, *autoregressive masking*, and *softmax* appear without introduction. The paper assumes you're already fluent in transformer basics. Several concepts required separate clarification before the surrounding argument made sense.

**The virtual weights jump.** The idea that you can multiply out the full path from embedding to unembedding and treat the composition as a single effective matrix is a conceptual leap. The paper states it efficiently, but internalizing *why* it's valid — and what it buys you analytically — took some wrangling.

---

## Key Concepts and Takeaways

### The Residual Stream as a Communication Channel

The residual stream isn't a sequence of transformations — it's a shared workspace. Each attention head and MLP reads from it and writes back to it additively. Nothing is overwritten; everything accumulates. This reframes how to think about what a layer is "doing": it's contributing a delta, not replacing a state.

### Logits and the Logit Lens

Logits are the raw unnormalized scores produced by multiplying the residual stream by the unembedding matrix **W_U** — one score per vocabulary token. Softmax converts them to probabilities. The logit lens exploits this: project *intermediate* residual stream states through **W_U** to see what the model is effectively predicting at each layer, before the computation is finished. Useful for understanding how predictions form across depth.

### The QK / OV Decomposition

Each attention head decomposes into two independent circuits:

- **QK circuit** (`W_Q W_K^T`): determines *where* to attend — which positions attend to which
- **OV circuit** (`W_V W_O`): determines *what to write* given the attended positions

These can be analyzed separately. The QK circuit controls routing; the OV circuit controls content. A head that attends to the right place but writes the wrong thing fails differently than one that routes incorrectly.

### Low-Rank Structure of the QK Matrix

`W_Q` and `W_K` are each shaped `d_model × d_head` (e.g. 768 × 64 in GPT-2 Small). Their product `W_QK = W_Q W_K^T` is a 768×768 matrix but has rank at most 64. The head can only distinguish tokens along 64 directions in the residual stream — everything orthogonal to those directions is invisible to its attention pattern.

This matters for interpretability: you can SVD `W_QK` to find the directions the head actually uses. And it matters for circuit generalization: a head trained on one distribution may have its 64 directions tuned to features present in that distribution, and fail to attend correctly on inputs that route through different subspace directions.

### Autoregressive (Causal) Masking

Before softmax, attention logits at position *i* for positions *j > i* are set to −∞, making them zero after softmax. This enforces a lower-triangular attention pattern: each position can only attend to itself and earlier positions. The mask enforces at inference time the same left-to-right dependency structure the model was trained with.

For circuit analysis, the mask is a hard constraint on information flow. When tracing a circuit, you can rule out any path that requires a position to read from a later position.

### Skip-Trigrams and Compositional Attention

In a one-layer model, attention heads can implement *skip-trigrams*: patterns of the form "if token A appears earlier, increase the probability of token C after token B." The head attends from B back to A, reads A's value via the OV circuit, and adds it to the residual stream at B's position before prediction. This is a concrete, mechanistic account of a learned behavior — not just "the model learned trigram statistics" but exactly *how* it computes them.

### Virtual Weights

Because the residual stream is additive, you can trace a signal from embedding all the way to unembedding through a specific path — bypassing intermediate layers — and compute an effective "virtual" weight matrix for that path. This path expansion is what makes circuit analysis tractable: rather than reasoning about all interactions, you identify the paths that matter and ignore the rest.

---

## Relation to Research Question

The low-rank structure of QK is directly relevant. If a circuit is defined by the specific subspace directions a set of heads use for routing and writing, then "circuit generalization" becomes a question about whether test inputs activate those same directions. Inputs that are superficially different but project similarly onto the relevant subspaces should generalize; inputs that look similar but live in different subspaces may not. This is a concrete, testable framing.
