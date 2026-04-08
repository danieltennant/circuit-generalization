# Lab Notebook — Circuit Generalization

*A running log of work done, decisions made, and things learned. New entries go at the top.*

---

## 2026-04-07

Finished Andrej Karpathy's "Let's build GPT from scratch." Very informative but super dense — estimate I fully understood about 2/3 of it. Appreciated that it pulled almost directly from the original papers rather than simplifying things away. May revisit his makemore series that he references during the video. Will likely need to rewatch some sections once the surrounding concepts are more solid.

---

## 2026-04-01

3Blue1Brown Essence of Linear Algebra episodes 1–4 complete (vectors, linear combinations, matrix multiplication, determinant). Key terms now solid: basis, linear combination, span. Episode 9 (dot products) remaining.

---

## 2026-03-30 — Week 2 Plan (Math & Transformer Foundations)

- [ ] Listen: [80k Hours — Neel Nanda on Mechanistic Interpretability](https://80000hours.org/podcast/episodes/neel-nanda-mechanistic-interpretability/) (~3 hrs, commute-friendly)
- [ ] Listen: [AXRP Ep. 14 — Chris Olah on Interpretability](https://axrp.net/episode/2021/12/02/episode-14-interpretability-chris-olah.html) (~1 hr, commute-friendly)
- [ ] Read: [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [ ] Watch: [Andrej Karpathy — Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) (~2 hrs)
- [~] Watch: [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (priority episodes: 1, 2, 3, 4 ✓ — episode 9 remaining)
- [ ] Pencil and paper: `resources/linear_algebra_for_mech_interp.pdf` sections 1–4
- [ ] Code: manual attention forward pass in PyTorch (no TransformerLens); einsum practice notebook

---

## 2026-03-29 — Week 1 notebook

Executed `w01-residual-stream.ipynb` and completed the reflection cell. Core concepts landed well: correctly identified the three residual stream dimensions (batch, position, d_model) and noted that mean norm grows across layers. Missed that the `<|endoftext|>` BOS token at position 0 is the real norm outlier — norm climbs to ~3000 by layer 9 while content tokens stay in the 50–175 range, a known GPT-2 quirk where the BOS token acts as a garbage collector for unplaced information. Also noted the L11 attention dominance but wasn't sure what to make of it — likely the final layer doing contextual assembly before unembedding, while earlier MLP-heavy layers handle lookup-style computation. Prediction quality observation was accurate: nothing surprising, with " over" → " the" at 34% being the standout.

---

## 2026-03-29

PR [callummcdougall/ARENA_3.0#295](https://github.com/callummcdougall/ARENA_3.0/pull/295) was accepted and merged. A small fix — replacing a deprecated `pkg_resources` call — but it's a tangible first contribution to the mech interp ecosystem. Nice milestone.

Finished reading the Math Framework paper (Elhage 2021) through the one-layer attention-only transformers section — covering zero-layer transformers, bigram statistics, and the full one-layer model including QK and OV circuit decomposition. Q&A session to clarify concepts: logits and the logit lens, softmax, tensor product (⊗), low-rank structure of the QK matrix and its implications for circuit analysis, and autoregressive (causal) masking. Linear algebra foundations from Week 1.5 are paying off — notation friction is significantly lower than the first pass through the paper.

---

## 2026-03-28

Generated a pencil-and-paper linear algebra exercise set (`resources/linear_algebra_for_mech_interp.pdf`) — 21 exercises across 5 sections with worked solutions, formatted in paper-style LaTeX. Sections cover vectors and dot products, matrix multiplication, linear transformations, attention scores by hand (full $QK^T/\sqrt{d}$ + softmax walkthrough), and einsum notation. Compiled with tectonic. The goal is to build intuition for the notation in the Math Framework paper before going deeper into ARENA.

Also added a Week 2 math foundations sprint to the research plan, inserting a focused week between the initial environment setup (Week 1) and the induction heads work (Week 3). Resources for that week: Illustrated Transformer, Andrej Karpathy's "Let's build GPT from scratch," 3Blue1Brown Essence of Linear Algebra (priority episodes), 80k Hours Neel Nanda (pulled forward), and AXRP Ep. 14.

---

## 2026-03-26

Worked through the key concepts of the Math Framework paper (Elhage 2021) via Q&A: residual stream as communication channel, virtual weights, attention heads as independent and additive, OV/QK circuit decomposition, path expansion trick, skip-trigrams. Concepts are landing but linear algebra notation is a friction point — added 3Blue1Brown Essence of Linear Algebra as an optional refresher before continuing with the paper.

Completed ARENA exercises 1.1, 1.2, and 1.3:
- 1.1: Inspected GPT-2 Small config — 12 layers, 12 heads, 1024 context window
- 1.2: Computed prediction accuracy from logits — model got 32/109 tokens correct (~29%)
- 1.3: Manually reproduced the layer 0 attention pattern from Q and K vectors (dot products → mask → scale → softmax); confirmed it matches the cached pattern. Linear algebra notation (einsum, tensor shapes) was a significant friction point here — the 3Blue1Brown refresher is now a higher priority before continuing.

Hit a setup issue in exercise 1.1 — the ARENA notebook uses `pkg_resources`, which was removed in `setuptools` v82.0.0. Fixed locally and submitted a PR to the ARENA repo to replace it with `importlib.metadata.packages_distributions`: [callummcdougall/ARENA_3.0#295](https://github.com/callummcdougall/ARENA_3.0/pull/295)

---

## 2026-03-24

Fixed bugs in `w01-residual-stream.ipynb` to get it running. Listened to a NotebookLM audio summary of "A Mathematical Framework for Transformer Circuits" (Elhage 2021) and started reading the paper. Plan reading covers: Summary of Results, Transformer Overview, Zero-Layer Transformers, and One-Layer Attention-Only Transformers — stopping before the Two-Layer section, which is Week 3 material.

---

## 2026-03-23

Finished 80k Hours Ep. 107: Chris Olah. Good orientation to the field before touching the math — the framing of mech interp as "trying to read the source code" rather than just observe behavior is a useful mental model to carry into the notebook work.

Created `w01-residual-stream.ipynb` — covers loading GPT-2 Small via HookedTransformer, running a forward pass with activation caching, inspecting residual stream norms across layers, decomposing attention vs MLP contributions, and verifying next-token predictions.

---

## 2026-03-22 — Week 1 Plan (Environment Setup & One-Layer Models)

- [x] Listen: [80k Hours Ep. 107 — Chris Olah](https://80000hours.org/podcast/episodes/chris-olah-interpretability-research/)
- [x] Read: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage 2021, sections 1–4 (residual stream, QK circuit, OV circuit, virtual attention heads)
- [ ] **Optional linear algebra refresher:** [3Blue1Brown — Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) (~3 hrs total) — covers matrix multiplication, linear transformations, dot products, subspaces, and eigenvalues; useful for making the paper's notation feel concrete rather than abstract
- [x] [ARENA exercises 1.1–1.3](https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb)
- [x] Create Week 1 notebook: load GPT-2 Small, inspect residual stream norms, decompose attention vs MLP contributions
- [ ] Execute notebook, review plots, and fill in reflection cell

---

*Entries above this line are most recent.*
