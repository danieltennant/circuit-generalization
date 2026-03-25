# Lab Notebook — Circuit Generalization

*A running log of work done, decisions made, and things learned. New entries go at the top.*

---

## 2026-03-24

Fixed bugs in `w01-residual-stream.ipynb` to get it running. Listened to a NotebookLM audio summary of "A Mathematical Framework for Transformer Circuits" (Elhage 2021) and started reading the paper. Plan reading covers: Summary of Results, Transformer Overview, Zero-Layer Transformers, and One-Layer Attention-Only Transformers — stopping before the Two-Layer section, which is Week 2 material.

---

## 2026-03-23

Finished 80k Hours Ep. 107: Chris Olah. Good orientation to the field before touching the math — the framing of mech interp as "trying to read the source code" rather than just observe behavior is a useful mental model to carry into the notebook work.

Created `w01-residual-stream.ipynb` — covers loading GPT-2 Small via HookedTransformer, running a forward pass with activation caching, inspecting residual stream norms across layers, decomposing attention vs MLP contributions, and verifying next-token predictions.

---

## 2026-03-22 — Week 1 Plan

- [x] Listen: [80k Hours Ep. 107 — Chris Olah](https://80000hours.org/podcast/episodes/chris-olah-interpretability-research/)
- [ ] Read: [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) — Elhage 2021, sections 1–4 (residual stream, QK circuit, OV circuit, virtual attention heads)
- [ ] [ARENA exercises 1.1–1.3](https://github.com/callummcdougall/ARENA_3.0/blob/main/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb)
- [x] Create Week 1 notebook: load GPT-2 Small, inspect residual stream norms, decompose attention vs MLP contributions
- [ ] Execute notebook, review plots, and fill in reflection cell

---

*Entries above this line are most recent.*
