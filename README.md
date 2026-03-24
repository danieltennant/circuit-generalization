# Circuit Generalization

Research into whether mechanistic interpretability circuits generalize across diverse inputs — and what predicts when they don't.

---

## The Question

When a circuit is identified for a task on a specific prompt, does that circuit activate consistently across other prompts that require the same underlying computation? What properties of the task, prompt, or model predict generalization?

This is an open problem flagged explicitly in Anthropic's 2025 "Biology of a Large Language Model" paper: their attribution graph methodology succeeds on roughly 25% of attempted prompts, and the reasons for failure are not well understood.

This project builds a measurement pipeline to systematically test circuit generalization across lexical, syntactic, domain, and adversarial variation, starting with the IOI (Indirect Object Identification) circuit in GPT-2 Small.

---

## Why This Matters

If safety-relevant circuits (e.g. refusal mechanisms) are prompt-specific rather than stable, then:
- Safety audits using circuit tracing give false confidence
- Findings from one context may not transfer to others
- Models may rely on surface-level pattern matching rather than stable internal algorithms

---

## Disclaimer

This is independent research I'm doing for my own education and curiosity. I'm not a professional researcher, and the work here should be read accordingly. I'm sharing it publicly in the spirit of open science and shared education — replication attempts, critiques, and suggestions are welcome.

---

## Structure

```
circuit-generalization/
├── LAB_NOTEBOOK.md  # Running log of work done, decisions made, and things learned
├── notebooks/       # Exploratory notebooks — named wNN-description.ipynb
├── src/             # Reusable Python modules extracted from notebooks
├── experiments/     # Experiment configs and raw output CSVs
├── notes/           # Replication write-ups and research notes (may be published)
└── results/         # Processed figures and summary tables
```

## Using the Code

**Environment setup:**

```bash
conda env create -f environment.yml
conda activate circuit-gen
```

Key dependencies: Python 3.11, PyTorch 2.10, TransformerLens, JupyterLab.

**Running notebooks:**

```bash
conda activate circuit-gen && jupyter lab
```

Notebooks are numbered by week (`w01`, `w02`, ...) and build on each other. Start from `w01-residual-stream.ipynb` if you're following along from the beginning.

**Reusable pipelines:**

As experiments mature, reusable code is extracted into `src/`. The core artifact of this project is a circuit scoring pipeline: given a circuit specification and a prompt dataset, it returns per-prompt activation scores. This will live in `src/circuit_scorer.py` once built.

---

## Research Arc

This is the first of three planned projects:

1. **Circuit Generalization** ← here — does a circuit activate consistently across diverse inputs?
2. **Fine-Tuning Effects** — do circuits survive fine-tuning, and what predicts fragility?
3. **Backdoor Detection** — can we plant a circuit via fine-tuning and then find it?
