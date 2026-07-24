# BlueDot Technical AI Safety Puzzle

Code for the [BlueDot Impact Technical AI Safety puzzle](https://bluedot.org/puzzles/technical-ai-safety).

The puzzle provides a small MLP trained to predict 8 binary features from short text inputs. After a particular hidden layer, 7 features are linearly represented; one is not. The goal is to find it, explain the geometry, and train a model with an even stranger representation.

## Scripts

| Script | What it does |
|---|---|
| `01_linear_probes.py` | Trains logistic regression probes on layer-2 activations and measures accuracy vs the full model for each feature. Country is the clear outlier (42.9% probe vs 96.4% full model). |
| `02_geometry_analysis.py` | Full geometry analysis: UMAP, per-dimension MI histograms, pairwise scatter plots, superposition (joint food × country colouring), and LDA projection revealing the band structure. |
| `03_ring_model.py` | Standalone Part 3. Defines a 2D ring label, embeds into 64 dims, trains an MLP with a 2-neuron bottleneck, evaluates with linear/nonlinear probes, and visualises the learned bottleneck geometry. |

## Setup

```bash
pip install numpy torch scikit-learn umap-learn matplotlib
```

## Usage

**Scripts 01 and 02** require the puzzle model and dataset. Adapt the loading section at the top of each file to match the puzzle's provided interface, then pass `acts` (layer-2 activations, shape `(N, 64)`) and `y` (binary labels, shape `(N, 8)`) to each function.

**Script 03** is fully self-contained — it generates synthetic data, trains the ring model, and produces all figures.

```bash
python 03_ring_model.py
```

## Key findings

- **Country** is the non-linear feature: a linear probe on layer-2 activations achieves only 42.9%, while the full model reaches 96.4%.
- Country and food **share a single activation direction** via superposition. Country occupies the *middle interval* of that axis (bounded above by food-only examples and below by neither), making it unreadable by any linear threshold but decodable via a bandpass (non-linear) operation.
- The **ring model** (Part 3) trains a 2-neuron bottleneck that spontaneously recovers the original 2D ring geometry. Non-linear probes match the full model at ~96%; a linear probe sits at baseline.
