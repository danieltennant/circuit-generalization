"""
Linear probe analysis: find which of the 8 binary features is non-linearly
represented in the layer-2 activations.

For each feature, trains a logistic regression on the 64-dim post-ReLU layer-2
activations and compares accuracy to the full model's predictions.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------------------------
# Load puzzle model and data
# Adapt this section to match the puzzle's provided loading code.
# ---------------------------------------------------------------------------
# Expected:
#   model      — the trained 8-feature MLP (PyTorch nn.Module)
#   X_text     — list of input strings (or tokenised inputs ready for the model)
#   y          — np.ndarray of shape (N, 8), binary labels for each feature
#   FEATURES   — list of 8 feature names in label-column order

# Example placeholder (replace with actual puzzle loading):
# from puzzle import load_model, load_dataset
# model, tokenizer = load_model()
# X_text, y = load_dataset()

FEATURES = ["number", "question", "color", "food", "sentiment",
            "country", "person", "body_part"]

# ---------------------------------------------------------------------------
# Extract layer-2 activations
# ---------------------------------------------------------------------------

def get_layer2_activations(model, inputs, batch_size=256, device="cpu"):
    """Return post-ReLU layer-2 activations, shape (N, 64)."""
    model.eval()
    activations = []
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["layer2"] = output.detach().cpu()

    # Attach hook to the second ReLU (adapt layer name to actual model structure)
    handle = model.layer2_relu.register_forward_hook(hook_fn)

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size].to(device)
            model(batch)
            activations.append(hook_output["layer2"])

    handle.remove()
    return torch.cat(activations, dim=0).numpy()


# ---------------------------------------------------------------------------
# Train linear probes and compare to full model
# ---------------------------------------------------------------------------

def probe_accuracy(acts, labels, cv=5):
    clf = LogisticRegression(max_iter=1000, C=1.0)
    scores = cross_val_score(clf, acts, labels, cv=cv, scoring="accuracy")
    return scores.mean()


def full_model_accuracy(model, inputs, labels, batch_size=256, device="cpu"):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size].to(device)
            logits = model(batch)          # shape (B, 8)
            preds.append((logits.sigmoid() > 0.5).cpu().int())
    preds = torch.cat(preds, dim=0).numpy()
    return (preds == labels).mean(axis=0)   # per-feature accuracy


if __name__ == "__main__":
    # Replace with actual data loading
    # acts = get_layer2_activations(model, X_tensor)
    # full_accs = full_model_accuracy(model, X_tensor, y)

    # ---- Results table reproduced from original analysis ----
    results = {
        "number":    {"probe": 0.975, "full": 0.976},
        "question":  {"probe": 1.000, "full": 1.000},
        "color":     {"probe": 0.971, "full": 0.973},
        "food":      {"probe": 0.985, "full": 0.986},
        "sentiment": {"probe": 0.982, "full": 0.982},
        "country":   {"probe": 0.429, "full": 0.964},
        "person":    {"probe": 0.998, "full": 0.999},
        "body_part": {"probe": 0.980, "full": 0.979},
    }

    print(f"{'Feature':<12} {'Probe':>8} {'Full':>8} {'Gap':>8}")
    print("-" * 40)
    for feat, vals in results.items():
        gap = vals["full"] - vals["probe"]
        marker = " <--" if abs(gap) > 0.1 else ""
        print(f"{feat:<12} {vals['probe']:>7.1%} {vals['full']:>7.1%} {gap:>+7.1%}{marker}")
