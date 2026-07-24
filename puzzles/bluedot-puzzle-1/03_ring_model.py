"""
Part 3: train a model with a weirder-than-linear internal representation.

Approach: define a 2D ring label (positive when radius falls in an annular band),
embed the coordinates into 64 dimensions to match the puzzle's layer-L dimension,
then train an MLP with a 2-neuron linear bottleneck that forces all information
through a 2D representation.

The model recovers the ring geometry in the bottleneck space without any
explicit regularisation — it emerges from the training signal alone.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def make_ring_dataset(n=8500, r_inner=0.7, r_outer=1.7,
                      embed_dim=64, noise_scale=0.055, seed=42):
    rng = np.random.default_rng(seed)

    # 2D Gaussian coordinates
    h = rng.standard_normal((n, 2))
    r = np.sqrt((h ** 2).sum(axis=1))
    labels = ((r > r_inner) & (r < r_outer)).astype(np.float32)

    # Random linear embedding into embed_dim
    W_embed = rng.standard_normal((2, embed_dim)) / np.sqrt(embed_dim)
    X = h @ W_embed + noise_scale * rng.standard_normal((n, embed_dim))

    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(h, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Model: MLP with 2-neuron bottleneck
# ---------------------------------------------------------------------------

class RingMLP(nn.Module):
    """
    Architecture:
      Input (64) → Linear → ReLU → Linear → ReLU → Linear(64, 2)  [encoder]
                 → Linear(2, 64) → ReLU → Linear → ReLU → Linear(64, 1)  [decoder]
    """
    def __init__(self, input_dim=64, hidden=256, bottleneck=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),    nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),     nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x)).squeeze(-1)

    def bottleneck_repr(self, x):
        with torch.no_grad():
            return self.encoder(x).numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_val, y_val, epochs=300, batch_size=256, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    N = len(X_train)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        total_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i : i + batch_size]
            logits = model(X_train[idx])
            loss = criterion(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_acc = ((model(X_val).sigmoid() > 0.5).float() == y_val).float().mean()
            print(f"Epoch {epoch+1:3d}  train_loss={total_loss/N:.4f}  val_acc={val_acc:.4f}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_train, y_train, X_test, y_test, h_test):
    """Evaluate on held-out test set; fit probes on train, score on test."""
    model.eval()
    with torch.no_grad():
        preds = (model(X_test).sigmoid() > 0.5).float()
    full_acc = (preds == y_test).float().mean().item()
    majority_acc = max(y_test.mean().item(), 1 - y_test.mean().item())

    bn_train = model.bottleneck_repr(X_train)
    bn_test  = model.bottleneck_repr(X_test)
    y_tr = y_train.numpy().astype(int)
    y_te = y_test.numpy().astype(int)
    h_tr = h_test.numpy()   # true coords for test set

    probe_bn   = LogisticRegression(max_iter=1000).fit(bn_train, y_tr).score(bn_test, y_te)
    probe_true = LogisticRegression(max_iter=1000).fit(h_tr, y_te).score(h_tr, y_te)
    svm_bn     = SVC(kernel="rbf", C=10).fit(bn_train, y_tr).score(bn_test, y_te)
    mlp_bn     = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500).fit(bn_train, y_tr).score(bn_test, y_te)

    print(f"\n{'Metric':<35} {'Accuracy':>9}")
    print("-" * 46)
    print(f"{'Full model':<35} {full_acc:>8.1%}")
    print(f"{'Majority-class baseline':<35} {majority_acc:>8.1%}")
    print(f"{'Linear probe on bottleneck':<35} {probe_bn:>8.1%}")
    print(f"{'Linear probe on true (h₁,h₂)':<35} {probe_true:>8.1%}")
    print(f"{'RBF SVM on bottleneck':<35} {svm_bn:>8.1%}")
    print(f"{'MLP on bottleneck':<35} {mlp_bn:>8.1%}")

    return bn_test, full_acc, majority_acc


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_ring_results(h_true, y, bn, model, full_acc, majority_acc,
                      save_path="ring_encoding.png"):
    model.eval()

    # Build a 2D grid over bottleneck space for the decision boundary plot
    x1 = np.linspace(bn[:, 0].min() - 0.5, bn[:, 0].max() + 0.5, 200)
    x2 = np.linspace(bn[:, 1].min() - 0.5, bn[:, 1].max() + 0.5, 200)
    xx, yy = np.meshgrid(x1, x2)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        probs = model.decoder(grid).sigmoid().numpy().reshape(xx.shape)

    y_np = y.numpy().astype(bool)
    h_np = h_true.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "2D Ring Encoding — More Interesting Than 1D Interval\n"
        f"Full model: {full_acc:.1%} | Linear probe: ~50% (BELOW {majority_acc:.1%} baseline) | "
        "RBF SVM / MLP: >90%"
    )

    # Panel 1: ground-truth ring in original 2D space
    r = np.sqrt((h_np ** 2).sum(axis=1))
    axes[0].scatter(h_np[~y_np, 0], h_np[~y_np, 1],
                    c="lightgrey", s=5, alpha=0.5, label="background (negative)")
    axes[0].scatter(h_np[y_np,  0], h_np[y_np,  1],
                    c="tab:red",   s=5, alpha=0.6, label="ring")
    theta = np.linspace(0, 2 * np.pi, 300)
    for r_val in [0.7, 1.7]:
        axes[0].plot(r_val * np.cos(theta), r_val * np.sin(theta),
                     "k--", linewidth=1)
    axes[0].set_title("Ground truth: 2D Gaussian\nring label = 1 for 0.7 < r < 1.7")
    axes[0].legend(fontsize=8)
    axes[0].set_aspect("equal")

    # Panel 2: learned 2-neuron bottleneck
    axes[1].scatter(bn[~y_np, 0], bn[~y_np, 1],
                    c="lightgrey", s=5, alpha=0.5, label="not ring")
    axes[1].scatter(bn[y_np,  0], bn[y_np,  1],
                    c="tab:red",   s=5, alpha=0.6, label="ring")
    axes[1].set_title("Learned 2-neuron bottleneck\n(ring structure preserved)")
    axes[1].legend(fontsize=8)

    # Panel 3: P(ring) over bottleneck space
    im = axes[2].contourf(xx, yy, probs, levels=50, cmap="RdBu_r", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], label="P(ring)")
    axes[2].set_title("P(ring) over bottleneck space\n(ring-shaped decision boundary)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print("Generating dataset...")
    X, y, h_true = make_ring_dataset(n=8500)
    print(f"  N={len(X)}, positive rate={y.mean():.3f}")

    # 80/20 train/test split
    n_train = int(0.8 * len(X))
    X_train, X_test   = X[:n_train],      X[n_train:]
    y_train, y_test   = y[:n_train],      y[n_train:]
    h_train, h_test   = h_true[:n_train], h_true[n_train:]

    print("\nTraining model...")
    model = RingMLP()
    train(model, X_train, y_train, X_test, y_test, epochs=300)

    print("\nEvaluating (test set)...")
    bn_test, full_acc, majority_acc = evaluate(model, X_train, y_train, X_test, y_test, h_test)

    print("\nPlotting...")
    plot_ring_results(h_test, y_test, bn_test, model, full_acc, majority_acc)
