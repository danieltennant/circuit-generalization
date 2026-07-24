"""
Geometry analysis: understand *how* the country feature is represented
non-linearly in layer-2 activation space.

Steps:
  1. UMAP visualisation coloured by each feature
  2. Per-dimension mutual information with country
  3. Dimension histograms for top-MI dims
  4. Pairwise scatter plots of top-MI dimensions (country colouring)
  5. Superposition: joint food × country 4-group colouring
  6. LDA projection onto shared food/country axis → band structure
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif
from umap import UMAP

# ---------------------------------------------------------------------------
# Assumed inputs (replace with actual loading)
# ---------------------------------------------------------------------------
# acts   : np.ndarray (N, 64) — post-ReLU layer-2 activations
# y      : np.ndarray (N, 8)  — binary labels per feature
# FEATURES = ["number","question","color","food","sentiment",
#              "country","person","body_part"]

FEATURES = ["number", "question", "color", "food", "sentiment",
            "country", "person", "body_part"]
FOOD_IDX    = FEATURES.index("food")
COUNTRY_IDX = FEATURES.index("country")
N_TOP_DIMS  = 16   # dims to highlight in histogram grid


# ---------------------------------------------------------------------------
# 1. UMAP
# ---------------------------------------------------------------------------

def plot_umap(acts, y, save_path="umap_country.png"):
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding = reducer.fit_transform(acts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("UMAP of layer-2 activations (train set)")

    # Left: colour by country
    country = y[:, COUNTRY_IDX].astype(bool)
    axes[0].scatter(embedding[~country, 0], embedding[~country, 1],
                    c="lightgrey", s=4, label="no country")
    axes[0].scatter(embedding[country, 0], embedding[country, 1],
                    c="tab:red", s=4, label="country")
    axes[0].set_title("Coloured by: country")
    axes[0].legend(markerscale=3, fontsize=8)

    # Right: colour by all features overlaid
    colors = plt.cm.tab10.colors
    for i, feat in enumerate(FEATURES):
        mask = y[:, i].astype(bool)
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1],
                        c=[colors[i]], s=4, label=feat, alpha=0.6)
    axes[1].set_title("All features overlaid")
    axes[1].legend(markerscale=3, fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# 2 & 3. Mutual information + dimension histograms
# ---------------------------------------------------------------------------

def plot_dim_histograms(acts, y, save_path="dim_histograms.png"):
    country = y[:, COUNTRY_IDX]
    mi = mutual_info_classif(acts, country, discrete_features=False,
                              random_state=42)
    top_dims = np.argsort(mi)[::-1][:N_TOP_DIMS]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(f"Per-dimension distributions (top {N_TOP_DIMS} by MI with 'country')")

    for ax, dim in zip(axes.flat, top_dims):
        vals_no  = acts[country == 0, dim]
        vals_yes = acts[country == 1, dim]
        bins = np.linspace(acts[:, dim].min(), acts[:, dim].max(), 40)
        ax.hist(vals_no,  bins=bins, color="lightgrey", alpha=0.8, label="no country", density=False)
        ax.hist(vals_yes, bins=bins, color="tab:red",   alpha=0.6, label="country",    density=False)
        ax.set_title(f"dim {dim}  MI={mi[dim]:.3f}", fontsize=9)
        ax.tick_params(labelsize=7)

    handles = [
        plt.Rectangle((0,0),1,1, color="lightgrey", alpha=0.8),
        plt.Rectangle((0,0),1,1, color="tab:red",   alpha=0.6),
    ]
    fig.legend(handles, ["no country", "country"], loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")
    return top_dims, mi


# ---------------------------------------------------------------------------
# 4. Pairwise scatter plots (country colouring)
# ---------------------------------------------------------------------------

def plot_dim_pairs(acts, y, top_dims, save_path="dim_pairs.png"):
    country = y[:, COUNTRY_IDX].astype(bool)
    pairs = [(top_dims[i], top_dims[j])
             for i in range(len(top_dims))
             for j in range(i+1, len(top_dims))][:10]

    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("Scatter plots of top-MI dimension-pairs  —  red=country")

    for ax, (d1, d2) in zip(axes.flat, pairs):
        ax.scatter(acts[~country, d1], acts[~country, d2],
                   c="lightgrey", s=1, alpha=0.3)
        ax.scatter(acts[country,  d1], acts[country,  d2],
                   c="tab:red",   s=1, alpha=0.5)
        ax.set_xlabel(f"dim {d1}", fontsize=8)
        ax.set_ylabel(f"dim {d2}", fontsize=8)
        ax.set_title(f"dims {d1} vs {d2}", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# 5. Superposition: joint food × country 4-group colouring
# ---------------------------------------------------------------------------

def plot_superposition(acts, y, top_dims, save_path="superposition.png"):
    food    = y[:, FOOD_IDX].astype(bool)
    country = y[:, COUNTRY_IDX].astype(bool)

    groups = {
        "neither":      (~food) & (~country),
        "food only":    food    & (~country),
        "country only": (~food) & country,
        "both":         food    & country,
    }
    group_colors = {
        "neither":      "lightgrey",
        "food only":    "tab:blue",
        "country only": "tab:red",
        "both":         "tab:cyan",
    }

    # Key pairs that revealed structure
    pairs = [
        (top_dims[0], top_dims[2]),   # dim 10 vs 44
        (top_dims[0], top_dims[1]),   # dim 10 vs 46
        (top_dims[1], top_dims[2]),   # dim 46 vs 44
        (top_dims[0], top_dims[3]),   # dim 10 vs 34
        (top_dims[1], top_dims[3]),   # dim 46 vs 34
        (top_dims[2], top_dims[3]),   # dim 44 vs 34
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Layer-2 activations coloured by food/country label combinations")

    for ax, (d1, d2) in zip(axes.flat, pairs):
        for label, mask in groups.items():
            ax.scatter(acts[mask, d1], acts[mask, d2],
                       c=group_colors[label], s=2, alpha=0.4, label=label)
        ax.set_xlabel(f"dim {d1}", fontsize=8)
        ax.set_ylabel(f"dim {d2}", fontsize=8)
        ax.set_title(f"dims {d1} vs {d2}", fontsize=8)
        ax.tick_params(labelsize=7)

    handles = [plt.Line2D([0],[0], marker='o', color='w',
                           markerfacecolor=c, markersize=8, label=l)
               for l, c in group_colors.items()]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# 6. LDA projection → band structure
# ---------------------------------------------------------------------------

def plot_band_structure(acts, y, save_path="band_structure.png"):
    food    = y[:, FOOD_IDX].astype(bool)
    country = y[:, COUNTRY_IDX].astype(bool)

    group_labels = np.zeros(len(acts), dtype=int)
    group_labels[food    & ~country] = 1   # food only
    group_labels[~food   &  country] = 2   # country only
    group_labels[food    &  country] = 3   # both

    lda = LinearDiscriminantAnalysis(n_components=1)
    projection = lda.fit_transform(acts, group_labels).ravel()

    group_names  = ["neither", "food only", "country only", "both"]
    group_colors = ["lightgrey", "tab:blue", "tab:red", "tab:cyan"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Band structure: country occupies the MIDDLE of the shared food/country axis")

    # Density histograms
    for i, (name, color) in enumerate(zip(group_names, group_colors)):
        mask = group_labels == i
        med = np.median(projection[mask])
        ax1.hist(projection[mask], bins=60, color=color, alpha=0.6,
                 label=name, density=False)
        ax1.axvline(med, color=color, linestyle="--", linewidth=1.2)
    ax1.set_xlabel("Projection onto shared food/country direction")
    ax1.set_ylabel("Density")
    ax1.set_title("1D projection: four groups along shared direction")
    ax1.legend(fontsize=9)

    # Box plot
    data_by_group = [projection[group_labels == i] for i in range(4)]
    bp = ax2.boxplot(data_by_group, tick_labels=group_names, notch=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.axhline(0, linestyle=":", color="black", linewidth=0.8)
    ax2.set_ylabel("Projection value")
    ax2.set_title("Distribution by group (notched box plot)")

    # Print medians
    print("\nMedian projections by group:")
    for i, name in enumerate(group_names):
        print(f"  {name:<15} {np.median(projection[group_labels==i]):.4f}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_mock_data(n=8500, dim=64, seed=42):
    """
    Generate synthetic activations and labels for smoke-testing the plots.
    The country feature is given a non-linear (bandpass) structure so the
    band-structure plot looks meaningful. Replace with real puzzle data to
    reproduce the original results.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, len(FEATURES))).astype(np.float32)

    # Build activations with a shared food/country direction in dim 10
    acts = np.abs(rng.standard_normal((n, dim))).astype(np.float32)  # post-ReLU → non-neg
    food    = y[:, FOOD_IDX].astype(bool)
    country = y[:, COUNTRY_IDX].astype(bool)

    acts[~food & ~country, 10] += 0.0
    acts[food  & ~country, 10] += 0.8   # food-only: highest
    acts[~food &  country, 10] += 0.3   # country-only: middle
    acts[food  &  country, 10] += 0.55  # both: between food and country-only

    return acts, y


if __name__ == "__main__":
    import sys
    mock = "--mock" in sys.argv

    if mock:
        print("Running with mock data (smoke test only — plots will not match originals)")
        acts, y = make_mock_data()
    else:
        # Load acts and y from the puzzle model here:
        # acts = ...   # np.ndarray (N, 64)
        # y    = ...   # np.ndarray (N, 8)
        raise RuntimeError("Pass --mock to run with synthetic data, or load real puzzle activations above.")

    top_dims, mi = plot_dim_histograms(acts, y)
    print(f"\nTop dims by MI with country: {top_dims[:8]}")

    plot_umap(acts, y)
    plot_dim_pairs(acts, y, top_dims)
    plot_superposition(acts, y, top_dims)
    plot_band_structure(acts, y)
