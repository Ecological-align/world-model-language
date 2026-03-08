"""
Visualization for RSA Experiment Results
------------------------------------------
Produces four key plots:

1. RSM heatmaps -- side-by-side similarity matrices with category blocks
2. RSM scatter -- upper triangle of WM vs LLM similarities
3. Category bar chart -- RSA r by concept category
4. MDS projection -- 2D visualization of concept structure in each space
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


CATEGORY_COLORS = {
    "physical": "#E74C3C",
    "actions":  "#E67E22",
    "spatial":  "#2ECC71",
    "social":   "#3498DB",
    "abstract": "#9B59B6",
}


def plot_rsm_heatmap(rsm: np.ndarray, title: str, ax,
                     category_boundaries: dict, concepts: list[str]):
    """Plot a single RSM as a heatmap with category block outlines."""
    N = rsm.shape[0]

    im = ax.imshow(rsm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Concept index", fontsize=10)
    ax.set_ylabel("Concept index", fontsize=10)

    # Draw category block boundaries
    for cat, (start, end) in category_boundaries.items():
        color = CATEGORY_COLORS.get(cat, "black")
        rect = Rectangle(
            (start - 0.5, start - 0.5),
            end - start, end - start,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        # Category label
        mid = (start + end) / 2
        ax.text(mid, -2, cat[:4], ha="center", va="bottom",
                fontsize=7, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.tick_params(labelsize=7)


def plot_rsm_scatter(rsm_wm: np.ndarray, rsm_lm: np.ndarray,
                     ax, r: float, concepts: list[str],
                     category_boundaries: dict):
    """Scatter plot of upper-triangle similarities: WM vs LLM."""
    N = rsm_wm.shape[0]
    triu_idx = np.triu_indices(N, k=1)

    wm_vals = rsm_wm[triu_idx]
    lm_vals = rsm_lm[triu_idx]

    # Color by which category the pair belongs to
    # Use the category of the first concept in the pair
    colors = []
    from concepts import CONCEPT_CATEGORIES
    for i, j in zip(*triu_idx):
        cat_i = CONCEPT_CATEGORIES.get(concepts[i], "abstract")
        colors.append(CATEGORY_COLORS.get(cat_i, "gray"))

    ax.scatter(wm_vals, lm_vals, c=colors, alpha=0.3, s=8, linewidths=0)

    # Trend line
    z = np.polyfit(wm_vals, lm_vals, 1)
    p = np.poly1d(z)
    x_line = np.linspace(wm_vals.min(), wm_vals.max(), 100)
    ax.plot(x_line, p(x_line), "k--", linewidth=1.5, alpha=0.7)

    ax.set_xlabel("WM pairwise similarity", fontsize=10)
    ax.set_ylabel("LLM pairwise similarity", fontsize=10)
    ax.set_title(f"Upper-triangle correlation (Spearman r = {r:.3f})",
                 fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Legend
    handles = [
        plt.scatter([], [], c=color, s=30, label=cat)
        for cat, color in CATEGORY_COLORS.items()
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left",
              framealpha=0.8, handletextpad=0.3)


def plot_category_bars(category_results: dict, ax):
    """Bar chart of RSA r by category."""
    cats = list(category_results.keys())
    rs = [category_results[c]["r"] for c in cats]
    ps = [category_results[c]["p"] for c in cats]
    colors = [CATEGORY_COLORS.get(c, "gray") for c in cats]

    bars = ax.bar(cats, rs, color=colors, alpha=0.85, edgecolor="white",
                  linewidth=1.2)

    # Significance stars
    for i, (r, p) in enumerate(zip(rs, ps)):
        if p < 0.001:
            star = "***"
        elif p < 0.01:
            star = "**"
        elif p < 0.05:
            star = "*"
        else:
            star = "ns"
        ax.text(i, r + 0.01, star, ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_ylabel("Spearman r", fontsize=10)
    ax.set_title("RSA by Concept Category", fontsize=12, fontweight="bold")
    ax.set_ylim(-0.3, 1.0)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8)

    # Reference lines
    for level, label in [(0.3, "moderate"), (0.5, "strong")]:
        ax.axhline(level, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.text(len(cats) - 0.5, level + 0.02, label,
                ha="right", va="bottom", fontsize=7, color="gray")


def plot_mds(wm_vectors: np.ndarray, lm_vectors: np.ndarray,
             concepts: list[str], ax_wm, ax_lm):
    """
    2D MDS projection of concept structure in each space.
    If representations are aligned, the shapes should look similar.
    """
    from concepts import CONCEPT_CATEGORIES

    for vectors, ax, title, marker in [
        (wm_vectors, ax_wm, "World Model (MDS)", "o"),
        (lm_vectors, ax_lm, "LLM (MDS)", "s"),
    ]:
        # Normalize + scale
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)

        # MDS -- use precomputed distances
        from sklearn.metrics import pairwise_distances
        dist_matrix = pairwise_distances(vectors_scaled, metric="cosine")
        dist_matrix = np.clip(dist_matrix, 0, 2)

        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, n_init=4, max_iter=300)
        coords = mds.fit_transform(dist_matrix)

        for concept, (x, y) in zip(concepts, coords):
            cat = CONCEPT_CATEGORIES.get(concept, "abstract")
            color = CATEGORY_COLORS.get(cat, "gray")
            ax.scatter(x, y, c=color, s=40, alpha=0.8,
                       marker=marker, linewidths=0)
            ax.annotate(concept, (x, y), fontsize=5, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points")

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


def plot_all(rsm_wm: np.ndarray, rsm_lm: np.ndarray,
             wm_vectors: np.ndarray, lm_vectors: np.ndarray,
             concepts: list[str], category_boundaries: dict,
             results: dict, save_path: str = "rsa_results.png"):
    """
    Master plot function -- produces the full 6-panel figure.
    """
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        wspace=0.35,
        hspace=0.45,
        left=0.06, right=0.97,
        top=0.93, bottom=0.07,
    )

    ax_rsm_wm   = fig.add_subplot(gs[0, 0])
    ax_rsm_lm   = fig.add_subplot(gs[0, 1])
    ax_scatter  = fig.add_subplot(gs[0, 2])
    ax_cats     = fig.add_subplot(gs[1, 0])
    ax_mds_wm   = fig.add_subplot(gs[1, 1])
    ax_mds_lm   = fig.add_subplot(gs[1, 2])

    r = results["overall_spearman_r"]

    # Panel 1 & 2: RSM heatmaps
    plot_rsm_heatmap(rsm_wm, "World Model RSM", ax_rsm_wm,
                     category_boundaries, concepts)
    plot_rsm_heatmap(rsm_lm, "LLM RSM", ax_rsm_lm,
                     category_boundaries, concepts)

    # Panel 3: Scatter
    plot_rsm_scatter(rsm_wm, rsm_lm, ax_scatter, r,
                     concepts, category_boundaries)

    # Panel 4: Category bars
    plot_category_bars(results["by_category"], ax_cats)

    # Panels 5 & 6: MDS
    plot_mds(wm_vectors, lm_vectors, concepts, ax_mds_wm, ax_mds_lm)

    # Title
    mode_str = "SIMULATION" if "simul" in save_path.lower() or True else "REAL MODELS"
    fig.suptitle(
        f"RSA Experiment: World Model vs LLM Representational Alignment\n"
        f"Overall Spearman r = {r:.4f}  |  "
        f"N = {len(concepts)} concepts  |  "
        f"NN agreement = {results['nn_agreement']:.3f}",
        fontsize=13, fontweight="bold", y=0.98
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved to: {save_path}")
    return fig
