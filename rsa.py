"""
Representational Similarity Analysis (RSA)
-------------------------------------------
The core measurement of the experiment.

RSA asks: do two systems organize the same concepts in the same
relational structure, even if their representations live in
completely different spaces?

It works by comparing *pairwise similarity matrices*, not raw vectors.
This means you never need to align the spaces -- you only need to know
whether concept A is closer to concept B than to concept C, in each system.

This is why RSA is the right tool here:
    - WM latents: ~1536 dimensional
    - LLM hiddens: ~4096 dimensional
    - These can't be directly compared
    - But their 200x200 similarity matrices can be correlated directly

A high RSA score means: "when the world model thinks two concepts are
similar, the LLM also thinks they are similar" -- independent of scale,
rotation, or dimensionality.
"""

import numpy as np
from scipy import stats
from typing import Optional
import warnings


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarities between all vectors.
    Returns a symmetric N x N matrix with 1s on the diagonal.
    """
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    normalized = vectors / norms
    rsm = normalized @ normalized.T
    return rsm


def euclidean_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    from sklearn.metrics import pairwise_distances
    return pairwise_distances(vectors, metric="euclidean")


def rsa_score(rsm_a: np.ndarray, rsm_b: np.ndarray,
              method: str = "spearman") -> tuple[float, float]:
    """
    Compute RSA score between two representational similarity matrices.

    Extracts the upper triangle (excluding diagonal) of each matrix
    and computes correlation. Using upper triangle only because the
    matrix is symmetric -- no need to count each pair twice.

    Args:
        rsm_a: [N, N] similarity matrix from system A (world model)
        rsm_b: [N, N] similarity matrix from system B (LLM)
        method: 'spearman' (rank correlation, robust to outliers)
                'pearson'  (linear correlation, assumes normal distribution)

    Returns:
        (correlation, p_value)

    Interpretation:
        r > 0.5  -- strong alignment
        r > 0.3  -- moderate alignment
        r > 0.1  -- weak but significant (with N=200+)
        r < 0.1  -- essentially no alignment
    """
    N = rsm_a.shape[0]
    assert rsm_a.shape == rsm_b.shape == (N, N), "Matrices must be same size"

    # Extract upper triangle (i < j), excluding diagonal
    triu_idx = np.triu_indices(N, k=1)
    vec_a = rsm_a[triu_idx]
    vec_b = rsm_b[triu_idx]

    if method == "spearman":
        r, p = stats.spearmanr(vec_a, vec_b)
    elif method == "pearson":
        r, p = stats.pearsonr(vec_a, vec_b)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(r), float(p)


def rsa_by_category(rsm_a: np.ndarray, rsm_b: np.ndarray,
                    category_boundaries: dict,
                    method: str = "spearman") -> dict:
    """
    Compute RSA within each concept category separately.

    This is the most informative part of the analysis.
    If WM/LLM agree strongly on PHYSICAL but weakly on ABSTRACT,
    that tells you something fundamental about what embodiment adds.

    Args:
        rsm_a, rsm_b: full [N, N] matrices
        category_boundaries: dict of category -> (start, end) indices
        method: correlation method

    Returns:
        dict of category -> {'r': float, 'p': float, 'n_pairs': int}
    """
    results = {}

    for category, (start, end) in category_boundaries.items():
        # Extract the sub-block for this category
        sub_a = rsm_a[start:end, start:end]
        sub_b = rsm_b[start:end, start:end]

        n = end - start
        if n < 3:
            warnings.warn(f"Category {category} has only {n} concepts, skipping")
            continue

        r, p = rsa_score(sub_a, sub_b, method)
        n_pairs = n * (n - 1) // 2

        results[category] = {
            "r": r,
            "p": p,
            "n_concepts": n,
            "n_pairs": n_pairs,
            "significant": p < 0.05,
        }

    return results


def permutation_test(rsm_a: np.ndarray, rsm_b: np.ndarray,
                     n_permutations: int = 1000,
                     method: str = "spearman") -> dict:
    """
    Permutation test to assess significance more rigorously than p-value.

    Randomly shuffles the concept labels in rsm_b many times and
    recomputes RSA. The true RSA score should exceed the permuted
    distribution.

    Args:
        rsm_a, rsm_b: [N, N] matrices
        n_permutations: number of shuffles (1000 is standard)

    Returns:
        {
            'observed_r': float,
            'permuted_mean': float,
            'permuted_std': float,
            'p_value': float,  # fraction of permutations that exceeded observed
            'z_score': float,  # how many SDs above permuted mean
        }
    """
    N = rsm_a.shape[0]
    observed_r, _ = rsa_score(rsm_a, rsm_b, method)

    permuted_rs = []
    for _ in range(n_permutations):
        # Shuffle concept labels in rsm_b
        perm = np.random.permutation(N)
        rsm_b_shuffled = rsm_b[np.ix_(perm, perm)]
        r_perm, _ = rsa_score(rsm_a, rsm_b_shuffled, method)
        permuted_rs.append(r_perm)

    permuted_rs = np.array(permuted_rs)
    p_value = np.mean(permuted_rs >= observed_r)
    z_score = (observed_r - permuted_rs.mean()) / (permuted_rs.std() + 1e-8)

    return {
        "observed_r": observed_r,
        "permuted_mean": float(permuted_rs.mean()),
        "permuted_std": float(permuted_rs.std()),
        "p_value": float(p_value),
        "z_score": float(z_score),
        "n_permutations": n_permutations,
    }


def nearest_neighbor_agreement(rsm_a: np.ndarray, rsm_b: np.ndarray,
                                k: int = 5) -> float:
    """
    For each concept, find its k nearest neighbors in each system.
    Measure the overlap between the two neighbor sets.

    This is a more intuitive complement to RSA:
    "Do the two systems agree on which concepts are most similar?"

    Returns:
        Mean Jaccard overlap between top-k neighbor sets across all concepts
    """
    N = rsm_a.shape[0]
    overlaps = []

    for i in range(N):
        # Get top-k neighbors (excluding self) in each system
        row_a = rsm_a[i].copy()
        row_b = rsm_b[i].copy()
        row_a[i] = -np.inf
        row_b[i] = -np.inf

        neighbors_a = set(np.argsort(row_a)[-k:])
        neighbors_b = set(np.argsort(row_b)[-k:])

        # Jaccard overlap
        intersection = len(neighbors_a & neighbors_b)
        union = len(neighbors_a | neighbors_b)
        overlaps.append(intersection / union if union > 0 else 0)

    return float(np.mean(overlaps))


def run_full_analysis(rsm_wm: np.ndarray, rsm_lm: np.ndarray,
                      category_boundaries: dict,
                      concepts: list[str]) -> dict:
    """
    Run the complete RSA analysis pipeline.
    """
    print("\n" + "="*60)
    print("RSA ANALYSIS: World Model vs LLM")
    print("="*60)

    # 1. Overall RSA
    r_spearman, p_spearman = rsa_score(rsm_wm, rsm_lm, "spearman")
    r_pearson, p_pearson = rsa_score(rsm_wm, rsm_lm, "pearson")

    print(f"\n[ OVERALL ALIGNMENT ]")
    print(f"  Spearman r = {r_spearman:.4f}  (p = {p_spearman:.2e})")
    print(f"  Pearson  r = {r_pearson:.4f}  (p = {p_pearson:.2e})")

    # 2. By category
    category_results = rsa_by_category(rsm_wm, rsm_lm, category_boundaries)

    print(f"\n[ BY CATEGORY ]")
    print(f"  {'Category':12s} | {'r':>7s} | {'p':>10s} | {'sig':>5s} | pairs")
    print(f"  {'-'*12}-+-{'-'*7}-+-{'-'*10}-+-{'-'*5}-+------")
    for cat, res in category_results.items():
        sig = "✓" if res["significant"] else " "
        print(f"  {cat:12s} | {res['r']:>7.4f} | {res['p']:>10.2e} | "
              f"  {sig}   | {res['n_pairs']}")

    # 3. Nearest neighbor agreement
    nn_agreement = nearest_neighbor_agreement(rsm_wm, rsm_lm, k=5)
    print(f"\n[ NEAREST NEIGHBOR AGREEMENT (k=5) ]")
    print(f"  Mean Jaccard overlap: {nn_agreement:.4f}")
    print(f"  (Random baseline: {5/(len(concepts)-1):.4f})")

    # 4. Permutation test (subset for speed)
    print(f"\n[ PERMUTATION TEST (1000 permutations) ]")
    perm_results = permutation_test(rsm_wm, rsm_lm, n_permutations=1000)
    print(f"  Observed r:      {perm_results['observed_r']:.4f}")
    print(f"  Permuted mean:   {perm_results['permuted_mean']:.4f} "
          f"± {perm_results['permuted_std']:.4f}")
    print(f"  Z-score:         {perm_results['z_score']:.2f}")
    print(f"  p (permutation): {perm_results['p_value']:.4f}")

    # 5. Interpretation
    print(f"\n[ INTERPRETATION ]")
    interpret(r_spearman, category_results, nn_agreement)

    return {
        "overall_spearman_r": r_spearman,
        "overall_spearman_p": p_spearman,
        "overall_pearson_r": r_pearson,
        "overall_pearson_p": p_pearson,
        "by_category": category_results,
        "nn_agreement": nn_agreement,
        "permutation_test": perm_results,
    }


def interpret(r: float, category_results: dict, nn_agreement: float):
    """Print human-readable interpretation of results."""

    if r > 0.5:
        verdict = "STRONG alignment -- the two systems share substantial representational structure."
        implication = "Shared codebook is well-motivated. Proceed to architecture phase."
    elif r > 0.3:
        verdict = "MODERATE alignment -- meaningful but partial overlap."
        implication = "Shared codebook possible but will be lossy. Characterize which categories drive agreement."
    elif r > 0.1:
        verdict = "WEAK alignment -- some structure but mostly divergent."
        implication = "Shared codebook would be a compression artifact, not a semantic protocol. Investigate why."
    else:
        verdict = "NO meaningful alignment detected."
        implication = "WM and LLM are representing concepts in fundamentally different ways. This is itself a publishable finding."

    print(f"  Overall: {verdict}")
    print(f"  Implication: {implication}")

    # Check for divergence pattern across categories
    if category_results:
        sorted_cats = sorted(category_results.items(), key=lambda x: x[1]["r"], reverse=True)
        best_cat, best_res = sorted_cats[0]
        worst_cat, worst_res = sorted_cats[-1]

        if best_res["r"] - worst_res["r"] > 0.2:
            print(f"\n  Notable divergence: '{best_cat}' aligns much better "
                  f"(r={best_res['r']:.3f}) than '{worst_cat}' (r={worst_res['r']:.3f})")
            print(f"  This suggests the two systems weight "
                  f"{'physical/embodied' if best_cat in ['physical', 'actions'] else 'abstract'} "
                  f"structure more similarly than other categories.")
