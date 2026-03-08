"""
Three-way RSA: Mistral 7B vs sentence-transformers vs CLIP (visual)
--------------------------------------------------------------------
Analysis is restricted to the intersection of concepts where all three
models have valid (non-zero) embeddings — in practice, the 33 concepts
where CLIP found Wikipedia images.

The absence of CLIP signal for abstract/spatial/action concepts is
itself a finding reported in the coverage section, not a problem to fix.

Output:
    lm_output/rsa_comparison.txt  -- full results
    lm_output/rsa_comparison.json -- machine-readable
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
from pathlib import Path
from scipy import stats

from extract_lm_standalone import ALL_CONCEPTS, CONCEPTS, CONCEPT_CATEGORIES
from rsa import cosine_similarity_matrix, rsa_score, permutation_test, nearest_neighbor_agreement


# ── Load embeddings ──────────────────────────────────────────────────────────

output_dir = Path("lm_output")

lm   = np.load(output_dir / "lm_hiddens.npy")    # [71, 4096] Mistral layer 16
st   = np.load(output_dir / "st_hiddens.npy")    # [71, 768]  all-mpnet-base-v2
clip = np.load(output_dir / "clip_hiddens.npy")  # [71, 768]  CLIP ViT-L/14

# ── Find valid intersection (non-zero CLIP embeddings) ───────────────────────

clip_norms = np.linalg.norm(clip, axis=-1)
valid_mask = clip_norms > 1e-8
valid_idx  = np.where(valid_mask)[0]
valid_concepts = [ALL_CONCEPTS[i] for i in valid_idx]
N_valid = len(valid_concepts)

# ── Coverage report ──────────────────────────────────────────────────────────

lines = []
lines.append("=" * 65)
lines.append("THREE-WAY RSA: Mistral 7B vs ST (all-mpnet) vs CLIP (visual)")
lines.append("=" * 65)

lines.append(f"\nValid concept intersection (CLIP coverage): {N_valid}/71")
lines.append("\nCoverage by category:")
lines.append(f"  {'Category':12s} | {'CLIP':>9s} | {'ST':>9s} | {'Mistral':>9s} | Valid concepts")
lines.append(f"  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+---------------")

category_valid_idx = {}  # category -> indices into valid_concepts list
for cat, items in CONCEPTS.items():
    total = len(items)
    cat_valid = [c for c in valid_concepts if CONCEPT_CATEGORIES[c] == cat]
    n_valid = len(cat_valid)
    category_valid_idx[cat] = [valid_concepts.index(c) for c in cat_valid]
    lines.append(
        f"  {cat:12s} | {n_valid:4d}/{total:<4d} | {total:4d}/{total:<4d} | {total:4d}/{total:<4d} | "
        f"{', '.join(cat_valid) if cat_valid else '(none)'}"
    )

lines.append(f"\n  Note: ST and Mistral have valid embeddings for all 71 concepts.")
lines.append(f"  CLIP coverage reflects whether a canonical visual exists on Wikipedia.")
lines.append(f"  Missing CLIP signal for action/spatial/abstract concepts is a finding.")

print("\n".join(lines))
lines_so_far = list(lines)

# ── Subset to valid concepts ──────────────────────────────────────────────────

lm_v   = lm[valid_idx]
st_v   = st[valid_idx]
clip_v = clip[valid_idx]

# ── Compute RSMs ─────────────────────────────────────────────────────────────

rsm_lm   = cosine_similarity_matrix(lm_v)
rsm_st   = cosine_similarity_matrix(st_v)
rsm_clip = cosine_similarity_matrix(clip_v)

# ── Pairwise RSA ─────────────────────────────────────────────────────────────

PAIRS = [
    ("Mistral", "ST",   rsm_lm,   rsm_st),
    ("Mistral", "CLIP", rsm_lm,   rsm_clip),
    ("ST",      "CLIP", rsm_st,   rsm_clip),
]

result_lines = []
result_lines.append(f"\n{'='*65}")
result_lines.append(f"PAIRWISE RSA ON {N_valid}-CONCEPT VALID SUBSET")
result_lines.append(f"{'='*65}")

all_results = {}

for name_a, name_b, rsm_a, rsm_b in PAIRS:
    r_sp, p_sp = rsa_score(rsm_a, rsm_b, "spearman")
    r_pe, p_pe = rsa_score(rsm_a, rsm_b, "pearson")
    nn = nearest_neighbor_agreement(rsm_a, rsm_b, k=5)

    pair_key = f"{name_a}_vs_{name_b}"
    all_results[pair_key] = {
        "spearman_r": r_sp, "spearman_p": p_sp,
        "pearson_r":  r_pe, "pearson_p":  p_pe,
        "nn_agreement_k5": nn,
        "by_category": {},
    }

    result_lines.append(f"\n[ {name_a} vs {name_b} ]")
    result_lines.append(f"  Overall  Spearman r = {r_sp:+.4f}  (p = {p_sp:.2e})")
    result_lines.append(f"  Overall  Pearson  r = {r_pe:+.4f}  (p = {p_pe:.2e})")
    result_lines.append(f"  Nearest-neighbor agreement (k=5): {nn:.4f}")
    result_lines.append(f"  (Random baseline: {5/(N_valid-1):.4f})")

    # Per-category RSA (using non-contiguous index selection)
    result_lines.append(f"\n  Per-category breakdown:")
    result_lines.append(f"    {'Category':12s} | {'n':>3s} | {'Spearman r':>10s} | {'p':>10s} | sig")
    result_lines.append(f"    {'-'*12}-+-----+-{'-'*10}-+-{'-'*10}-+----")

    for cat, idx_in_valid in category_valid_idx.items():
        n = len(idx_in_valid)
        if n < 3:
            result_lines.append(f"    {cat:12s} | {n:3d} | {'(n<3, skip)':>10s} |            |")
            all_results[pair_key]["by_category"][cat] = {"n": n, "skipped": True}
            continue
        sub_a = rsm_a[np.ix_(idx_in_valid, idx_in_valid)]
        sub_b = rsm_b[np.ix_(idx_in_valid, idx_in_valid)]
        r_cat, p_cat = rsa_score(sub_a, sub_b, "spearman")
        sig = "✓" if p_cat < 0.05 else " "
        result_lines.append(
            f"    {cat:12s} | {n:3d} | {r_cat:+10.4f} | {p_cat:10.2e} | {sig}"
        )
        all_results[pair_key]["by_category"][cat] = {
            "n": n, "spearman_r": r_cat, "spearman_p": p_cat, "significant": p_cat < 0.05
        }

print("\n".join(result_lines))

# ── Permutation tests ────────────────────────────────────────────────────────

perm_lines = []
perm_lines.append(f"\n{'='*65}")
perm_lines.append("PERMUTATION TESTS (1000 permutations each)")
perm_lines.append(f"{'='*65}")

for name_a, name_b, rsm_a, rsm_b in PAIRS:
    perm = permutation_test(rsm_a, rsm_b, n_permutations=1000)
    pair_key = f"{name_a}_vs_{name_b}"
    all_results[pair_key]["permutation_test"] = perm
    perm_lines.append(
        f"\n  {name_a} vs {name_b}:"
        f"  observed r={perm['observed_r']:+.4f}"
        f"  permuted={perm['permuted_mean']:+.4f}±{perm['permuted_std']:.4f}"
        f"  z={perm['z_score']:+.2f}"
        f"  p={perm['p_value']:.4f}"
    )

print("\n".join(perm_lines))

# ── Interpretation ───────────────────────────────────────────────────────────

interp_lines = []
interp_lines.append(f"\n{'='*65}")
interp_lines.append("INTERPRETATION")
interp_lines.append(f"{'='*65}")

ms_r  = all_results["Mistral_vs_ST"]["spearman_r"]
mc_r  = all_results["Mistral_vs_CLIP"]["spearman_r"]
sc_r  = all_results["ST_vs_CLIP"]["spearman_r"]

interp_lines.append(f"\n  Mistral vs ST:   r = {ms_r:+.4f}  (both text-based)")
interp_lines.append(f"  Mistral vs CLIP: r = {mc_r:+.4f}  (text vs visual)")
interp_lines.append(f"  ST     vs CLIP:  r = {sc_r:+.4f}  (text vs visual)")

text_text = ms_r
text_vis  = max(mc_r, sc_r)
gap = text_text - text_vis

if gap > 0.15:
    interp_lines.append(
        f"\n  Text-text agreement ({text_text:.3f}) >> text-visual ({text_vis:.3f})."
        f"\n  Gap = {gap:.3f}: text models share structure visual CLIP does not."
        f"\n  Physical concepts drive CLIP alignment; relational/action concepts diverge."
    )
elif gap < -0.05:
    interp_lines.append(
        f"\n  Text-visual agreement ({text_vis:.3f}) >= text-text ({text_text:.3f})."
        f"\n  Unexpected: visual structure aligns with LLM as well as text encoders."
    )
else:
    interp_lines.append(
        f"\n  Text-text and text-visual alignment are similar ({text_text:.3f} vs {text_vis:.3f})."
        f"\n  On this {N_valid}-concept subset, visual and text representations are comparably structured."
    )

interp_lines.append(
    f"\n  Coverage note: CLIP has 0% coverage of purely spatial/abstract concepts."
    f"\n  The RSA comparison is biased toward physical concepts where CLIP excels."
    f"\n  Interpret cross-modal scores with this selection bias in mind."
)

print("\n".join(interp_lines))

# ── Save ────────────────────────────────────────────────────────────────────

full_report = "\n".join(lines_so_far + result_lines + perm_lines + interp_lines)

with open(output_dir / "rsa_comparison.txt", "w", encoding="utf-8") as f:
    f.write(full_report)

# Serialise numpy floats for JSON
def to_python(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_python(v) for v in obj]
    return obj

with open(output_dir / "rsa_comparison.json", "w", encoding="utf-8") as f:
    json.dump(to_python(all_results), f, indent=2)

print(f"\nSaved to {output_dir}/rsa_comparison.txt and rsa_comparison.json")
