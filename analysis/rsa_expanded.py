"""
RSA on the expanded 49-concept set.

Re-runs the core RSA analysis from experiments 1-3 but using the
stable 49-concept embeddings instead of 17.

Outputs:
  - Full pairwise RSA table (all model pairs)
  - Physical-only results
  - Comparison with original 17-concept figures
  - Saved to lm_output/rsa_expanded_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import json
from scipy.stats import spearmanr
from itertools import combinations

# ── Load embeddings ──────────────────────────────────────────────────────────

print("Loading 49-concept embeddings...")
lm  = np.load("lm_output/lm_hiddens_expanded.npy")    # [49, 4096]
st  = np.load("lm_output/st_hiddens_expanded.npy")    # [49, 768]
cli = np.load("lm_output/clip_hiddens_expanded.npy")  # [49, 768]
vj  = np.load("lm_output/vjepa2_hiddens_expanded.npy")# [49, 1024]
mae = np.load("lm_output/mae_hiddens_expanded.npy")   # [49, 1024]

with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)

all_concepts = idx["all_concepts"]  # 49 concepts
orig_concepts = idx["original_concepts"]  # 17 original

print(f"Total concepts: {len(all_concepts)}")
print(f"Original concepts: {len(orig_concepts)}")

# ── RSA utilities ─────────────────────────────────────────────────────────────

def build_rdm(vecs):
    """Cosine distance RDM."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs_n = vecs / norms
    sim = vecs_n @ vecs_n.T
    return 1.0 - sim

def spearman_rdm(rdm_a, rdm_b):
    n = rdm_a.shape[0]
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(rdm_a[idx], rdm_b[idx])
    return float(r), float(p)

# ── Model pairs ───────────────────────────────────────────────────────────────

models = {
    "Mistral": lm,
    "ST": st,
    "CLIP": cli,
    "V-JEPA2": vj,
    "MAE": mae,
}

# Focus pairs (matching original experiments)
focus_pairs = [
    ("Mistral", "CLIP"),
    ("ST", "CLIP"),
    ("V-JEPA2", "CLIP"),
    ("MAE", "CLIP"),
    ("V-JEPA2", "ST"),
    ("Mistral", "V-JEPA2"),
    ("Mistral", "ST"),
]

# ── Physical-only indices ─────────────────────────────────────────────────────

orig_idx = [all_concepts.index(c) for c in orig_concepts]

# ── Run RSA: all 49 concepts ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("RSA RESULTS: ALL 49 CONCEPTS")
print("=" * 70)
print(f"{'Pair':<25} {'r':>8} {'p':>10} {'sig':>6}")
print("-" * 55)

results_49 = {}
for a, b in focus_pairs:
    rdm_a = build_rdm(models[a])
    rdm_b = build_rdm(models[b])
    r, p = spearman_rdm(rdm_a, rdm_b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {a} vs {b:<18} {r:>+.3f}  {p:>10.4f}  {sig}")
    results_49[f"{a}_vs_{b}"] = {"r": r, "p": p}

# ── Run RSA: original 17 concepts only (using expanded embeddings) ────────────

print("\n" + "=" * 70)
print("RSA RESULTS: ORIGINAL 17 CONCEPTS (from expanded embeddings)")
print("=" * 70)
print(f"{'Pair':<25} {'r':>8} {'p':>10} {'sig':>6}")
print("-" * 55)

results_17 = {}
for a, b in focus_pairs:
    rdm_a = build_rdm(models[a][orig_idx])
    rdm_b = build_rdm(models[b][orig_idx])
    r, p = spearman_rdm(rdm_a, rdm_b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {a} vs {b:<18} {r:>+.3f}  {p:>10.4f}  {sig}")
    results_17[f"{a}_vs_{b}"] = {"r": r, "p": p}

# ── Comparison table ──────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("COMPARISON: 17 concepts (original) vs 49 concepts (expanded)")
print("=" * 70)
print(f"{'Pair':<25} {'r_17':>8} {'r_49':>8} {'change':>8}")
print("-" * 55)

original_17_reported = {
    "Mistral_vs_CLIP":   0.020,   # physical-only from paper
    "ST_vs_CLIP":        None,
    "V-JEPA2_vs_CLIP":   0.404,
    "MAE_vs_CLIP":       0.087,
    "V-JEPA2_vs_ST":     None,
    "Mistral_vs_V-JEPA2": None,
    "Mistral_vs_ST":     None,
}

for a, b in focus_pairs:
    key = f"{a}_vs_{b}"
    r49 = results_49[key]["r"]
    r17 = results_17[key]["r"]
    orig = original_17_reported.get(key)
    orig_str = f"{orig:+.3f}" if orig is not None else "   n/a"
    change = r49 - r17
    print(f"  {a} vs {b:<18} {r17:>+.3f}  {r49:>+.3f}  {change:>+.3f}")

# ── Key finding callout ───────────────────────────────────────────────────────

vj_clip_49 = results_49["V-JEPA2_vs_CLIP"]["r"]
vj_clip_17 = results_17["V-JEPA2_vs_CLIP"]["r"]
mae_clip_49 = results_49["MAE_vs_CLIP"]["r"]
lm_clip_49 = results_49["Mistral_vs_CLIP"]["r"]

print("\n" + "=" * 70)
print("KEY FINDINGS SUMMARY")
print("=" * 70)
print(f"V-JEPA 2 vs CLIP (17 concepts): r = {vj_clip_17:+.3f}")
print(f"V-JEPA 2 vs CLIP (49 concepts): r = {vj_clip_49:+.3f}")
print(f"MAE vs CLIP      (49 concepts): r = {mae_clip_49:+.3f}")
print(f"Mistral vs CLIP  (49 concepts): r = {lm_clip_49:+.3f}")
print()

gap = vj_clip_49 - mae_clip_49
print(f"V-JEPA2 - MAE gap (49 concepts): {gap:+.3f}")
print()

if vj_clip_49 > 0.3 and results_49["V-JEPA2_vs_CLIP"]["p"] < 0.05:
    print("✓ V-JEPA 2 / CLIP alignment holds at 49 concepts")
else:
    print("! V-JEPA 2 / CLIP alignment weakens or loses significance at 49 concepts")

if mae_clip_49 < 0.1:
    print("✓ MAE / CLIP near-zero holds at 49 concepts")
else:
    print(f"! MAE / CLIP is {mae_clip_49:+.3f} at 49 concepts — different from original")

# ── Save ──────────────────────────────────────────────────────────────────────

output = {
    "rsa_49_concepts": results_49,
    "rsa_17_concepts_rerun": results_17,
    "n_concepts_49": 49,
    "n_concepts_17": len(orig_idx),
}
with open("lm_output/rsa_expanded_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/rsa_expanded_results.json")
