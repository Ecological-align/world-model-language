"""
Three diagnostics in one script:

A) CLIP text encoder baseline
   CLIP has a text encoder trained contrastively against its vision encoder.
   CLIP-text RSA with CLIP-image provides an upper bound: what does
   perfectly-paired image-text training achieve?
   We then compare Mistral and ST against this ceiling.

B) Bootstrap CIs on RSA values
   All reported RSA correlations get 95% CIs via 10,000 bootstrap resamples
   of the RDM upper triangle. Tests whether V-JEPA vs CLIP (r=+0.917)
   is significantly different from MAE vs CLIP (r=+0.723).

C) Embedding norm statistics
   Reports mean L2 norm and std per modality, and between-modality vs
   within-concept variance ratio — the geometric explanation for collapse.

Outputs: lm_output/clip_text_bootstrap_norms.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import json
import torch
from scipy.stats import spearmanr

# ── Load existing embeddings ──────────────────────────────────────────────────
print("Loading 49-concept embeddings...")
lm  = np.load("lm_output/lm_hiddens_expanded.npy")     # [49, 4096]
st  = np.load("lm_output/st_hiddens_expanded.npy")     # [49, 768]
cli = np.load("lm_output/clip_hiddens_expanded.npy")   # [49, 768]
vj  = np.load("lm_output/vjepa2_hiddens_expanded.npy") # [49, 1024]
mae = np.load("lm_output/mae_hiddens_expanded.npy")    # [49, 1024]

with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)
concepts = idx["all_concepts"]
N = len(concepts)

# ── RSA utilities ─────────────────────────────────────────────────────────────
def build_rdm(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs_n = vecs / norms
    sim = vecs_n @ vecs_n.T
    return 1.0 - sim

def spearman_rdm(rdm_a, rdm_b):
    n = rdm_a.shape[0]
    idx_u = np.triu_indices(n, k=1)
    r, p = spearmanr(rdm_a[idx_u], rdm_b[idx_u])
    return float(r), float(p)

def bootstrap_rdm_ci(rdm_a, rdm_b, n_boot=10000, ci=95):
    """Bootstrap 95% CI for Spearman RSA correlation."""
    n = rdm_a.shape[0]
    idx_u = np.triu_indices(n, k=1)
    a_vals = rdm_a[idx_u]
    b_vals = rdm_b[idx_u]
    rng = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_boot):
        boot_idx = rng.integers(0, len(a_vals), len(a_vals))
        r, _ = spearmanr(a_vals[boot_idx], b_vals[boot_idx])
        boot_rs.append(r)
    lo = np.percentile(boot_rs, (100 - ci) / 2)
    hi = np.percentile(boot_rs, 100 - (100 - ci) / 2)
    return float(lo), float(hi)

def bootstrap_diff_ci(rdm_ref, rdm_a, rdm_b, n_boot=10000):
    """Bootstrap CI for difference r(ref,a) - r(ref,b). Tests if gap is significant."""
    n = rdm_ref.shape[0]
    idx_u = np.triu_indices(n, k=1)
    ref_v = rdm_ref[idx_u]
    a_v   = rdm_a[idx_u]
    b_v   = rdm_b[idx_u]
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        bi = rng.integers(0, len(ref_v), len(ref_v))
        ra, _ = spearmanr(ref_v[bi], a_v[bi])
        rb, _ = spearmanr(ref_v[bi], b_v[bi])
        diffs.append(ra - rb)
    lo = np.percentile(diffs, 2.5)
    hi = np.percentile(diffs, 97.5)
    p_val = np.mean(np.array(diffs) <= 0)  # one-sided: P(diff <= 0)
    return float(lo), float(hi), float(p_val)

# ══════════════════════════════════════════════════════════════════════════════
# A) CLIP TEXT ENCODER BASELINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("A) CLIP TEXT ENCODER BASELINE")
print("=" * 65)

try:
    from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Loading CLIP ViT-L/14...")
    model_name = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    clip_model.eval()
    clip_tok = CLIPTokenizer.from_pretrained(model_name)

    print("Extracting CLIP text embeddings for 49 concepts...")
    clip_text_vecs = []
    with torch.no_grad():
        for concept in concepts:
            inputs = clip_tok(
                [f"a photo of {concept}", concept],
                return_tensors="pt", padding=True, truncation=True
            ).to(device)
            # transformers 5.x: get_text_features returns object, not tensor
            text_out = clip_model.text_model(**inputs)
            pooled = text_out.pooler_output  # [2, hidden]
            projected = clip_model.text_projection(pooled)  # [2, 768]
            vec = projected.mean(0).cpu().numpy()
            clip_text_vecs.append(vec)

    clip_text = np.array(clip_text_vecs)  # [49, 768]
    print(f"CLIP text embeddings: {clip_text.shape}")

    clip_text_extracted = True
except Exception as e:
    print(f"CLIP text extraction failed: {e}")
    print("Skipping CLIP text baseline.")
    clip_text = None
    clip_text_extracted = False

# ══════════════════════════════════════════════════════════════════════════════
# B) BOOTSTRAP CIs ON RSA VALUES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("B) BOOTSTRAP 95% CIs ON RSA (10,000 resamples)")
print("=" * 65)

rdm_lm  = build_rdm(lm)
rdm_st  = build_rdm(st)
rdm_cli = build_rdm(cli)
rdm_vj  = build_rdm(vj)
rdm_mae = build_rdm(mae)

pairs = [
    ("V-JEPA 2 vs CLIP", rdm_vj,  rdm_cli),
    ("MAE vs CLIP",       rdm_mae, rdm_cli),
    ("Mistral vs CLIP",   rdm_lm,  rdm_cli),
    ("ST vs CLIP",        rdm_st,  rdm_cli),
    ("Mistral vs ST",     rdm_lm,  rdm_st),
    ("V-JEPA 2 vs MAE",   rdm_vj,  rdm_mae),
]

print(f"\n  {'Pair':<28} {'r':>7}  {'95% CI':>20}  {'p':>8}")
print("  " + "-" * 70)

bootstrap_results = {}
for name, a, b in pairs:
    r, p = spearman_rdm(a, b)
    lo, hi = bootstrap_rdm_ci(a, b)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:<28} {r:>+.3f}  [{lo:>+.3f}, {hi:>+.3f}]  {p:>8.4f} {sig}")
    bootstrap_results[name] = {"r": r, "p": p, "ci_lo": lo, "ci_hi": hi}

# Test if V-JEPA - MAE gap is significant
print("\n  Testing V-JEPA vs CLIP - MAE vs CLIP gap:")
lo_diff, hi_diff, p_diff = bootstrap_diff_ci(rdm_cli, rdm_vj, rdm_mae)
gap = bootstrap_results["V-JEPA 2 vs CLIP"]["r"] - bootstrap_results["MAE vs CLIP"]["r"]
print(f"  Gap = {gap:+.3f}, 95% CI [{lo_diff:+.3f}, {hi_diff:+.3f}], p(gap≤0) = {p_diff:.4f}")
if p_diff < 0.05:
    print("  → V-JEPA 2 significantly outperforms MAE (one-sided p < 0.05)")
else:
    print("  → Gap is not statistically significant at p < 0.05")

# CLIP text comparisons (if extracted)
clip_text_results = {}
if clip_text_extracted:
    rdm_clip_text = build_rdm(clip_text)
    print("\n  CLIP text encoder comparisons:")
    ct_pairs = [
        ("CLIP-image vs CLIP-text (ceiling)", rdm_cli, rdm_clip_text),
        ("V-JEPA 2 vs CLIP-text",             rdm_vj,  rdm_clip_text),
        ("MAE vs CLIP-text",                  rdm_mae, rdm_clip_text),
        ("Mistral vs CLIP-text",              rdm_lm,  rdm_clip_text),
        ("ST vs CLIP-text",                   rdm_st,  rdm_clip_text),
    ]
    print(f"\n  {'Pair':<40} {'r':>7}  {'95% CI':>20}  {'p':>8}")
    print("  " + "-" * 80)
    for name, a, b in ct_pairs:
        r, p = spearman_rdm(a, b)
        lo, hi = bootstrap_rdm_ci(a, b)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {name:<40} {r:>+.3f}  [{lo:>+.3f}, {hi:>+.3f}]  {p:>8.4f} {sig}")
        clip_text_results[name] = {"r": r, "p": p, "ci_lo": lo, "ci_hi": hi}

# ══════════════════════════════════════════════════════════════════════════════
# C) EMBEDDING NORM STATISTICS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("C) EMBEDDING NORM STATISTICS")
print("=" * 65)

modalities = {
    "Mistral": lm,
    "ST":      st,
    "CLIP":    cli,
    "V-JEPA2": vj,
    "MAE":     mae,
}

print(f"\n  {'Model':<12} {'Mean L2':>10} {'Std L2':>10} {'Min':>8} {'Max':>8}")
print("  " + "-" * 55)

norm_stats = {}
all_vecs = {}
for name, vecs in modalities.items():
    norms = np.linalg.norm(vecs, axis=1)
    print(f"  {name:<12} {norms.mean():>10.2f} {norms.std():>10.2f} "
          f"{norms.min():>8.2f} {norms.max():>8.2f}")
    norm_stats[name] = {
        "mean": float(norms.mean()), "std": float(norms.std()),
        "min": float(norms.min()), "max": float(norms.max())
    }
    # Normalize for geometry analysis
    all_vecs[name] = vecs / (norms[:, None] + 1e-8)

# Between-modality vs within-concept variance
# For each concept pair of modalities: how much does geometry differ?
print("\n  Between-modality variance (mean pairwise RDM distance):")
print(f"  {'Pair':<28} {'Mean |rdm_A - rdm_B|':>22}")
print("  " + "-" * 55)

geometry_stats = {}
model_names = list(modalities.keys())
for i, na in enumerate(model_names):
    for j, nb in enumerate(model_names):
        if j <= i: continue
        rdm_a = build_rdm(modalities[na])
        rdm_b = build_rdm(modalities[nb])
        n = rdm_a.shape[0]
        idx_u = np.triu_indices(n, k=1)
        diff = float(np.mean(np.abs(rdm_a[idx_u] - rdm_b[idx_u])))
        print(f"  {na} vs {nb:<20} {diff:>22.4f}")
        geometry_stats[f"{na}_vs_{nb}"] = diff

# Within-concept variance: how much does concept X vary across its representations?
# For ST vs VJ: compute per-concept cosine distance
print("\n  Within-concept cross-modal RDM distance (ST vs V-JEPA 2):")
# ST and VJ have different dims (768 vs 1024), so use RDM-based comparison
rdm_st_n = build_rdm(all_vecs["ST"])
rdm_vj_n = build_rdm(all_vecs["V-JEPA2"])
idx_u = np.triu_indices(rdm_st_n.shape[0], k=1)
per_concept_dist = np.abs(rdm_st_n[idx_u] - rdm_vj_n[idx_u])
print(f"  Mean RDM element distance: {per_concept_dist.mean():.4f}")
print(f"  Std:                       {per_concept_dist.std():.4f}")

# Between-modality: mean RDM distance (already computed above)
between_rdm = float(np.mean(per_concept_dist))
print(f"  (This is the mean absolute RDM difference across all concept pairs)")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "clip_text_extracted": clip_text_extracted,
    "bootstrap_rsa_49concepts": bootstrap_results,
    "vjepa_mae_gap": {"gap": gap, "ci_lo": lo_diff, "ci_hi": hi_diff, "p": p_diff},
    "clip_text_rsa": clip_text_results,
    "norm_statistics": norm_stats,
    "geometry_stats": geometry_stats,
    "rdm_distance_st_vj": {
        "mean": float(per_concept_dist.mean()),
        "std": float(per_concept_dist.std()),
    }
}
with open("lm_output/clip_text_bootstrap_norms.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/clip_text_bootstrap_norms.json")
