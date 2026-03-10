"""
architecture_control_probe.py
==============================

Architecture-controlled comparison: all ViT-Base models (768-dim, ~86M params).

EXPERIMENTAL DESIGN
===================
This probe isolates training data + objective from architecture, by
holding architecture fixed at ViT-Base across all three models:

  Model             | Arch      | Data   | Objective       |
  ------------------|-----------|--------|-----------------|
  MAE-Base          | ViT-Base  | Image  | Reconstruction  |  <- NEW
  VideoMAE-Base/K400| ViT-Base  | Video  | Reconstruction  |
  VideoMAE-Base/SSv2| ViT-Base  | Video  | Reconstruction  |

All three use identical architecture. The only difference is training data.
If MAE-Base aligns HIGH while VideoMAE variants align LOW:
  → Architecture is not the confound. Image data × reconstruction is the driver.

If MAE-Base aligns LOW (similar to VideoMAE):
  → Architecture/capacity was driving MAE-Large's high alignment.

Also runs the full model comparison (all 6 visual models) with bootstrap CIs
to provide statistical uncertainty estimates.

Usage:
  python architecture_control_probe.py

Requires:
  - mae_base_hiddens_phrase.npy       (run extract_mae_base.py)
  - videomae_hiddens_phrase.npy       (already exists)
  - videomae_ssv2_hiddens_phrase.npy  (already exists)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUTPUT_DIR   = "lm_output/phrase_level"
RESULTS_FILE = "lm_output/architecture_control_results.json"

EMBED_DIM  = 64
N_CODES    = 16
LAMBDA_CM  = 0.5
LAMBDA_DIV = 0.1
N_SPLITS   = 10
N_SEEDS    = 5

# Use Mistral as the reference LLM for the architecture control
# (also run all 4 for the bootstrap CI section)
LM_VARIANTS = {
    "mistral_7b":  {"file": "lm_hiddens_phrase.npy",        "label": "Mistral-7B"},
    "qwen25_7b":   {"file": "qwen25_7b_hiddens_phrase.npy",  "label": "Qwen2.5-7B"},
    "llama31_8b":  {"file": "llama31_8b_hiddens_phrase.npy", "label": "Llama-3.1-8B"},
    "qwen25_32b":  {"file": "qwen25_32b_hiddens_phrase.npy", "label": "Qwen2.5-32B"},
}

# Architecture-controlled set (all ViT-Base)
ARCH_CONTROL_MODELS = {
    "mae_base":      ("mae_base_hiddens_phrase.npy",         "MAE-Base",       "image", "reconstruction"),
    "vmae_k400":     ("videomae_hiddens_phrase.npy",          "VideoMAE-K400",  "video", "reconstruction"),
    "vmae_ssv2":     ("videomae_ssv2_hiddens_phrase.npy",     "VideoMAE-SSv2",  "video", "reconstruction"),
}

# Full model set (for bootstrap CIs)
ALL_VISUAL_MODELS = {
    "mae":           ("mae_hiddens_phrase.npy",              "MAE-Large",      "image", "reconstruction"),
    "mae_base":      ("mae_base_hiddens_phrase.npy",         "MAE-Base",       "image", "reconstruction"),
    "dinov2":        ("dinov2_hiddens_phrase.npy",            "DINOv2",         "image", "distillation"),
    "vmae_k400":     ("videomae_hiddens_phrase.npy",          "VideoMAE-K400",  "video", "reconstruction"),
    "vmae_ssv2":     ("videomae_ssv2_hiddens_phrase.npy",     "VideoMAE-SSv2",  "video", "reconstruction"),
    "vis":           ("vjepa2_hiddens_phrase.npy",            "V-JEPA2",        "video", "temporal_pred"),
    "clip":          ("clip_hiddens_phrase.npy",              "CLIP",           "image", "contrastive"),
}


# ── VQ Codebook ────────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    def __init__(self, dim_a, dim_b, embed_dim, n_codes):
        super().__init__()
        self.proj_a   = nn.Linear(dim_a, embed_dim)
        self.proj_b   = nn.Linear(dim_b, embed_dim)
        self.codebook = nn.Embedding(n_codes, embed_dim)
        nn.init.orthogonal_(self.codebook.weight)

    def quantize(self, z):
        d   = torch.cdist(z, self.codebook.weight)
        idx = d.argmin(dim=-1)
        return idx, self.codebook(idx)

    def forward(self, a, b):
        za = F.normalize(self.proj_a(a), dim=-1)
        zb = F.normalize(self.proj_b(b), dim=-1)
        idx_a, qa = self.quantize(za)
        idx_b, qb = self.quantize(zb)

        rec = (F.mse_loss(qa, za.detach()) + F.mse_loss(qb, zb.detach()) +
               0.25 * (F.mse_loss(za, qa.detach()) + F.mse_loss(zb, qb.detach())))

        sim    = torch.mm(za, zb.T) / 0.07
        labels = torch.arange(len(za), device=a.device)
        cm     = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

        avg_a = F.softmax(-torch.cdist(za, self.codebook.weight) * 5, dim=-1).mean(0)
        avg_b = F.softmax(-torch.cdist(zb, self.codebook.weight) * 5, dim=-1).mean(0)
        div   = (-(avg_a * (avg_a + 1e-8).log()).sum()
                 - (avg_b * (avg_b + 1e-8).log()).sum()) / 2

        loss      = rec + LAMBDA_CM * cm - LAMBDA_DIV * div
        agreement = (idx_a == idx_b).float().mean().item()
        active    = len(idx_a.unique())
        return loss, agreement, active


def run_one(a_arr, b_arr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    N     = len(a_arr)
    perm  = np.random.permutation(N)
    split = int(0.8 * N)
    tr_i, te_i = perm[:split], perm[split:]

    a_tr = torch.tensor(a_arr[tr_i], dtype=torch.float32)
    b_tr = torch.tensor(b_arr[tr_i], dtype=torch.float32)
    a_te = torch.tensor(a_arr[te_i], dtype=torch.float32)
    b_te = torch.tensor(b_arr[te_i], dtype=torch.float32)

    model = VQCodebook(a_arr.shape[1], b_arr.shape[1], EMBED_DIM, N_CODES)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(300):
        model.train()
        opt.zero_grad()
        loss, _, _ = model(a_tr, b_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, tr_agr, active = model(a_tr, b_tr)
        _, te_agr, _      = model(a_te, b_te)
    return tr_agr, te_agr, active


def run_pair_full(lm_arr, vis_arr, label, n_runs=None):
    """Run all seeds/splits, return full distribution of test agreements."""
    test_agrs = []
    seed = 0
    runs = n_runs or (N_SEEDS * N_SPLITS)
    for _ in range(runs):
        _, te, _ = run_one(lm_arr, vis_arr, seed)
        test_agrs.append(te * 100)
        seed += 1

    arr    = np.array(test_agrs)
    mean   = arr.mean()
    std    = arr.std()
    # Bootstrap 95% CI
    boot   = np.array([np.random.choice(arr, len(arr), replace=True).mean()
                       for _ in range(2000)])
    ci_lo  = np.percentile(boot, 2.5)
    ci_hi  = np.percentile(boot, 97.5)
    print(f"    {label:<22}  {mean:.1f}% ± {std:.1f}  95% CI [{ci_lo:.1f}, {ci_hi:.1f}]")
    return {"mean": mean, "std": std, "ci_lo": ci_lo, "ci_hi": ci_hi, "runs": list(arr)}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("ARCHITECTURE CONTROL PROBE + BOOTSTRAP CIs")
    print("=" * 72)

    # Load event index
    with open(os.path.join(OUTPUT_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    concepts    = list(dict.fromkeys(e["concept"] for e in event_index))
    print(f"Concepts: {len(concepts)}  Events: {len(event_index)}\n")

    def concept_means(arr):
        out = []
        for c in concepts:
            idxs = [i for i, e in enumerate(event_index) if e["concept"] == c]
            out.append(arr[idxs].mean(axis=0))
        return np.array(out)

    # Load all visual models
    all_vis = {}
    for key, (fname, label, dtype, obj) in ALL_VISUAL_MODELS.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            arr = np.load(path)
            all_vis[key] = (concept_means(arr), label, dtype, obj)
            print(f"  Loaded {label:<22} shape={arr.shape}  [{dtype}, {obj}]")
        else:
            print(f"  [SKIP] {label:<22} not found")

    if "mae_base" not in all_vis:
        print("\nERROR: mae_base_hiddens_phrase.npy not found.")
        print("Run extract_mae_base.py first.")
        return

    print()

    # ── Part 1: Architecture-Controlled Comparison ────────────────────────────
    print("=" * 72)
    print("PART 1 — ARCHITECTURE CONTROL (all ViT-Base, 768-dim)")
    print("Tests whether architecture drives MAE-Large's high alignment")
    print("=" * 72)

    all_results  = {}
    arch_summary = {}

    for lm_key, lm_cfg in LM_VARIANTS.items():
        lm_path = os.path.join(OUTPUT_DIR, lm_cfg["file"])
        if not os.path.exists(lm_path):
            continue
        lm_arr     = np.load(lm_path)
        lm_concept = concept_means(lm_arr)

        print(f"\n  LLM: {lm_cfg['label']}")
        print(f"  {'─'*60}")
        lm_res = {}
        for key in ["mae_base", "vmae_k400", "vmae_ssv2"]:
            if key not in all_vis:
                continue
            vis_arr, vis_lbl, *_ = all_vis[key]
            r = run_pair_full(lm_concept, vis_arr, f"LM↔{vis_lbl}")
            lm_res[vis_lbl] = r
        all_results[lm_key] = lm_res
        arch_summary[lm_cfg["label"]] = lm_res

    # Architecture control summary
    print(f"\n{'='*72}")
    print("ARCHITECTURE CONTROL SUMMARY (all ViT-Base)")
    print(f"{'='*72}")
    print(f"  {'LLM':<16}  {'LM↔MAE-Base':>13}  {'LM↔VMae-K400':>14}  {'LM↔VMae-SSv2':>14}  Verdict")
    print(f"  {'-'*68}")

    arch_confirmed = 0
    arch_total     = 0
    for lm_label, res in arch_summary.items():
        mae_v  = res.get("MAE-Base",      {}).get("mean", float("nan"))
        vmk_v  = res.get("VideoMAE-K400", {}).get("mean", float("nan"))
        vms_v  = res.get("VideoMAE-SSv2", {}).get("mean", float("nan"))
        mae_ci = res.get("MAE-Base",      {})
        vmk_ci = res.get("VideoMAE-K400", {})

        # Check if MAE-Base CI is above VideoMAE-K400 CI (non-overlapping)
        ci_separated = mae_ci.get("ci_lo", 0) > vmk_ci.get("ci_hi", float("inf"))
        verdict = "✓ MAE-Base HIGH, VMae LOW" if mae_v > vmk_v + 3 else "? SIMILAR"
        print(f"  {lm_label:<16}  {mae_v:>12.1f}%  {vmk_v:>13.1f}%  {vms_v:>13.1f}%  {verdict}")
        arch_total += 1
        if mae_v > vmk_v + 3:
            arch_confirmed += 1

    print(f"\n  Architecture confound ruled out: {arch_confirmed}/{arch_total} LLMs")
    if arch_confirmed == arch_total:
        print("  → Architecture is NOT driving the result.")
        print("    MAE-Base (image, recon) > VideoMAE-Base (video, recon)")
        print("    Same architecture, same objective — only training data differs.")
    else:
        print("  → Partial: architecture may be contributing to MAE-Large advantage.")

    # ── Part 2: Bootstrap CIs on Full Model Set ───────────────────────────────
    print(f"\n{'='*72}")
    print("PART 2 — BOOTSTRAP 95% CIs (full model set, Mistral-7B)")
    print("Provides statistical uncertainty for all 6 visual model comparisons")
    print(f"{'='*72}")

    lm_path = os.path.join(OUTPUT_DIR, "lm_hiddens_phrase.npy")
    if not os.path.exists(lm_path):
        print("Mistral embeddings not found, skipping.")
    else:
        lm_arr     = np.load(lm_path)
        lm_concept = concept_means(lm_arr)

        print(f"\n  LLM: Mistral-7B")
        print(f"  {'─'*60}")

        order   = ["mae", "mae_base", "dinov2", "vmae_k400", "vmae_ssv2", "vis", "clip"]
        ci_res  = {}
        for key in order:
            if key not in all_vis:
                continue
            vis_arr, vis_lbl, *_ = all_vis[key]
            r = run_pair_full(lm_concept, vis_arr, f"LM↔{vis_lbl}")
            ci_res[vis_lbl] = r

        # Check for non-overlapping CIs between key pairs
        print(f"\n{'='*72}")
        print("KEY PAIRWISE CI COMPARISONS (Mistral-7B)")
        print(f"{'='*72}")

        def ci_compare(lbl_a, lbl_b, question):
            if lbl_a not in ci_res or lbl_b not in ci_res:
                return
            a = ci_res[lbl_a]
            b = ci_res[lbl_b]
            overlap = not (a["ci_lo"] > b["ci_hi"] or b["ci_lo"] > a["ci_hi"])
            sig = "NON-OVERLAPPING (significant)" if not overlap else "OVERLAPPING (not significant)"
            print(f"\n  {question}")
            print(f"    {lbl_a}: {a['mean']:.1f}% [{a['ci_lo']:.1f}, {a['ci_hi']:.1f}]")
            print(f"    {lbl_b}: {b['mean']:.1f}% [{b['ci_lo']:.1f}, {b['ci_hi']:.1f}]")
            print(f"    → CIs are {sig}")

        ci_compare("MAE-Large",  "DINOv2",         "Obj isolated (image data fixed): MAE-Large vs DINOv2")
        ci_compare("MAE-Base",   "VideoMAE-K400",  "Data isolated (recon fixed, ViT-Base): MAE-Base vs VideoMAE-K400")
        ci_compare("MAE-Large",  "VideoMAE-K400",  "Obj+Data: MAE-Large vs VideoMAE-K400")
        ci_compare("MAE-Large",  "V-JEPA2",        "Full gap: MAE-Large vs V-JEPA2")
        ci_compare("DINOv2",     "VideoMAE-K400",  "Image non-recon vs Video recon: DINOv2 vs VideoMAE")
        ci_compare("VideoMAE-K400", "V-JEPA2",     "Reconstruction vs temporal pred (video): VMae vs V-JEPA2")

        all_results["bootstrap_ci_mistral"] = ci_res

    # Save all results
    save_data = {
        "architecture_control": all_results,
        "arch_confirmed": arch_confirmed,
        "arch_total": arch_total,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
