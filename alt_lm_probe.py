"""
alt_lm_probe.py
===============

Compares the codebook alignment pattern across all available LLMs
against the same visual models (V-JEPA 2, MAE, CLIP).

Question: Is the LM↔MAE > LM↔VIS pattern specific to Mistral,
or does it replicate across architectures and scales?

If the pattern replicates → structural finding about language models
If Mistral-specific → model quirk, weaker claim

Expected output:
  ┌──────────────────────────────────────────────────────────┐
  │           LM↔VIS    LM↔MAE   LM↔CLIP   Rank             │
  ├──────────────────────────────────────────────────────────┤
  │ Mistral-7B  40.6%    48.8%    44.4%   MAE>CLIP>VIS  ✓   │
  │ Qwen2.5-7B  XX.X%    XX.X%    XX.X%   ???               │
  │ Llama3.1-8B XX.X%    XX.X%    XX.X%   ???               │
  │ Qwen2.5-32B XX.X%    XX.X%    XX.X%   ???               │
  └──────────────────────────────────────────────────────────┘

Replication criterion: MAE > CLIP > VIS for 3+ LLMs = strong evidence

Usage:
  python alt_lm_probe.py

Requires: extract_alt_lm.py to have been run first
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
RESULTS_FILE = "lm_output/alt_lm_probe_results.json"

# ── Codebook parameters ────────────────────────────────────────────────────────
EMBED_DIM  = 64
N_CODES    = 16
LAMBDA_CM  = 0.5
LAMBDA_DIV = 0.1
N_RUNS     = 50    # 10 splits × 5 seeds
N_SPLITS   = 10
N_SEEDS    = 5

# ── LLM variants to compare ───────────────────────────────────────────────────
LM_VARIANTS = {
    "mistral_7b": {
        "file":  "lm_hiddens_phrase.npy",
        "label": "Mistral-7B",
    },
    "qwen25_7b": {
        "file":  "qwen25_7b_hiddens_phrase.npy",
        "label": "Qwen2.5-7B",
    },
    "llama31_8b": {
        "file":  "llama31_8b_hiddens_phrase.npy",
        "label": "Llama-3.1-8B",
    },
    "qwen25_32b": {
        "file":  "qwen25_32b_hiddens_phrase.npy",
        "label": "Qwen2.5-32B",
    },
}

# ── Visual models ──────────────────────────────────────────────────────────────
VISUAL_MODELS = {
    "vis":          ("vjepa2_hiddens_phrase.npy",          "V-JEPA2"),
    "mae":          ("mae_hiddens_phrase.npy",             "MAE"),
    "clip":         ("clip_hiddens_phrase.npy",            "CLIP"),
    "videomae":     ("videomae_hiddens_phrase.npy",        "VideoMAE"),
    "videomae_ssv2":("videomae_ssv2_hiddens_phrase.npy",  "VideoMAE-SSv2"),
    "dinov2":       ("dinov2_hiddens_phrase.npy",          "DINOv2"),
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


def run_pair(lm_arr, vis_arr, label):
    train_agrs, test_agrs, actives = [], [], []
    seed = 0
    for _ in range(N_SEEDS):
        for _ in range(N_SPLITS):
            tr, te, ac = run_one(lm_arr, vis_arr, seed)
            train_agrs.append(tr)
            test_agrs.append(te)
            actives.append(ac)
            seed += 1

    mean_tr = np.mean(train_agrs) * 100
    std_tr  = np.std(train_agrs)  * 100
    mean_te = np.mean(test_agrs)  * 100
    mean_ac = np.mean(actives)
    print(f"    {label:<14} train={mean_tr:.1f}%±{std_tr:.1f}  "
          f"test={mean_te:.1f}%  codes={mean_ac:.1f}")
    return {"train": mean_tr, "train_std": std_tr, "test": mean_te, "codes": mean_ac}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("ALT LLM PROBE: Does LM↔MAE > LM↔VIS replicate across LLMs?")
    print("=" * 65)

    # Load visual models (shared across all LLM variants)
    vis_embs = {}
    for key, (fname, label) in VISUAL_MODELS.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            vis_embs[key] = (np.load(path), label)
            print(f"Loaded {label}: {vis_embs[key][0].shape}")
        else:
            print(f"WARNING: {path} not found — skipping {label}")

    if len(vis_embs) < 2:
        print("ERROR: Need at least 2 visual modalities.")
        sys.exit(1)

    # Load event index to get concept-level mean embeddings
    idx_path = os.path.join(OUTPUT_DIR, "event_index.json")
    with open(idx_path) as f:
        event_index = json.load(f)["events"]
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    N_concepts = len(concepts)
    print(f"\nConcepts: {N_concepts}  Events: {len(event_index)}")

    def concept_means(arr):
        """Average phrase embeddings per concept."""
        out = []
        for c in concepts:
            idxs = [i for i, e in enumerate(event_index) if e["concept"] == c]
            out.append(arr[idxs].mean(axis=0))
        return np.array(out)

    vis_concept = {k: (concept_means(arr), lbl) for k, (arr, lbl) in vis_embs.items()}

    # ── Run each LLM variant ──────────────────────────────────────────────────
    all_results  = {}
    summary_rows = []

    for lm_key, lm_cfg in LM_VARIANTS.items():
        lm_path = os.path.join(OUTPUT_DIR, lm_cfg["file"])
        if not os.path.exists(lm_path):
            print(f"\n[SKIP] {lm_cfg['label']}: {lm_path} not found")
            print(f"  Run: python extract_alt_lm.py --model {lm_key}")
            continue

        lm_raw     = np.load(lm_path)
        lm_concept = concept_means(lm_raw)
        print(f"\n{'─'*65}")
        print(f"LLM: {lm_cfg['label']}  shape={lm_raw.shape}")
        print(f"{'─'*65}")

        lm_results = {}
        for vis_key, (vis_arr, vis_lbl) in vis_concept.items():
            r = run_pair(lm_concept, vis_arr, f"LM↔{vis_lbl}")
            lm_results[f"LM↔{vis_lbl}"] = r

        all_results[lm_key] = lm_results

        # Determine ranking for this LLM
        scores  = {k: v["test"] for k, v in lm_results.items()}
        ranking = sorted(scores, key=scores.get, reverse=True)
        pattern_matches = (
            ranking[0] == "LM↔MAE" and ranking[-1] == "LM↔V-JEPA2"
        ) if len(ranking) >= 3 else None

        summary_rows.append({
            "lm":      lm_cfg["label"],
            "scores":  scores,
            "ranking": ranking,
            "matches_mistral_pattern": pattern_matches,
        })

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("SUMMARY: Does LM↔MAE > LM↔CLIP > LM↔VIS replicate?")
    print(f"{'='*65}")
    print(f"  {'LLM':<16} {'LM↔VIS':>8} {'LM↔MAE':>8} {'LM↔CLIP':>9}  Ranking         Match?")
    print(f"  {'-'*64}")

    replications = 0
    total        = 0
    for row in summary_rows:
        s   = row["scores"]
        vis  = s.get("LM↔V-JEPA2", s.get("LM↔VIS", float("nan")))
        mae  = s.get("LM↔MAE",     float("nan"))
        clip = s.get("LM↔CLIP",    float("nan"))
        m    = "✓" if row["matches_mistral_pattern"] else ("✗" if row["matches_mistral_pattern"] is False else "?")
        rank_str = " > ".join(r.replace("LM↔", "") for r in row["ranking"])
        print(f"  {row['lm']:<16} {vis:>7.1f}%  {mae:>7.1f}%  {clip:>8.1f}%  {rank_str:<18}  {m}")
        if row["matches_mistral_pattern"] is not None:
            total += 1
            if row["matches_mistral_pattern"]:
                replications += 1

    print(f"\n  Pattern replications: {replications}/{total} LLMs")
    if total > 0:
        if replications == total:
            print("  → STRONG EVIDENCE: pattern is structural, not Mistral-specific")
        elif replications >= total * 0.5:
            print("  → PARTIAL REPLICATION: holds in majority of LLMs")
        else:
            print("  → WEAK REPLICATION: pattern may be Mistral-specific")

    # Save results
    save_data = {
        "results":      all_results,
        "summary":      summary_rows,
        "replications": replications,
        "total":        total,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
