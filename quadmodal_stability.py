"""
quadmodal_stability.py
======================

Runs the 4-modality experiment across 6 different hyperparameter configs
to verify the agreement matrix pattern is stable.

Key question: does LM↔MAE > LM↔VIS and VIS↔anything remain lowest
across different lambda_cm, lambda_div, codebook_dim, and epoch settings?

Uses 5 seeds per config (not 50) for speed. ~25 min total.

Usage:
    python quadmodal_stability.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from itertools import combinations

DATA_DIR    = "lm_output/phrase_level"
OUTPUT_FILE = "lm_output/phrase_level/quadmodal_stability_results.json"

MODALITIES = ["LM", "VIS", "CLIP", "MAE"]
PAIRS = list(combinations(MODALITIES, 2))
N_TEST_CONCEPTS = 10
SEEDS = [42, 123, 7, 99, 2025]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Configs to sweep ──────────────────────────────────────────────────────────
# Vary: lambda_cm, lambda_div, codebook_dim, epochs
# Original was: lambda_cm=0.04, lambda_div=0.5, dim=256, epochs=400
CONFIGS = [
    # label,        lam_cm, lam_div, dim,  epochs
    ("original",    0.04,   0.5,    256,   400),   # baseline — should match prior run
    ("low_cm",      0.01,   0.5,    256,   400),   # less cross-modal pressure
    ("high_div",    0.04,   1.5,    256,   400),   # more diversity pressure
    ("large_cb",    0.04,   0.5,    512,   400),   # bigger codebook space
    ("more_epochs", 0.04,   0.5,    256,   800),   # longer training
    ("balanced",    0.02,   1.0,    256,   600),   # intermediate everything
]

# ── Load embeddings ────────────────────────────────────────────────────────────
print("Loading embeddings...")
lm_h    = np.load(os.path.join(DATA_DIR, "lm_hiddens_phrase.npy"))
vjepa_h = np.load(os.path.join(DATA_DIR, "vjepa2_hiddens_phrase.npy"))
clipt_h = np.load(os.path.join(DATA_DIR, "clip_text_hiddens_phrase.npy"))
mae_h   = np.load(os.path.join(DATA_DIR, "mae_hiddens_phrase.npy"))

with open(os.path.join(DATA_DIR, "event_index.json")) as f:
    event_index = json.load(f)
events   = event_index["events"]
concepts = event_index["concepts"]
N        = len(events)
event_concepts = [e["concept"] for e in events]

DIMS = {"LM": lm_h.shape[1], "VIS": vjepa_h.shape[1],
        "CLIP": clipt_h.shape[1], "MAE": mae_h.shape[1]}
print(f"  {N} events, {len(concepts)} concepts\n")

# ── Architecture ──────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    def __init__(self, num_codes, dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_codes, 1/num_codes)
    def forward(self, z):
        dists = (z.pow(2).sum(1, keepdim=True)
                 - 2 * z @ self.embeddings.weight.T
                 + self.embeddings.weight.pow(2).sum(1))
        idx = dists.argmin(1)
        q   = self.embeddings(idx)
        return z + (q - z).detach(), idx, F.mse_loss(z, q.detach())

class ModalityEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x): return self.net(x)

class QuadModal(nn.Module):
    def __init__(self, dims, cb_dim, num_codes=64):
        super().__init__()
        self.encoders = nn.ModuleDict({m: ModalityEncoder(dims[m], cb_dim) for m in MODALITIES})
        self.decoders = nn.ModuleDict({m: nn.Linear(cb_dim, dims[m])       for m in MODALITIES})
        self.cb = VQCodebook(num_codes, cb_dim)
    def forward(self, inputs):
        out = {}
        for m, x in inputs.items():
            z = self.encoders[m](x)
            q, idx, c = self.cb(z)
            out[f"{m}_z"] = z; out[f"{m}_idx"] = idx; out[f"{m}_commit"] = c
            out[f"{m}_recon"] = self.decoders[m](q)
        return out

def nt_xent(z1, z2, T=0.1):
    B = z1.shape[0]
    z = torch.cat([F.normalize(z1,dim=1), F.normalize(z2,dim=1)], 0)
    sim = (z @ z.T / T).fill_diagonal_(float('-inf'))
    labels = torch.cat([torch.arange(B,2*B,device=z.device), torch.arange(0,B,device=z.device)])
    return F.cross_entropy(sim, labels)

def div_loss(z1, z2, labels):
    z1n, z2n = F.normalize(z1,dim=1), F.normalize(z2,dim=1)
    B = z1.shape[0]
    same = (labels.unsqueeze(0)==labels.unsqueeze(1)) & ~torch.eye(B,dtype=torch.bool,device=z1.device)
    same = same.triu(1)
    if not same.any(): return torch.tensor(0., device=z1.device)
    return ((z1n@z1n.T)[same] + (z2n@z2n.T)[same]).mean()/2

def to_t(arr): return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

def make_split(seed):
    rng = random.Random(seed * 1000)
    test_c = set(rng.sample(concepts, N_TEST_CONCEPTS))
    tr, te = zip(*[(i, e["concept"] in test_c) for i, e in enumerate(events)])
    train_rows = [i for i, t in zip(tr, te) if not t]
    test_rows  = [i for i, t in zip(tr, te) if t]
    return list(range(len(events))), train_rows, test_rows  # use all rows, but track split

def run_one(seed, lam_cm, lam_div, cb_dim, epochs):
    torch.manual_seed(seed); random.seed(seed)

    rng = random.Random(seed * 1000)
    test_c = set(rng.sample(concepts, N_TEST_CONCEPTS))
    train_rows = [i for i, e in enumerate(events) if e["concept"] not in test_c]
    test_rows  = [i for i, e in enumerate(events) if e["concept"] in test_c]

    train_concepts = [event_concepts[r] for r in train_rows]
    uc = list(set(train_concepts))
    cmap = {c: i for i, c in enumerate(uc)}
    labels_t = torch.tensor([cmap[c] for c in train_concepts], device=DEVICE)

    tr_in = {
        "LM": to_t(lm_h[train_rows]), "VIS": to_t(vjepa_h[train_rows]),
        "CLIP": to_t(clipt_h[train_rows]), "MAE": to_t(mae_h[train_rows]),
    }

    model = QuadModal(DIMS, cb_dim).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(epochs):
        model.train()
        out = model(tr_in)
        rec    = sum(F.mse_loss(out[f"{m}_recon"], tr_in[m]) for m in MODALITIES)
        commit = sum(out[f"{m}_commit"] for m in MODALITIES)
        cm     = sum(nt_xent(out[f"{a}_z"], out[f"{b}_z"]) for a, b in PAIRS)
        div    = div_loss(out["LM_z"], out["VIS_z"], labels_t)
        loss   = rec + 0.25*commit + lam_cm*cm + lam_div*div
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out = model(tr_in)
        agree = {f"{a}_{b}": (out[f"{a}_idx"]==out[f"{b}_idx"]).float().mean().item()
                 for a, b in PAIRS}
        active = {m: len(set(out[f"{m}_idx"].cpu().tolist())) for m in MODALITIES}
    return agree, active

# ── Main ──────────────────────────────────────────────────────────────────────

all_config_results = {}

# Header
print(f"{'Config':<14} {'LM↔VIS':>7} {'LM↔CLIP':>8} {'LM↔MAE':>7} "
      f"{'VIS↔CLIP':>9} {'VIS↔MAE':>8} {'CLIP↔MAE':>9}  Pattern")
print("-" * 90)

for label, lam_cm, lam_div, cb_dim, epochs in CONFIGS:
    seed_results = []
    for seed in SEEDS:
        agree, active = run_one(seed, lam_cm, lam_div, cb_dim, epochs)
        seed_results.append({"agree": agree, "active": active})

    means = {k: float(np.mean([r["agree"][k] for r in seed_results])) for k in seed_results[0]["agree"]}
    stds  = {k: float(np.std( [r["agree"][k] for r in seed_results])) for k in seed_results[0]["agree"]}

    # Check the pattern: is LM↔MAE the top or near-top pair? Is VIS the outlier?
    sorted_pairs = sorted(means.items(), key=lambda x: -x[1])
    top_pair     = sorted_pairs[0][0]
    bottom_pair  = sorted_pairs[-1][0]
    vis_is_outlier = all(means[k] < 0.5 for k in means if "VIS" in k and k != "LM_VIS")
    lm_mae_high    = means["LM_MAE"] >= means["LM_VIS"]

    pattern = "✓ stable" if (lm_mae_high and "VIS" in bottom_pair) else "✗ differs"

    print(f"  {label:<12} "
          f"{means['LM_VIS']*100:>6.1f}% "
          f"{means['LM_CLIP']*100:>7.1f}% "
          f"{means['LM_MAE']*100:>6.1f}% "
          f"{means['VIS_CLIP']*100:>8.1f}% "
          f"{means['VIS_MAE']*100:>7.1f}% "
          f"{means['CLIP_MAE']*100:>8.1f}%  {pattern}")

    all_config_results[label] = {
        "hyperparameters": {"lambda_cm": lam_cm, "lambda_div": lam_div,
                            "codebook_dim": cb_dim, "epochs": epochs},
        "means": means, "stds": stds,
        "top_pair": top_pair, "bottom_pair": bottom_pair,
        "pattern_stable": bool(lm_mae_high and "VIS" in bottom_pair),
    }

# ── Stability summary ─────────────────────────────────────────────────────────
print("\n" + "="*70)
stable_count = sum(1 for v in all_config_results.values() if v["pattern_stable"])
print(f"Pattern stable in {stable_count}/{len(CONFIGS)} configs")

print("\nRanking of pairs (averaged across all configs):")
pair_keys = [f"{a}_{b}" for a, b in PAIRS]
grand_means = {k: np.mean([v["means"][k] for v in all_config_results.values()]) for k in pair_keys}
for k, v in sorted(grand_means.items(), key=lambda x: -x[1]):
    a, b = k.split("_")
    stab = np.std([v2["means"][k] for v2 in all_config_results.values()])
    print(f"  {a}↔{b}: {v*100:.1f}% (std across configs: {stab*100:.1f}%)")

print("\nConclusion:")
lm_mae_rank = sorted(grand_means.keys(), key=lambda k: -grand_means[k]).index("LM_MAE") + 1
vis_clip_rank = sorted(grand_means.keys(), key=lambda k: -grand_means[k]).index("VIS_CLIP") + 1
if lm_mae_rank <= 2 and vis_clip_rank >= 5:
    print("  STABLE: LM↔MAE consistently high, VIS↔CLIP consistently low.")
    print("  The agreement matrix pattern is robust to hyperparameter variation.")
    print("  V-JEPA 2 (temporal prediction) is structurally the most distinct model.")
else:
    print(f"  MIXED: LM↔MAE ranks #{lm_mae_rank}, VIS↔CLIP ranks #{vis_clip_rank}.")
    print("  Pattern has some sensitivity to hyperparameters — interpret with care.")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({
        "grand_means": grand_means,
        "configs": all_config_results,
    }, f, indent=2)
print(f"\nSaved → {OUTPUT_FILE}")
