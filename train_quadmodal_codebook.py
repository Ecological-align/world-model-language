"""
train_quadmodal_codebook.py
===========================

4-modality shared codebook:
  - Mistral      (LM)        — language model, language-supervised
  - V-JEPA 2     (VIS)       — video prediction, physics-supervised
  - CLIP-text    (CLIP)      — vision-language contrastive, language-supervised
  - MAE          (MAE)       — masked image autoencoder, self-supervised (no language)

Theory: the binary split is language-supervised vs physics-supervised.
  Predicted clustering: {LM, CLIP} ↔ {VIS, MAE}

If MAE groups with V-JEPA 2 → training objective drives the split, not modality.
If MAE groups with CLIP/LM   → something else is driving it.

Runs 10 splits × 5 seeds = 50 runs.

Usage:
    python train_quadmodal_codebook.py
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
OUTPUT_FILE = "lm_output/phrase_level/quadmodal_codebook_results.json"

CODEBOOK_DIM    = 256
NUM_CODES       = 64
EPOCHS          = 400
LR              = 1e-3
N_SPLITS        = 10
N_SEEDS         = 5
N_TEST_CONCEPTS = 10
TEMPERATURE     = 0.1

# 4 pairwise losses now, so reduce per-pair λ_cm further
LAMBDA_CM  = 0.04
LAMBDA_DIV = 0.5
LAMBDA_REC = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load ──────────────────────────────────────────────────────────────────────
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

MODALITIES = ["LM", "VIS", "CLIP", "MAE"]
DIMS = {
    "LM":   lm_h.shape[1],
    "VIS":  vjepa_h.shape[1],
    "CLIP": clipt_h.shape[1],
    "MAE":  mae_h.shape[1],
}
print(f"  LM:        {lm_h.shape}")
print(f"  V-JEPA 2:  {vjepa_h.shape}")
print(f"  CLIP-text: {clipt_h.shape}")
print(f"  MAE:       {mae_h.shape}")
print(f"  Events: {N}, Concepts: {len(concepts)}")
print(f"  Pairs: {list(combinations(MODALITIES, 2))}")

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
        q_st = z + (q - z).detach()
        return q_st, idx, F.mse_loss(z, q.detach())

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, codebook_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, codebook_dim), nn.LayerNorm(codebook_dim),
        )
    def forward(self, x): return self.net(x)

class QuadModalCodebook(nn.Module):
    def __init__(self, dims, codebook_dim, num_codes):
        super().__init__()
        self.encoders  = nn.ModuleDict({m: ModalityEncoder(dims[m], codebook_dim) for m in MODALITIES})
        self.decoders  = nn.ModuleDict({m: nn.Linear(codebook_dim, dims[m])       for m in MODALITIES})
        self.codebook  = VQCodebook(num_codes, codebook_dim)

    def forward(self, inputs):
        out = {}
        for m, x in inputs.items():
            z = self.encoders[m](x)
            q, idx, commit = self.codebook(z)
            out[f"{m}_z"]      = z
            out[f"{m}_q"]      = q
            out[f"{m}_idx"]    = idx
            out[f"{m}_commit"] = commit
            out[f"{m}_recon"]  = self.decoders[m](q)
        return out

# ── Losses ────────────────────────────────────────────────────────────────────

def nt_xent(z1, z2):
    B = z1.shape[0]
    z = torch.cat([F.normalize(z1, dim=1), F.normalize(z2, dim=1)], dim=0)
    sim = z @ z.T / TEMPERATURE
    sim.fill_diagonal_(float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(0, B,   device=z.device)])
    return F.cross_entropy(sim, labels)

def diversity_loss(z1, z2, concept_labels):
    z1n, z2n = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    B = z1.shape[0]
    same = (concept_labels.unsqueeze(0) == concept_labels.unsqueeze(1))
    same = same & ~torch.eye(B, dtype=torch.bool, device=z1.device)
    same = same.triu(diagonal=1)
    if not same.any():
        return torch.tensor(0.0, device=z1.device)
    return ((z1n @ z1n.T)[same] + (z2n @ z2n.T)[same]).mean() / 2

def compute_loss(out, inputs, concept_labels):
    pairs = list(combinations(MODALITIES, 2))

    rec    = sum(F.mse_loss(out[f"{m}_recon"], inputs[m]) for m in MODALITIES)
    commit = sum(out[f"{m}_commit"] for m in MODALITIES)
    cm     = sum(nt_xent(out[f"{a}_z"], out[f"{b}_z"]) for a, b in pairs)
    div    = diversity_loss(out["LM_z"], out["VIS_z"], concept_labels)

    return LAMBDA_REC * rec + 0.25 * commit + LAMBDA_CM * cm + LAMBDA_DIV * div

# ── Training ──────────────────────────────────────────────────────────────────

def to_t(arr): return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

def make_split(seed):
    rng = random.Random(seed * 1000)
    test_concepts = rng.sample(concepts, N_TEST_CONCEPTS)
    train_rows, test_rows = [], []
    for i, e in enumerate(events):
        (test_rows if e["concept"] in test_concepts else train_rows).append(i)
    return train_rows, test_rows

def run_once(seed, split_seed):
    torch.manual_seed(seed)
    random.seed(seed)

    train_rows, test_rows = make_split(split_seed)
    train_concepts = [event_concepts[r] for r in train_rows]
    unique_c = list(set(train_concepts))
    cmap     = {c: i for i, c in enumerate(unique_c)}
    labels_t = torch.tensor([cmap[c] for c in train_concepts], device=DEVICE)

    inputs_tr = {
        "LM":   to_t(lm_h[train_rows]),
        "VIS":  to_t(vjepa_h[train_rows]),
        "CLIP": to_t(clipt_h[train_rows]),
        "MAE":  to_t(mae_h[train_rows]),
    }

    model = QuadModalCodebook(DIMS, CODEBOOK_DIM, NUM_CODES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        out  = model(inputs_tr)
        loss = compute_loss(out, inputs_tr, labels_t)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        out_tr = model(inputs_tr)

        # All 6 pairwise agreements (train)
        pairs = list(combinations(MODALITIES, 2))
        train_agree = {
            f"{a}_{b}": (out_tr[f"{a}_idx"] == out_tr[f"{b}_idx"]).float().mean().item()
            for a, b in pairs
        }

        # Active codes per modality
        active = {m: len(set(out_tr[f"{m}_idx"].cpu().numpy().tolist())) for m in MODALITIES}

        # Sense diversity (LM)
        all_lm = out_tr["LM_idx"].cpu().numpy()
        divs = [len(set(all_lm[[i for i, c in enumerate(train_concepts) if c == concept]].tolist()))
                for concept in unique_c]
        mean_div = float(np.mean(divs))

        # Test agreements
        inputs_te = {
            "LM":   to_t(lm_h[test_rows]),
            "VIS":  to_t(vjepa_h[test_rows]),
            "CLIP": to_t(clipt_h[test_rows]),
            "MAE":  to_t(mae_h[test_rows]),
        }
        out_te = model(inputs_te)
        test_agree = {
            f"{a}_{b}": (out_te[f"{a}_idx"] == out_te[f"{b}_idx"]).float().mean().item()
            for a, b in pairs
        }

    result = {"mean_sense_diversity": mean_div}
    result.update({f"train_{k}": v for k, v in train_agree.items()})
    result.update({f"test_{k}":  v for k, v in test_agree.items()})
    result.update({f"active_{m}": active[m] for m in MODALITIES})
    return result

# ── Main ──────────────────────────────────────────────────────────────────────

all_results = []
total = N_SPLITS * N_SEEDS
done  = 0

print(f"\nRunning {N_SPLITS} splits × {N_SEEDS} seeds = {total} runs...")
print(f"\n{'Run':>4} | {'LM↔VIS':>7} {'LM↔CLIP':>8} {'LM↔MAE':>7} {'VIS↔CLIP':>9} {'VIS↔MAE':>8} {'CLIP↔MAE':>9} | Codes(LM) Div")
print("-" * 95)

for split_seed in range(N_SPLITS):
    for seed in range(N_SEEDS):
        r = run_once(seed, split_seed)
        all_results.append(r)
        done += 1
        if done % 5 == 0 or done == total:
            print(f"{done:>4} | "
                  f"{r['train_LM_VIS']*100:>6.1f}% "
                  f"{r['train_LM_CLIP']*100:>7.1f}% "
                  f"{r['train_LM_MAE']*100:>6.1f}% "
                  f"{r['train_VIS_CLIP']*100:>8.1f}% "
                  f"{r['train_VIS_MAE']*100:>7.1f}% "
                  f"{r['train_CLIP_MAE']*100:>8.1f}% | "
                  f"{r['active_LM']:>9} {r['mean_sense_diversity']:>3.1f}")

def m(key): return float(np.mean([r[key] for r in all_results]))
def s(key): return float(np.std( [r[key] for r in all_results]))

pairs = [("LM","VIS"), ("LM","CLIP"), ("LM","MAE"),
         ("VIS","CLIP"), ("VIS","MAE"), ("CLIP","MAE")]

print("\n" + "="*70)
print("PAIRWISE AGREEMENT MATRIX (train means)")
print("="*70)
print(f"{'':10}", end="")
for b in MODALITIES: print(f"{b:>10}", end="")
print()
for a in MODALITIES:
    print(f"{a:<10}", end="")
    for b in MODALITIES:
        if a == b:
            print(f"{'100.0%':>10}", end="")
        else:
            key = f"train_{a}_{b}" if f"train_{a}_{b}" in all_results[0] else f"train_{b}_{a}"
            print(f"{m(key)*100:>9.1f}%", end="")
    print()

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for a, b in pairs:
    key = f"train_{a}_{b}"
    print(f"  {a}↔{b}: train {m(key)*100:.1f}% ± {s(key)*100:.1f}%  |  "
          f"test {m(f'test_{a}_{b}')*100:.1f}% ± {s(f'test_{a}_{b}')*100:.1f}%")

print(f"\n  Active codes (LM):   {m('active_LM'):.1f} ± {s('active_LM'):.1f}")
print(f"  Active codes (VIS):  {m('active_VIS'):.1f} ± {s('active_VIS'):.1f}")
print(f"  Active codes (CLIP): {m('active_CLIP'):.1f} ± {s('active_CLIP'):.1f}")
print(f"  Active codes (MAE):  {m('active_MAE'):.1f} ± {s('active_MAE'):.1f}")
print(f"  Sense diversity:     {m('mean_sense_diversity'):.1f} ± {s('mean_sense_diversity'):.1f}")

# ── The key test ──────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("THEORY TEST: language-supervised vs physics-supervised clustering")
print("="*70)

lm_vis   = m("train_LM_VIS")
vis_mae  = m("train_VIS_MAE")
lm_clip  = m("train_LM_CLIP")
clip_mae = m("train_CLIP_MAE")
lm_mae   = m("train_LM_MAE")
vis_clip = m("train_VIS_CLIP")

print(f"\n  Within language-supervised  (LM↔CLIP):   {lm_clip*100:.1f}%")
print(f"  Within physics-supervised   (VIS↔MAE):   {vis_mae*100:.1f}%")
print(f"  Cross-group (LM↔MAE):                    {lm_mae*100:.1f}%")
print(f"  Cross-group (VIS↔CLIP):                  {vis_clip*100:.1f}%")

within_lang   = lm_clip
within_phys   = vis_mae
cross_average = (lm_mae + vis_clip) / 2

print(f"\n  Within-group average:  {(within_lang + within_phys)/2 * 100:.1f}%")
print(f"  Cross-group average:   {cross_average * 100:.1f}%")

if within_lang > cross_average and within_phys > cross_average:
    print("\n  ✓ CONFIRMED: language-supervised models cluster together,")
    print("               physics-supervised models cluster together.")
    print("               The gap is training objective, not modality.")
elif vis_mae > vis_clip:
    print("\n  ~ PARTIAL: VIS↔MAE > VIS↔CLIP supports the theory,")
    print("              but the separation is not clean.")
else:
    print("\n  ✗ NOT CONFIRMED: MAE does not cluster with V-JEPA 2.")
    print("    The split may be driven by something other than training objective.")

# ── Save ──────────────────────────────────────────────────────────────────────
summary = {
    "n_runs": total,
    "hyperparameters": {
        "codebook_dim": CODEBOOK_DIM, "num_codes": NUM_CODES,
        "epochs": EPOCHS, "lr": LR,
        "lambda_cm": LAMBDA_CM, "lambda_div": LAMBDA_DIV,
    },
    "means": {k: m(k) for k in all_results[0]},
    "stds":  {k: s(k) for k in all_results[0]},
    "theory_test": {
        "within_language_supervised_LM_CLIP": lm_clip,
        "within_physics_supervised_VIS_MAE":  vis_mae,
        "cross_LM_MAE":   lm_mae,
        "cross_VIS_CLIP": vis_clip,
        "confirmed": bool(within_lang > cross_average and within_phys > cross_average),
    },
    "all_runs": all_results,
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {OUTPUT_FILE}")
