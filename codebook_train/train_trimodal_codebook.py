"""
train_trimodal_codebook.py
==========================

3-modality shared codebook: Mistral (LM) + V-JEPA 2 (visual) + CLIP-text (vision-language).

Key hypothesis: with 3 modalities there are 3 pairwise cross-modal constraints
(LM↔VIS, LM↔CLIP, VIS↔CLIP) that cannot all be satisfied by a binary codebook,
forcing the model to use more codes and capture genuine sense structure.

Each modality pair has a different "flavor" of disagreement:
  LM   ↔ VIS   : language-vs-physical-world gap (your main finding)
  LM   ↔ CLIP  : language-model-vs-vision-language gap
  VIS  ↔ CLIP  : predictive-video-model-vs-contrastive-image gap

Runs 10 splits × 5 seeds = 50 runs. ~40 min.

Usage:
    python train_trimodal_codebook.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

DATA_DIR    = "lm_output/phrase_level"
OUTPUT_FILE = "lm_output/phrase_level/trimodal_codebook_results.json"

# ── Hyperparameters ────────────────────────────────────────────────────────────
CODEBOOK_DIM    = 256
NUM_CODES       = 64
EPOCHS          = 400
LR              = 1e-3
N_SPLITS        = 10
N_SEEDS         = 5
N_TEST_CONCEPTS = 10
TEMPERATURE     = 0.1

# Best config from lambda sweep + adjusted for 3 modalities
# With 3 pairwise losses, total cross-modal signal is 3× stronger,
# so we reduce each individual λ_cm proportionally.
LAMBDA_CM  = 0.05   # per modality-pair (×3 pairs = 0.15 effective)
LAMBDA_DIV = 0.5    # diversity penalty
LAMBDA_REC = 1.0    # reconstruction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load embeddings ────────────────────────────────────────────────────────────
print("Loading embeddings...")
lm_h     = np.load(os.path.join(DATA_DIR, "lm_hiddens_phrase.npy"))     # [251, 4096]
vjepa_h  = np.load(os.path.join(DATA_DIR, "vjepa2_hiddens_phrase.npy")) # [251, 1024]
clipt_h  = np.load(os.path.join(DATA_DIR, "clip_text_hiddens_phrase.npy")) # [251, 768]

with open(os.path.join(DATA_DIR, "event_index.json")) as f:
    event_index = json.load(f)
events   = event_index["events"]
concepts = event_index["concepts"]
N        = len(events)

event_concepts = [e["concept"] for e in events]
concept_to_rows = defaultdict(list)
for i, e in enumerate(events):
    concept_to_rows[e["concept"]].append(i)

print(f"  LM:        {lm_h.shape}")
print(f"  V-JEPA 2:  {vjepa_h.shape}")
print(f"  CLIP-text: {clipt_h.shape}")
print(f"  Events: {N}, Concepts: {len(concepts)}")

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
        commitment = F.mse_loss(z, q.detach())
        return q_st, idx, commitment

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, codebook_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, codebook_dim), nn.LayerNorm(codebook_dim),
        )
    def forward(self, x):
        return self.net(x)

class TriModalCodebook(nn.Module):
    def __init__(self, lm_dim, vis_dim, clip_dim, codebook_dim, num_codes):
        super().__init__()
        self.lm_enc   = ModalityEncoder(lm_dim,   codebook_dim)
        self.vis_enc  = ModalityEncoder(vis_dim,  codebook_dim)
        self.clip_enc = ModalityEncoder(clip_dim, codebook_dim)
        self.codebook = VQCodebook(num_codes, codebook_dim)
        self.lm_dec   = nn.Linear(codebook_dim, lm_dim)
        self.vis_dec  = nn.Linear(codebook_dim, vis_dim)
        self.clip_dec = nn.Linear(codebook_dim, clip_dim)

    def forward(self, lm_x, vis_x, clip_x):
        lm_z   = self.lm_enc(lm_x)
        vis_z  = self.vis_enc(vis_x)
        clip_z = self.clip_enc(clip_x)

        lm_q,   lm_idx,   lm_c   = self.codebook(lm_z)
        vis_q,  vis_idx,  vis_c  = self.codebook(vis_z)
        clip_q, clip_idx, clip_c = self.codebook(clip_z)

        return {
            "lm_z": lm_z, "vis_z": vis_z, "clip_z": clip_z,
            "lm_q": lm_q, "vis_q": vis_q, "clip_q": clip_q,
            "lm_idx": lm_idx, "vis_idx": vis_idx, "clip_idx": clip_idx,
            "lm_commit": lm_c, "vis_commit": vis_c, "clip_commit": clip_c,
            "lm_recon":   self.lm_dec(lm_q),
            "vis_recon":  self.vis_dec(vis_q),
            "clip_recon": self.clip_dec(clip_q),
        }

# ── Losses ────────────────────────────────────────────────────────────────────

def nt_xent(z1, z2, temperature=TEMPERATURE):
    B = z1.shape[0]
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    z   = torch.cat([z1n, z2n], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2*B, device=z.device).bool()
    sim  = sim.masked_fill(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B, device=z.device),
                        torch.arange(0, B,   device=z.device)])
    return F.cross_entropy(sim, labels)

def diversity_loss(z1, z2, concept_labels):
    z1n = F.normalize(z1, dim=1)
    z2n = F.normalize(z2, dim=1)
    B = z1.shape[0]
    same = (concept_labels.unsqueeze(0) == concept_labels.unsqueeze(1))
    same = same & ~torch.eye(B, dtype=torch.bool, device=z1.device)
    same = same.triu(diagonal=1)
    if not same.any():
        return torch.tensor(0.0, device=z1.device)
    return ((z1n @ z1n.T)[same] + (z2n @ z2n.T)[same]).mean() / 2

def compute_loss(out, lm_x, vis_x, clip_x, concept_labels):
    # Reconstruction
    rec = (F.mse_loss(out["lm_recon"],   lm_x) +
           F.mse_loss(out["vis_recon"],  vis_x) +
           F.mse_loss(out["clip_recon"], clip_x))

    # Commitment
    commit = out["lm_commit"] + out["vis_commit"] + out["clip_commit"]

    # 3 pairwise cross-modal losses
    cm = (nt_xent(out["lm_z"],  out["vis_z"])  +
          nt_xent(out["lm_z"],  out["clip_z"]) +
          nt_xent(out["vis_z"], out["clip_z"]))

    # Diversity (use LM and VIS as anchor pair)
    div = diversity_loss(out["lm_z"], out["vis_z"], concept_labels)

    return (LAMBDA_REC * rec +
            0.25 * commit +
            LAMBDA_CM * cm +
            LAMBDA_DIV * div)

# ── Helpers ───────────────────────────────────────────────────────────────────

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

    lm_tr   = to_t(lm_h[train_rows])
    vis_tr  = to_t(vjepa_h[train_rows])
    clip_tr = to_t(clipt_h[train_rows])

    train_concepts = [event_concepts[r] for r in train_rows]
    unique_c = list(set(train_concepts))
    cmap     = {c: i for i, c in enumerate(unique_c)}
    labels_t = torch.tensor([cmap[c] for c in train_concepts], device=DEVICE)

    model = TriModalCodebook(
        lm_h.shape[1], vjepa_h.shape[1], clipt_h.shape[1],
        CODEBOOK_DIM, NUM_CODES
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        model.train()
        out  = model(lm_tr, vis_tr, clip_tr)
        loss = compute_loss(out, lm_tr, vis_tr, clip_tr, labels_t)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        # ── Train metrics ──
        out_tr = model(lm_tr, vis_tr, clip_tr)
        lm_vis_agree   = (out_tr["lm_idx"]  == out_tr["vis_idx"]).float().mean().item()
        lm_clip_agree  = (out_tr["lm_idx"]  == out_tr["clip_idx"]).float().mean().item()
        vis_clip_agree = (out_tr["vis_idx"] == out_tr["clip_idx"]).float().mean().item()
        n_active_lm    = len(set(out_tr["lm_idx"].cpu().numpy().tolist()))
        n_active_vis   = len(set(out_tr["vis_idx"].cpu().numpy().tolist()))
        n_active_clip  = len(set(out_tr["clip_idx"].cpu().numpy().tolist()))

        # ── Sense diversity (LM indices, train) ──
        all_lm = out_tr["lm_idx"].cpu().numpy()
        divs = [len(set(all_lm[[i for i, c in enumerate(train_concepts) if c == concept]].tolist()))
                for concept in unique_c]
        mean_div = float(np.mean(divs))

        # ── Test metrics ──
        lm_te   = to_t(lm_h[test_rows])
        vis_te  = to_t(vjepa_h[test_rows])
        clip_te = to_t(clipt_h[test_rows])
        out_te  = model(lm_te, vis_te, clip_te)
        test_lm_vis   = (out_te["lm_idx"] == out_te["vis_idx"]).float().mean().item()
        test_lm_clip  = (out_te["lm_idx"] == out_te["clip_idx"]).float().mean().item()
        test_vis_clip = (out_te["vis_idx"] == out_te["clip_idx"]).float().mean().item()

    return {
        "train_lm_vis_agree":   lm_vis_agree,
        "train_lm_clip_agree":  lm_clip_agree,
        "train_vis_clip_agree": vis_clip_agree,
        "test_lm_vis_agree":    test_lm_vis,
        "test_lm_clip_agree":   test_lm_clip,
        "test_vis_clip_agree":  test_vis_clip,
        "n_active_lm":    n_active_lm,
        "n_active_vis":   n_active_vis,
        "n_active_clip":  n_active_clip,
        "mean_sense_diversity": mean_div,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

all_results = []
split_seeds = list(range(N_SPLITS))
seeds       = list(range(N_SEEDS))
total = N_SPLITS * N_SEEDS
done  = 0

print(f"\nRunning {N_SPLITS} splits × {N_SEEDS} seeds = {total} runs...\n")
print(f"{'Run':>4} | {'LM↔VIS':>7} {'LM↔CLIP':>8} {'VIS↔CLIP':>9} | "
      f"{'Codes(LM)':>10} {'Diversity':>10}")
print("-" * 60)

for split_seed in split_seeds:
    for seed in seeds:
        r = run_once(seed, split_seed)
        all_results.append(r)
        done += 1
        if done % 5 == 0 or done == total:
            print(f"{done:>4} | "
                  f"{r['train_lm_vis_agree']*100:>6.1f}% "
                  f"{r['train_lm_clip_agree']*100:>7.1f}% "
                  f"{r['train_vis_clip_agree']*100:>8.1f}% | "
                  f"{r['n_active_lm']:>10} "
                  f"{r['mean_sense_diversity']:>10.1f}")

# ── Summary ───────────────────────────────────────────────────────────────────
def m(key): return float(np.mean([r[key] for r in all_results]))
def s(key): return float(np.std( [r[key] for r in all_results]))

print("\n" + "="*60)
print("SUMMARY (mean ± std over all runs)")
print("="*60)
print(f"  Train LM↔VIS agreement:   {m('train_lm_vis_agree')*100:.1f}% ± {s('train_lm_vis_agree')*100:.1f}%")
print(f"  Train LM↔CLIP agreement:  {m('train_lm_clip_agree')*100:.1f}% ± {s('train_lm_clip_agree')*100:.1f}%")
print(f"  Train VIS↔CLIP agreement: {m('train_vis_clip_agree')*100:.1f}% ± {s('train_vis_clip_agree')*100:.1f}%")
print(f"  Test  LM↔VIS agreement:   {m('test_lm_vis_agree')*100:.1f}% ± {s('test_lm_vis_agree')*100:.1f}%")
print(f"  Test  LM↔CLIP agreement:  {m('test_lm_clip_agree')*100:.1f}% ± {s('test_lm_clip_agree')*100:.1f}%")
print(f"  Test  VIS↔CLIP agreement: {m('test_vis_clip_agree')*100:.1f}% ± {s('test_vis_clip_agree')*100:.1f}%")
print(f"  Active codes (LM):        {m('n_active_lm'):.1f} ± {s('n_active_lm'):.1f}")
print(f"  Active codes (VIS):       {m('n_active_vis'):.1f} ± {s('n_active_vis'):.1f}")
print(f"  Active codes (CLIP):      {m('n_active_clip'):.1f} ± {s('n_active_clip'):.1f}")
print(f"  Sense diversity:          {m('mean_sense_diversity'):.1f} ± {s('mean_sense_diversity'):.1f}")

# ── Compare to 2-modality baseline from prior run ─────────────────────────────
baseline_lm_vis  = 0.885
baseline_codes   = 2.0
baseline_div     = 1.4

print("\n" + "-"*60)
print("vs. 2-modality baseline (Exp 9):")
print(f"  LM↔VIS agreement: {baseline_lm_vis*100:.1f}% → {m('train_lm_vis_agree')*100:.1f}%")
print(f"  Active codes:     {baseline_codes:.1f} → {m('n_active_lm'):.1f}")
print(f"  Sense diversity:  {baseline_div:.1f} → {m('mean_sense_diversity'):.1f}")

# Interpret VIS↔CLIP gap: this is the "new" gap exposed by adding CLIP-text
vis_clip = m('train_vis_clip_agree')
lm_clip  = m('train_lm_clip_agree')
lm_vis   = m('train_lm_vis_agree')
print(f"\n  VIS↔CLIP gap ({vis_clip*100:.1f}%) vs LM↔VIS gap ({lm_vis*100:.1f}%)")
if vis_clip < lm_vis:
    print("  → Video model and CLIP-text disagree MORE than LM and video model")
    print("    (CLIP-text is closer to LM than to V-JEPA 2)")
elif vis_clip > lm_vis:
    print("  → Video model and CLIP-text agree MORE than LM and video model")
    print("    (CLIP-text bridges the gap: more aligned with visual than LM is)")
else:
    print("  → VIS↔CLIP and LM↔VIS gaps are similar")

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
    "all_runs": all_results,
}

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved → {OUTPUT_FILE}")
