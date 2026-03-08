"""
lambda_sweep_phrase.py
======================

Quick sweep over (lambda_cm, lambda_div) pairs to find the combination
that maximizes active codes and sense diversity while keeping agreement high.

The default (lambda_cm=0.5, lambda_div=0.1) collapses to 2 active codes.
Hypothesis: reducing contrastive pressure and increasing diversity pressure
will force the codebook to use more than 2 codes.

Runs 5 seeds per config (not 50) for speed. ~15 min total.

Usage:
    python lambda_sweep_phrase.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict

from phrase_bank import PHRASE_BANK

DATA_DIR    = "lm_output/phrase_level"
OUTPUT_FILE = "lm_output/phrase_level/lambda_sweep_results.json"

# ── Sweep grid ─────────────────────────────────────────────────────────────────
# Each tuple: (lambda_cm, lambda_div, label)
CONFIGS = [
    (0.5,  0.1,  "baseline"),         # original
    (0.1,  1.0,  "low_cm_high_div"),  # proposed
    (0.1,  0.3,  "low_cm_med_div"),
    (0.3,  1.0,  "med_cm_high_div"),
    (0.05, 2.0,  "very_low_cm"),
    (0.5,  1.0,  "high_div_only"),
    (0.1,  0.0,  "no_div"),           # control: what does low_cm alone do?
    (0.0,  1.0,  "div_only"),         # control: diversity alone, no cross-modal
]

CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 400
LR           = 1e-3
SEEDS        = [42, 123, 7, 99, 2025]
N_TEST_CONCEPTS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading embeddings...")
lm_h    = np.load(os.path.join(DATA_DIR, "lm_hiddens_phrase.npy"))
vjepa_h = np.load(os.path.join(DATA_DIR, "vjepa2_hiddens_phrase.npy"))

with open(os.path.join(DATA_DIR, "event_index.json")) as f:
    event_index = json.load(f)

events   = event_index["events"]
concepts = event_index["concepts"]
N        = len(events)

event_concepts = [e["concept"] for e in events]
concept_to_rows = defaultdict(list)
for i, e in enumerate(events):
    concept_to_rows[e["concept"]].append(i)

print(f"  {N} events, {len(concepts)} concepts")

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
        indices = dists.argmin(1)
        quantized = self.embeddings(indices)
        quantized_st = z + (quantized - z).detach()
        commitment = F.mse_loss(z, quantized.detach())
        return quantized_st, indices, commitment

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, codebook_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, codebook_dim), nn.LayerNorm(codebook_dim),
        )
    def forward(self, x):
        return self.encoder(x)

class CrossModalCodebook(nn.Module):
    def __init__(self, lm_dim, vis_dim, codebook_dim, num_codes):
        super().__init__()
        self.lm_encoder  = ModalityEncoder(lm_dim, codebook_dim)
        self.vis_encoder = ModalityEncoder(vis_dim, codebook_dim)
        self.codebook    = VQCodebook(num_codes, codebook_dim)
        self.lm_decoder  = nn.Linear(codebook_dim, lm_dim)
        self.vis_decoder = nn.Linear(codebook_dim, vis_dim)

    def forward(self, lm_x, vis_x):
        lm_z  = self.lm_encoder(lm_x)
        vis_z = self.vis_encoder(vis_x)
        lm_q,  lm_idx,  lm_commit  = self.codebook(lm_z)
        vis_q, vis_idx, vis_commit = self.codebook(vis_z)
        return {
            "lm_z": lm_z, "vis_z": vis_z,
            "lm_q": lm_q, "vis_q": vis_q,
            "lm_idx": lm_idx, "vis_idx": vis_idx,
            "lm_commit": lm_commit, "vis_commit": vis_commit,
            "lm_recon": self.lm_decoder(lm_q),
            "vis_recon": self.vis_decoder(vis_q),
        }

# ── Loss ──────────────────────────────────────────────────────────────────────

def nt_xent(lm_z, vis_z, temperature=0.1):
    B = lm_z.shape[0]
    z_lm  = F.normalize(lm_z,  dim=1)
    z_vis = F.normalize(vis_z, dim=1)
    z_all = torch.cat([z_lm, z_vis], dim=0)
    sim   = z_all @ z_all.T / temperature
    mask  = torch.eye(2*B, device=z_all.device).bool()
    sim   = sim.masked_fill(mask, float('-inf'))
    labels = torch.cat([torch.arange(B, 2*B, device=z_all.device),
                        torch.arange(0, B,   device=z_all.device)])
    return F.cross_entropy(sim, labels)

def diversity_loss_vectorized(lm_z, vis_z, concept_labels_tensor):
    z_lm_n  = F.normalize(lm_z,  dim=1)
    z_vis_n = F.normalize(vis_z, dim=1)
    B = lm_z.shape[0]
    same = (concept_labels_tensor.unsqueeze(0) == concept_labels_tensor.unsqueeze(1))
    same = same & ~torch.eye(B, dtype=torch.bool, device=lm_z.device)
    same = same.triu(diagonal=1)
    if not same.any():
        return torch.tensor(0.0, device=lm_z.device)
    sim_lm  = z_lm_n  @ z_lm_n.T
    sim_vis = z_vis_n @ z_vis_n.T
    return ((sim_lm[same] + sim_vis[same]) / 2).mean()

def total_loss(out, lm_x, vis_x, concept_labels_tensor, lam_cm, lam_div):
    recon  = F.mse_loss(out["lm_recon"], lm_x) + F.mse_loss(out["vis_recon"], vis_x)
    commit = out["lm_commit"] + out["vis_commit"]
    cm     = nt_xent(out["lm_z"], out["vis_z"]) if lam_cm > 0 else torch.tensor(0.0, device=lm_x.device)
    div    = diversity_loss_vectorized(out["lm_z"], out["vis_z"], concept_labels_tensor) if lam_div > 0 else torch.tensor(0.0, device=lm_x.device)
    return recon + 0.25 * commit + lam_cm * cm + lam_div * div

# ── Training ──────────────────────────────────────────────────────────────────

def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32).to(DEVICE)

def make_split(seed=0):
    rng = random.Random(seed)
    test_concepts = rng.sample(concepts, N_TEST_CONCEPTS)
    train_rows, test_rows = [], []
    for i, e in enumerate(events):
        (test_rows if e["concept"] in test_concepts else train_rows).append(i)
    return train_rows, test_rows

def run_config(lam_cm, lam_div, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    train_rows, test_rows = make_split(seed)

    lm_train  = to_tensor(lm_h[train_rows])
    vis_train = to_tensor(vjepa_h[train_rows])
    train_concept_names = [event_concepts[r] for r in train_rows]
    unique_c = list(set(train_concept_names))
    cmap = {c: i for i, c in enumerate(unique_c)}
    labels_t = torch.tensor([cmap[c] for c in train_concept_names], device=DEVICE)

    model = CrossModalCodebook(lm_h.shape[1], vjepa_h.shape[1], CODEBOOK_DIM, NUM_CODES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        out  = model(lm_train, vis_train)
        loss = total_loss(out, lm_train, vis_train, labels_t, lam_cm, lam_div)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_train = model(lm_train, vis_train)
        train_agree = (out_train["lm_idx"] == out_train["vis_idx"]).float().mean().item()
        n_active = len(set(out_train["lm_idx"].cpu().numpy().tolist()))

        lm_test  = to_tensor(lm_h[test_rows])
        vis_test = to_tensor(vjepa_h[test_rows])
        out_test = model(lm_test, vis_test)
        test_agree = (out_test["lm_idx"] == out_test["vis_idx"]).float().mean().item()

        # Sense diversity per concept (train)
        all_lm_idx = out_train["lm_idx"].cpu().numpy()
        diversities = []
        for concept in unique_c:
            rows_local = [i for i, c in enumerate(train_concept_names) if c == concept]
            codes = set(all_lm_idx[rows_local].tolist())
            diversities.append(len(codes))
        mean_div = np.mean(diversities)

    return {
        "train_agree": train_agree,
        "test_agree":  test_agree,
        "n_active": n_active,
        "mean_sense_diversity": mean_div,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

print(f"\n{'Config':<22} {'λ_cm':>6} {'λ_div':>6} | {'Train':>7} {'Test':>7} {'Codes':>6} {'Diversity':>10}")
print("-" * 75)

all_results = {}

for lam_cm, lam_div, label in CONFIGS:
    seed_results = [run_config(lam_cm, lam_div, s) for s in SEEDS]

    train_m = np.mean([r["train_agree"]          for r in seed_results])
    test_m  = np.mean([r["test_agree"]            for r in seed_results])
    codes_m = np.mean([r["n_active"]              for r in seed_results])
    div_m   = np.mean([r["mean_sense_diversity"]  for r in seed_results])
    codes_s = np.std( [r["n_active"]              for r in seed_results])

    print(f"  {label:<20} {lam_cm:>6.2f} {lam_div:>6.2f} | "
          f"{train_m*100:>6.1f}% {test_m*100:>6.1f}% "
          f"{codes_m:>5.1f}±{codes_s:.1f} {div_m:>9.1f}")

    all_results[label] = {
        "lambda_cm": lam_cm, "lambda_div": lam_div,
        "train_agree_mean": float(train_m), "test_agree_mean": float(test_m),
        "n_active_mean": float(codes_m), "n_active_std": float(codes_s),
        "mean_sense_diversity": float(div_m),
        "seed_results": seed_results,
    }

print()

# Find best config by active codes (with agreement > 70% constraint)
best = max(
    [(k, v) for k, v in all_results.items() if v["train_agree_mean"] > 0.7],
    key=lambda x: x[1]["n_active_mean"]
)
print(f"Best config (max codes, agreement > 70%): {best[0]}")
print(f"  Active codes: {best[1]['n_active_mean']:.1f}")
print(f"  Train/Test agreement: {best[1]['train_agree_mean']*100:.1f}% / {best[1]['test_agree_mean']*100:.1f}%")
print(f"  Sense diversity: {best[1]['mean_sense_diversity']:.1f}")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2)
print(f"\nSaved to: {OUTPUT_FILE}")
