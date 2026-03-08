"""
train_phrase_codebook.py
========================

Train the codebook on phrase-level event embeddings.

Key differences from concept-level training:

1. UNIT OF ALIGNMENT: event pairs, not concept pairs.
   "water flowing downhill" ↔ [image of stream] is one training example.
   This forces the codebook to align PHYSICAL SENSES, not concept categories.

2. DUAL CONTRASTIVE OBJECTIVE:
   (a) Cross-modal: same event should get the same code regardless of modality
   (b) Within-concept diversity: different events for the same concept should
       be allowed to get DIFFERENT codes (physical sense separation)

   The original single NT-Xent loss treated all same-concept pairs as positives.
   Here we define positives strictly as same-event cross-modal pairs.
   Same-concept different-event pairs are NEGATIVES — we want distinct physical
   sense codes, not concept codes.

3. EVALUATION METRICS:
   - Event-level code agreement: do LLM and WM agree on codes per event?
   - Concept-level sense diversity: how many distinct codes does each concept use?
   - PIQA sense probe: does phrase-level RSA correlate with PIQA improvement?

4. GENERALIZATION TEST:
   Split by concept (not event): hold out N concepts entirely, train on rest.
   This is the same test as before but now at phrase level.

Outputs:
  lm_output/phrase_level/codebook_results.json

Run:
  python train_phrase_codebook.py
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

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "lm_output/phrase_level"
OUTPUT_FILE   = "lm_output/phrase_level/codebook_results.json"

LAMBDA        = 0.5       # cross-modal contrastive weight (optimal from prior work)
LAMBDA_DIV    = 0.1       # within-concept diversity penalty (new)
CODEBOOK_DIM  = 256
NUM_CODES     = 64
EPOCHS        = 400       # more epochs — phrase-level has more data
LR            = 1e-3
SEEDS         = [42, 123, 7, 99, 2025]
N_TEST_CONCEPTS = 10      # held-out concepts for generalization test
N_SPLITS      = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load data ──────────────────────────────────────────────────────────────────

print("Loading phrase-level embeddings...")
lm_h    = np.load(os.path.join(DATA_DIR, "lm_hiddens_phrase.npy"))
st_h    = np.load(os.path.join(DATA_DIR, "st_hiddens_phrase.npy"))
clip_h  = np.load(os.path.join(DATA_DIR, "clip_hiddens_phrase.npy"))
vjepa_h = np.load(os.path.join(DATA_DIR, "vjepa2_hiddens_phrase.npy"))
mae_h   = np.load(os.path.join(DATA_DIR, "mae_hiddens_phrase.npy"))

with open(os.path.join(DATA_DIR, "event_index.json")) as f:
    event_index = json.load(f)

events   = event_index["events"]
concepts = event_index["concepts"]
N        = len(events)

print(f"  Events: {N}, Concepts: {len(concepts)}")

# Build event→concept mapping and concept→event-rows mapping
event_concepts = [e["concept"] for e in events]  # [N] concept labels
concept_to_rows = defaultdict(list)
for i, e in enumerate(events):
    concept_to_rows[e["concept"]].append(i)


# ── Architecture (same as before) ────────────────────────────────────────────

class VQCodebook(nn.Module):
    def __init__(self, num_codes, dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_codes, 1/num_codes)

    def forward(self, z):
        dists = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )
        indices = dists.argmin(1)
        quantized = self.embeddings(indices)
        quantized_st = z + (quantized - z).detach()
        commitment = F.mse_loss(z, quantized.detach())
        return quantized_st, indices, commitment


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, codebook_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, codebook_dim),
            nn.LayerNorm(codebook_dim),
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
        lm_recon  = self.lm_decoder(lm_q)
        vis_recon = self.vis_decoder(vis_q)
        return {
            "lm_z": lm_z, "vis_z": vis_z,
            "lm_q": lm_q, "vis_q": vis_q,
            "lm_idx": lm_idx, "vis_idx": vis_idx,
            "lm_commit": lm_commit, "vis_commit": vis_commit,
            "lm_recon": lm_recon, "vis_recon": vis_recon,
        }


# ── Loss functions ─────────────────────────────────────────────────────────────

def nt_xent_event_level(lm_z, vis_z, temperature=0.1):
    """
    Cross-modal NT-Xent where positives are SAME EVENT, different modality.
    Negatives are ALL other events (both cross-modal and within-modality).

    This is stricter than concept-level: same concept but different event
    counts as a negative. Forces sense-level alignment, not concept-level.
    """
    B = lm_z.shape[0]
    z_lm  = F.normalize(lm_z,  dim=1)
    z_vis = F.normalize(vis_z, dim=1)

    # Build [2B, 2B] similarity matrix
    z_all = torch.cat([z_lm, z_vis], dim=0)  # [2B, D]
    sim   = z_all @ z_all.T / temperature     # [2B, 2B]

    # Mask diagonal (self-similarity)
    mask = torch.eye(2*B, device=z_all.device).bool()
    sim  = sim.masked_fill(mask, float('-inf'))

    # Positive pairs: (i, B+i) and (B+i, i)
    labels = torch.cat([
        torch.arange(B, 2*B, device=z_all.device),
        torch.arange(0, B,   device=z_all.device),
    ])

    return F.cross_entropy(sim, labels)


def within_concept_diversity_loss(lm_z, vis_z, event_concepts_batch, temperature=0.1):
    """
    Diversity penalty: push DIFFERENT events for the SAME concept APART.
    Vectorized: builds a same-concept mask and computes masked mean similarity.
    """
    B = lm_z.shape[0]
    z_lm  = F.normalize(lm_z,  dim=1)
    z_vis = F.normalize(vis_z, dim=1)

    # Build same-concept mask [B, B] (excluding diagonal)
    # Map concepts to integer labels for fast comparison
    unique_concepts = list(set(event_concepts_batch))
    concept_map = {c: i for i, c in enumerate(unique_concepts)}
    labels = torch.tensor([concept_map[c] for c in event_concepts_batch],
                          device=lm_z.device)
    same_concept = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
    # Exclude diagonal (self-pairs)
    same_concept = same_concept & ~torch.eye(B, dtype=torch.bool, device=lm_z.device)
    # Upper triangle only to avoid double-counting
    same_concept = same_concept.triu(diagonal=1)

    n_pairs = same_concept.sum()
    if n_pairs == 0:
        return torch.tensor(0.0, device=lm_z.device)

    # Cosine similarity matrices
    sim_lm  = z_lm  @ z_lm.T   # [B, B]
    sim_vis = z_vis @ z_vis.T   # [B, B]

    # Mean similarity over same-concept pairs (minimize to push apart)
    loss = ((sim_lm[same_concept] + sim_vis[same_concept]) / 2).mean()
    return loss


def total_loss(out, lm_x, vis_x, event_concepts_batch, lambda_cm, lambda_div):
    # Reconstruction
    recon = F.mse_loss(out["lm_recon"], lm_x) + F.mse_loss(out["vis_recon"], vis_x)
    # Commitment
    commit = out["lm_commit"] + out["vis_commit"]
    # Cross-modal contrastive (same event = positive)
    cm = nt_xent_event_level(out["lm_z"], out["vis_z"])
    # Within-concept diversity (same concept different event = push apart)
    div = within_concept_diversity_loss(out["lm_z"], out["vis_z"], event_concepts_batch)

    total = recon + 0.25 * commit + lambda_cm * cm + lambda_div * div
    return total, {
        "recon": recon.item(), "commit": commit.item(),
        "cm": cm.item(), "div": div.item()
    }


# ── Training ───────────────────────────────────────────────────────────────────

def to_tensor(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)


def train_one_run(train_rows, test_rows, lm_data, vis_data, seed):
    torch.manual_seed(seed)
    random.seed(seed)

    lm_dim  = lm_data.shape[1]
    vis_dim = vis_data.shape[1]

    model = CrossModalCodebook(lm_dim, vis_dim, CODEBOOK_DIM, NUM_CODES).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    lm_train  = to_tensor(lm_data[train_rows])
    vis_train = to_tensor(vis_data[train_rows])
    train_concepts = [event_concepts[r] for r in train_rows]

    for epoch in range(EPOCHS):
        model.train()
        out = model(lm_train, vis_train)
        loss, _ = total_loss(out, lm_train, vis_train, train_concepts, LAMBDA, LAMBDA_DIV)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        # Training agreement
        out_train = model(lm_train, vis_train)
        train_agree = (out_train["lm_idx"] == out_train["vis_idx"]).float().mean().item()

        # Test agreement (held-out concepts)
        lm_test  = to_tensor(lm_data[test_rows])
        vis_test = to_tensor(vis_data[test_rows])
        out_test = model(lm_test, vis_test)
        test_agree = (out_test["lm_idx"] == out_test["vis_idx"]).float().mean().item()

        # Sense diversity: for each concept, how many unique codes does LM use?
        all_lm_idx = out_train["lm_idx"].cpu().numpy()
        all_vis_idx = out_train["vis_idx"].cpu().numpy()

        concept_diversity = {}
        for concept, rows in concept_to_rows.items():
            train_mask = [r for r in rows if r in train_rows]
            if not train_mask:
                continue
            local_lm_codes  = all_lm_idx[[train_rows.index(r) for r in train_mask if r in train_rows]]
            n_unique_lm  = len(set(local_lm_codes.tolist()))
            concept_diversity[concept] = n_unique_lm

        mean_diversity = np.mean(list(concept_diversity.values()))
        n_active_codes = len(set(all_lm_idx.tolist()))

    return {
        "train_agree": train_agree,
        "test_agree":  test_agree,
        "n_active_codes": n_active_codes,
        "mean_sense_diversity": mean_diversity,
        "concept_diversity": concept_diversity,
    }


# ── Generalization splits ──────────────────────────────────────────────────────

def make_splits(n_splits, n_test_concepts, seed=0):
    rng = random.Random(seed)
    splits = []
    for _ in range(n_splits):
        test_concepts = rng.sample(concepts, n_test_concepts)
        train_concepts_split = [c for c in concepts if c not in test_concepts]

        train_rows = []
        test_rows  = []
        for i, e in enumerate(events):
            if e["concept"] in train_concepts_split:
                train_rows.append(i)
            else:
                test_rows.append(i)

        splits.append({
            "train_rows":     train_rows,
            "test_rows":      test_rows,
            "test_concepts":  test_concepts,
        })
    return splits


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Events: {N} total, {len(concepts)} concepts")

    # Use Mistral + V-JEPA 2 as the primary pair (matches paper)
    lm_data  = lm_h.copy()
    vis_data = vjepa_h.copy()

    splits = make_splits(N_SPLITS, N_TEST_CONCEPTS)

    print(f"\nRunning {N_SPLITS} splits × {len(SEEDS)} seeds = {N_SPLITS * len(SEEDS)} runs\n")

    all_results = []
    run_idx = 0

    for split_i, split in enumerate(splits):
        for seed in SEEDS:
            run_idx += 1
            res = train_one_run(
                split["train_rows"], split["test_rows"],
                lm_data, vis_data, seed
            )
            res["split"] = split_i
            res["seed"]  = seed
            res["test_concepts"] = split["test_concepts"]
            all_results.append(res)

            if run_idx % 10 == 0 or run_idx <= 5:
                print(f"  Run {run_idx:3d}/{N_SPLITS*len(SEEDS)} | "
                      f"Train {res['train_agree']*100:.1f}% | "
                      f"Test {res['test_agree']*100:.1f}% | "
                      f"Active codes: {res['n_active_codes']} | "
                      f"Sense diversity: {res['mean_sense_diversity']:.1f}")

    # Aggregate
    train_agrees = [r["train_agree"] for r in all_results]
    test_agrees  = [r["test_agree"]  for r in all_results]
    diversities  = [r["mean_sense_diversity"] for r in all_results]
    active_codes = [r["n_active_codes"] for r in all_results]

    summary = {
        "train_agree_mean": float(np.mean(train_agrees)),
        "train_agree_std":  float(np.std(train_agrees)),
        "test_agree_mean":  float(np.mean(test_agrees)),
        "test_agree_std":   float(np.std(test_agrees)),
        "gap_mean":         float(np.mean(train_agrees) - np.mean(test_agrees)),
        "mean_sense_diversity": float(np.mean(diversities)),
        "mean_active_codes": float(np.mean(active_codes)),
        "n_runs": len(all_results),
    }

    # Per-concept diversity (aggregated)
    concept_div_all = defaultdict(list)
    for r in all_results:
        for c, d in r["concept_diversity"].items():
            concept_div_all[c].append(d)
    concept_div_summary = {
        c: {"mean": float(np.mean(v)), "std": float(np.std(v))}
        for c, v in concept_div_all.items()
    }

    output = {
        "summary": summary,
        "concept_diversity": concept_div_summary,
        "all_results": all_results,
        "config": {
            "lambda_cm": LAMBDA,
            "lambda_div": LAMBDA_DIV,
            "codebook_dim": CODEBOOK_DIM,
            "num_codes": NUM_CODES,
            "epochs": EPOCHS,
            "n_test_concepts": N_TEST_CONCEPTS,
            "n_splits": N_SPLITS,
            "seeds": SEEDS,
        }
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print("PHRASE-LEVEL CODEBOOK RESULTS")
    print(f"{'='*60}")
    print(f"Train agreement:    {summary['train_agree_mean']*100:.1f}% ± {summary['train_agree_std']*100:.1f}%")
    print(f"Test agreement:     {summary['test_agree_mean']*100:.1f}% ± {summary['test_agree_std']*100:.1f}%")
    print(f"Generalization gap: {summary['gap_mean']*100:.1f}%")
    print(f"Active codes:       {summary['mean_active_codes']:.1f}")
    print(f"Sense diversity:    {summary['mean_sense_diversity']:.1f} unique codes/concept")
    print()
    print("Top 10 most sense-diverse concepts:")
    sorted_div = sorted(concept_div_summary.items(), key=lambda x: -x[1]["mean"])
    for c, d in sorted_div[:10]:
        print(f"  {c:<14}  {d['mean']:.1f} ± {d['std']:.1f} codes")
    print()
    print(f"Saved to: {OUTPUT_FILE}")

    # What to look for
    print(f"\n{'='*60}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*60}")
    print()
    print("Sense diversity > 1.0:")
    print("  The codebook is learning sub-concept sense structure.")
    print("  Compare to concept-level run: diversity should be ~1.0 there.")
    print()
    print("Generalization gap:")
    print("  If gap < 10%: phrase-level training generalizes to unseen concepts")
    print("  If gap > 30%: sense structure is concept-specific, not transferable")
    print()
    print("Active codes:")
    print("  Concept-level run: ~2 active codes (binary partition)")
    print("  Phrase-level run should be >> 2 if sense structure is learned")
    print()
    print("Per-concept diversity to compare with PIQA per-concept results:")
    print("  If diversity correlates with PIQA improvement: sense-level codebook")
    print("  helps on concepts where disambiguation matters most.")


if __name__ == "__main__":
    main()
