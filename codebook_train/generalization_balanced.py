"""
Generalization verification: batch-balanced train/test splits.

The original generalization experiment may have a confound:
  - Group B (code 0) = exactly the 17 original concepts
  - Group A (code 1) = exactly the 32 new concepts

If the codebook learned "original batch vs new batch" rather than
cross-modal semantic structure, test performance would drop when
train/test splits are forced to mix both batches.

This script runs two conditions:

  CONDITION A — Batch-confounded (replicates original experiment)
    Train: 37 concepts from ONE batch (biased toward original-heavy)
    Test: 12 held-out concepts
    Expected: ~97% if our original result was real or spurious

  CONDITION B — Batch-balanced (the real test)
    Each split is forced to contain BOTH original and new concepts
    in both train AND test sets.
    Train: 37 concepts = ~12 original + ~25 new
    Test:  12 concepts = ~5 original + ~7 new
    Expected: ~97% if generalisation is real, ~0% if batch artifact

Both conditions: 10 splits × 5 seeds = 50 runs each.
Outputs: lm_output/generalization_balanced_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from itertools import combinations

# ── Config ────────────────────────────────────────────────────────────────────
LAMBDA       = 0.5
CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 300
LR           = 1e-3
N_SPLITS     = 10
SEEDS        = [42, 123, 7, 99, 2025]
N_TEST       = 12   # held-out per split

# ── Load ──────────────────────────────────────────────────────────────────────
st_all = np.load("lm_output/st_hiddens_expanded.npy")    # [49, 768]
vj_all = np.load("lm_output/vjepa2_hiddens_expanded.npy") # [49, 1024]

with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)

all_concepts  = idx["all_concepts"]   # 49 total
orig_concepts = idx["original_concepts"]  # 17 original
N = len(all_concepts)

orig_idx = [all_concepts.index(c) for c in orig_concepts]
new_idx  = [i for i in range(N) if i not in orig_idx]

print(f"Total: {N}, Original: {len(orig_idx)}, New: {len(new_idx)}")
print(f"Test size: {N_TEST}, Train size: {N - N_TEST}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── Architecture ──────────────────────────────────────────────────────────────

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
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, codebook_dim), nn.LayerNorm(codebook_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 512), nn.ReLU(),
            nn.Linear(512, input_dim),
        )
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)


class CrossModalCodebook(nn.Module):
    def __init__(self, dim_a, dim_b, codebook_dim, num_codes):
        super().__init__()
        self.enc_a = ModalityEncoder(dim_a, codebook_dim)
        self.enc_b = ModalityEncoder(dim_b, codebook_dim)
        self.vq    = VQCodebook(num_codes, codebook_dim)

    def forward(self, x_a, x_b):
        z_a = self.enc_a.encode(x_a)
        z_b = self.enc_b.encode(x_b)
        q_a, idx_a, comm_a = self.vq(z_a)
        q_b, idx_b, comm_b = self.vq(z_b)
        recon_a = self.enc_a.decode(q_a)
        recon_b = self.enc_b.decode(q_b)
        return z_a, z_b, q_a, q_b, idx_a, idx_b, recon_a, recon_b, comm_a, comm_b


def nt_xent_loss(z_a, z_b, temperature=0.5):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    Nb = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2*Nb, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(Nb, 2*Nb), torch.arange(0, Nb)]).to(z.device)
    return F.cross_entropy(sim, labels)


def run_one(train_idx, test_idx, seed):
    """Train on train_idx, evaluate code agreement on test_idx."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_st_train = torch.tensor(st_all[train_idx], dtype=torch.float32).to(device)
    x_vj_train = torch.tensor(vj_all[train_idx], dtype=torch.float32).to(device)
    x_st_test  = torch.tensor(st_all[test_idx],  dtype=torch.float32).to(device)
    x_vj_test  = torch.tensor(vj_all[test_idx],  dtype=torch.float32).to(device)

    model = CrossModalCodebook(768, 1024, CODEBOOK_DIM, NUM_CODES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        z_a, z_b, q_a, q_b, idx_a, idx_b, ra, rb, ca, cb = model(x_st_train, x_vj_train)
        recon_loss   = F.mse_loss(ra, x_st_train) + F.mse_loss(rb, x_vj_train)
        commit_loss  = ca + cb
        contrast_loss = nt_xent_loss(z_a, z_b)
        loss = recon_loss + 0.25 * commit_loss + LAMBDA * contrast_loss
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        _, _, _, _, tr_a, tr_b, _, _, _, _ = model(x_st_train, x_vj_train)
        _, _, _, _, te_a, te_b, _, _, _, _ = model(x_st_test,  x_vj_test)

    train_agree = float((tr_a == tr_b).float().mean())
    test_agree  = float((te_a == te_b).float().mean())

    # Per-concept test agreement
    per_concept = {all_concepts[test_idx[i]]: int(te_a[i] == te_b[i])
                   for i in range(len(test_idx))}

    return train_agree, test_agree, per_concept


# ── Generate splits ───────────────────────────────────────────────────────────

rng = random.Random(0)

# Condition A: uncontrolled splits (may be batch-confounded)
splits_a = []
for _ in range(N_SPLITS):
    test = rng.sample(range(N), N_TEST)
    train = [i for i in range(N) if i not in test]
    splits_a.append((train, test))

# Condition B: balanced splits — test must contain both original and new
# Target: ~5 original + ~7 new in test (proportional to 17/32 ratio)
N_TEST_ORIG = 5
N_TEST_NEW  = N_TEST - N_TEST_ORIG   # 7

splits_b = []
for _ in range(N_SPLITS):
    test_orig = rng.sample(orig_idx, N_TEST_ORIG)
    test_new  = rng.sample(new_idx,  N_TEST_NEW)
    test  = test_orig + test_new
    train = [i for i in range(N) if i not in test]
    splits_b.append((train, test))

    # Verify balance
    orig_in_train = sum(1 for i in train if i in orig_idx)
    new_in_train  = sum(1 for i in train if i in new_idx)
    assert len(test) == N_TEST
    assert len(train) == N - N_TEST

# ── Run Condition A ───────────────────────────────────────────────────────────

print("=" * 65)
print("CONDITION A: Uncontrolled splits (may be batch-confounded)")
print("=" * 65)

results_a = []
for si, (train_idx, test_idx) in enumerate(splits_a):
    orig_in_test = sum(1 for i in test_idx if i in orig_idx)
    new_in_test  = sum(1 for i in test_idx if i in new_idx)
    split_results = []
    for seed in SEEDS:
        tr, te, per_c = run_one(train_idx, test_idx, seed)
        split_results.append({"train": tr, "test": te, "per_concept": per_c, "seed": seed})
    mean_te = np.mean([r["test"] for r in split_results])
    mean_tr = np.mean([r["train"] for r in split_results])
    print(f"  Split {si+1:2d} (orig_in_test={orig_in_test}, new_in_test={new_in_test}): "
          f"train={mean_tr:.0%}  test={mean_te:.0%}")
    results_a.extend(split_results)

all_train_a = [r["train"] for r in results_a]
all_test_a  = [r["test"]  for r in results_a]
print(f"\n  CONDITION A SUMMARY ({len(results_a)} runs):")
print(f"  Train: {np.mean(all_train_a):.1%} ± {np.std(all_train_a):.1%}")
print(f"  Test:  {np.mean(all_test_a):.1%}  ± {np.std(all_test_a):.1%}")
print(f"  Gap:   {np.mean(all_train_a) - np.mean(all_test_a):.1%}")

# ── Run Condition B ───────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CONDITION B: Batch-balanced splits (5 original + 7 new in test)")
print("=" * 65)

results_b = []
for si, (train_idx, test_idx) in enumerate(splits_b):
    orig_in_test = sum(1 for i in test_idx if i in orig_idx)
    new_in_test  = sum(1 for i in test_idx if i in new_idx)
    split_results = []
    for seed in SEEDS:
        tr, te, per_c = run_one(train_idx, test_idx, seed)
        split_results.append({"train": tr, "test": te, "per_concept": per_c, "seed": seed})
    mean_te = np.mean([r["test"] for r in split_results])
    mean_tr = np.mean([r["train"] for r in split_results])
    print(f"  Split {si+1:2d} (orig_in_test={orig_in_test}, new_in_test={new_in_test}): "
          f"train={mean_tr:.0%}  test={mean_te:.0%}")
    results_b.extend(split_results)

all_train_b = [r["train"] for r in results_b]
all_test_b  = [r["test"]  for r in results_b]
print(f"\n  CONDITION B SUMMARY ({len(results_b)} runs):")
print(f"  Train: {np.mean(all_train_b):.1%} ± {np.std(all_train_b):.1%}")
print(f"  Test:  {np.mean(all_test_b):.1%}  ± {np.std(all_test_b):.1%}")
print(f"  Gap:   {np.mean(all_train_b) - np.mean(all_test_b):.1%}")

# ── Per-concept test agreement (Condition B) ──────────────────────────────────

print("\n" + "─" * 65)
print("PER-CONCEPT TEST AGREEMENT (Condition B, original concepts only)")
print("─" * 65)

concept_appearances_b = {}
concept_agreements_b  = {}
for r in results_b:
    for c, agreed in r["per_concept"].items():
        concept_appearances_b[c] = concept_appearances_b.get(c, 0) + 1
        concept_agreements_b[c]  = concept_agreements_b.get(c, 0)  + agreed

orig_test_agreements = {c: concept_agreements_b[c] / concept_appearances_b[c]
                        for c in orig_concepts if c in concept_agreements_b}
new_test_agreements  = {c: concept_agreements_b[c] / concept_appearances_b[c]
                        for c in [all_concepts[i] for i in new_idx]
                        if c in concept_agreements_b}

print("\n  Original concepts (when held out):")
for c, ag in sorted(orig_test_agreements.items(), key=lambda x: x[1]):
    print(f"    {c:<15} {ag:.0%}")

print("\n  New concepts (when held out):")
for c, ag in sorted(new_test_agreements.items(), key=lambda x: x[1]):
    print(f"    {c:<15} {ag:.0%}")

# ── Verdict ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("VERDICT")
print("=" * 65)
mean_te_a = np.mean(all_test_a)
mean_te_b = np.mean(all_test_b)
drop = mean_te_a - mean_te_b

print(f"  Condition A test agreement: {mean_te_a:.1%}")
print(f"  Condition B test agreement: {mean_te_b:.1%}")
print(f"  Drop from A → B:            {drop:.1%}")

if drop > 0.30:
    verdict = "BATCH ARTIFACT CONFIRMED — generalisation was spurious"
elif drop > 0.10:
    verdict = "PARTIAL ARTIFACT — batch identity explains some but not all generalisation"
else:
    verdict = "GENERALISATION IS REAL — batch balance does not affect result"
print(f"\n  {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────

output = {
    "condition_a_uncontrolled": {
        "train_mean": float(np.mean(all_train_a)),
        "train_std":  float(np.std(all_train_a)),
        "test_mean":  float(np.mean(all_test_a)),
        "test_std":   float(np.std(all_test_a)),
        "gap":        float(np.mean(all_train_a) - np.mean(all_test_a)),
        "n_runs":     len(results_a),
    },
    "condition_b_balanced": {
        "train_mean": float(np.mean(all_train_b)),
        "train_std":  float(np.std(all_train_b)),
        "test_mean":  float(np.mean(all_test_b)),
        "test_std":   float(np.std(all_test_b)),
        "gap":        float(np.mean(all_train_b) - np.mean(all_test_b)),
        "n_runs":     len(results_b),
    },
    "verdict": verdict,
    "drop_A_to_B": float(drop),
}
with open("lm_output/generalization_balanced_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/generalization_balanced_results.json")
