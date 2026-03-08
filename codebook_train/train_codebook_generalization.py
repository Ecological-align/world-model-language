"""
Held-out generalization test for the contrastive codebook.

Trains the codebook on 12 concepts, evaluates on 5 held-out concepts.
Tests whether the codebook discovers semantic structure or just memorizes concept IDs.

If generalization is good: the codebook is learning real cross-modal structure.
If generalization fails: the 99% agreement result was memorization, not alignment.

Run after train_codebook_contrastive_multiseed.py.
Results saved to lm_output/codebook_generalization_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import json
import random

# ── Config ──────────────────────────────────────────────────────────────────

LAMBDA       = 0.5          # optimal lambda from multi-seed experiment
CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 300
LR           = 1e-3
BATCH_SIZE   = 12           # train on all train concepts each batch
SEEDS        = [42, 123, 7, 99, 2025]
N_SPLITS     = 10           # number of random train/test splits to average over
N_TEST       = 5            # held-out concepts per split

OUTPUT_FILE  = "lm_output/codebook_generalization_results.json"

# ── Physical concepts (must match extraction order) ──────────────────────────

PHYSICAL_CONCEPTS = [
    "apple", "chair", "water", "fire", "stone",
    "rope", "door", "container", "shadow", "mirror",
    "knife", "wheel", "hand", "wall", "hole",
    "bridge", "ladder"
]

assert len(PHYSICAL_CONCEPTS) == 17

# ── VQ Codebook Architecture (same as contrastive scripts) ───────────────────

class VQCodebook(nn.Module):
    def __init__(self, num_codes, dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_codes, dim)
        nn.init.uniform_(self.embeddings.weight, -1/num_codes, 1/num_codes)

    def forward(self, z):
        # z: [B, dim]
        # returns: quantized, indices, commitment_loss
        dists = (
            z.pow(2).sum(1, keepdim=True)
            - 2 * z @ self.embeddings.weight.T
            + self.embeddings.weight.pow(2).sum(1)
        )
        indices = dists.argmin(1)
        quantized = self.embeddings(indices)
        # straight-through estimator
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
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


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


# ── NT-Xent Loss ─────────────────────────────────────────────────────────────

def nt_xent_loss(z_a, z_b, temperature=0.5):
    """Contrastive loss: pulls same-concept pairs together, pushes different apart."""
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    N = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)           # [2N, dim]
    sim = z @ z.T / temperature                  # [2N, 2N]
    # mask out self-similarity
    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    # positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ── RSA utility ──────────────────────────────────────────────────────────────

def spearman_r(a, b):
    """Spearman correlation between upper triangles of two RDMs."""
    from scipy.stats import spearmanr
    n = a.shape[0]
    idx = np.triu_indices(n, k=1)
    return spearmanr(a[idx], b[idx]).statistic


def build_rdm(vecs):
    """Cosine distance matrix."""
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim = vecs @ vecs.T
    return 1 - sim


# ── Training ─────────────────────────────────────────────────────────────────

def train_and_evaluate(st_train, vj_train, st_test, vj_test, seed, lam=LAMBDA):
    """
    Train codebook on train concepts, evaluate on held-out test concepts.
    Returns dict of metrics.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dim_st = st_train.shape[1]
    dim_vj = vj_train.shape[1]

    model = CrossModalCodebook(dim_st, dim_vj, CODEBOOK_DIM, NUM_CODES).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)

    x_st = torch.tensor(st_train, dtype=torch.float32).to(device)
    x_vj = torch.tensor(vj_train, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        z_a, z_b, q_a, q_b, idx_a, idx_b, recon_a, recon_b, comm_a, comm_b = model(x_st, x_vj)

        recon_loss = (
            F.mse_loss(recon_a, x_st) +
            F.mse_loss(recon_b, x_vj)
        )
        commit_loss = comm_a + comm_b
        contrast_loss = nt_xent_loss(z_a, z_b)

        loss = recon_loss + 0.25 * commit_loss + lam * contrast_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

    # ── Evaluate on TRAIN concepts ──
    model.eval()
    with torch.no_grad():
        _, _, _, _, idx_a_tr, idx_b_tr, recon_a_tr, recon_b_tr, _, _ = model(x_st, x_vj)

    idx_a_tr = idx_a_tr.cpu().numpy()
    idx_b_tr = idx_b_tr.cpu().numpy()
    train_agreement = np.mean(idx_a_tr == idx_b_tr)
    train_n_codes   = len(set(idx_a_tr) | set(idx_b_tr))

    recon_cos_tr = float(F.cosine_similarity(
        recon_a_tr, x_st).mean().item() +
        F.cosine_similarity(recon_b_tr, x_vj).mean().item()) / 2

    # ── Evaluate on TEST (held-out) concepts ──
    x_st_te = torch.tensor(st_test, dtype=torch.float32).to(device)
    x_vj_te = torch.tensor(vj_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, _, _, idx_a_te, idx_b_te, recon_a_te, recon_b_te, _, _ = model(x_st_te, x_vj_te)

    idx_a_te = idx_a_te.cpu().numpy()
    idx_b_te = idx_b_te.cpu().numpy()
    test_agreement = np.mean(idx_a_te == idx_b_te)
    test_n_codes   = len(set(idx_a_te) | set(idx_b_te))

    recon_cos_te = float(F.cosine_similarity(
        recon_a_te, x_st_te).mean().item() +
        F.cosine_similarity(recon_b_te, x_vj_te).mean().item()) / 2

    # ── Post-VQ RSA on test concepts ──
    if len(st_test) >= 4:
        rdm_st_te = build_rdm(st_test)
        rdm_vj_te = build_rdm(vj_test)
        rsa_pre = spearman_r(rdm_st_te, rdm_vj_te)

        # post-VQ: build RDM from code assignments
        # use one-hot codes as "representation"
        n_te = len(st_test)
        codes_a = np.eye(NUM_CODES)[idx_a_te]
        codes_b = np.eye(NUM_CODES)[idx_b_te]
        rdm_codes_a = build_rdm(codes_a)
        rdm_codes_b = build_rdm(codes_b)
        rsa_post_a = spearman_r(rdm_st_te, rdm_codes_a)
        rsa_post_b = spearman_r(rdm_vj_te, rdm_codes_b)
    else:
        rsa_pre = rsa_post_a = rsa_post_b = float('nan')

    return {
        "train_agreement":  float(train_agreement),
        "train_n_codes":    int(train_n_codes),
        "train_recon_cos":  float(recon_cos_tr),
        "test_agreement":   float(test_agreement),
        "test_n_codes":     int(test_n_codes),
        "test_recon_cos":   float(recon_cos_te),
        "rsa_pre_vq_test":  float(rsa_pre),
        "rsa_post_vq_a":    float(rsa_post_a),
        "rsa_post_vq_b":    float(rsa_post_b),
        "test_indices_st":  idx_a_te.tolist(),
        "test_indices_vj":  idx_b_te.tolist(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading representations...")
    st_all = np.load("lm_output/st_hiddens.npy")     # [71, 768]
    vj_all = np.load("lm_output/vjepa2_hiddens.npy") # [71, 1024]

    # Physical concepts are the first 17 rows (adjust if your ordering differs)
    # Verify by checking that the script used the same ordering
    st_phys = st_all[:17]
    vj_phys = vj_all[:17]

    print(f"ST shape: {st_phys.shape}, VJ shape: {vj_phys.shape}")
    print(f"Running {N_SPLITS} train/test splits × {len(SEEDS)} seeds = "
          f"{N_SPLITS * len(SEEDS)} total runs\n")

    all_results = []

    # Generate fixed random splits for reproducibility
    rng = random.Random(42)
    splits = []
    all_indices = list(range(17))
    seen_splits = set()
    attempts = 0
    while len(splits) < N_SPLITS and attempts < 1000:
        test_idx = tuple(sorted(rng.sample(all_indices, N_TEST)))
        if test_idx not in seen_splits:
            seen_splits.add(test_idx)
            train_idx = [i for i in all_indices if i not in test_idx]
            splits.append((train_idx, list(test_idx)))
        attempts += 1

    print("Splits:")
    for i, (train_idx, test_idx) in enumerate(splits):
        train_concepts = [PHYSICAL_CONCEPTS[j] for j in train_idx]
        test_concepts  = [PHYSICAL_CONCEPTS[j] for j in test_idx]
        print(f"  Split {i+1}: test = {test_concepts}")
    print()

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        train_concepts = [PHYSICAL_CONCEPTS[j] for j in train_idx]
        test_concepts  = [PHYSICAL_CONCEPTS[j] for j in test_idx]

        st_train = st_phys[train_idx]
        vj_train = vj_phys[train_idx]
        st_test  = st_phys[test_idx]
        vj_test  = vj_phys[test_idx]

        split_seed_results = []
        for seed in SEEDS:
            result = train_and_evaluate(st_train, vj_train, st_test, vj_test, seed)
            result["split"]          = split_idx + 1
            result["seed"]           = seed
            result["train_concepts"] = train_concepts
            result["test_concepts"]  = test_concepts
            split_seed_results.append(result)

            print(f"  Split {split_idx+1}/test={test_concepts} | seed={seed} | "
                  f"train_agree={result['train_agreement']:.0%} | "
                  f"test_agree={result['test_agreement']:.0%} | "
                  f"test_codes={result['test_n_codes']}")

        all_results.extend(split_seed_results)

        # Per-split summary
        train_agrees = [r["train_agreement"] for r in split_seed_results]
        test_agrees  = [r["test_agreement"]  for r in split_seed_results]
        print(f"  → Split {split_idx+1} summary: "
              f"train {np.mean(train_agrees):.0%}±{np.std(train_agrees):.0%} | "
              f"test  {np.mean(test_agrees):.0%}±{np.std(test_agrees):.0%}\n")

    # ── Global summary ──
    train_agrees_all = [r["train_agreement"] for r in all_results]
    test_agrees_all  = [r["test_agreement"]  for r in all_results]
    test_recon_all   = [r["test_recon_cos"]  for r in all_results]

    summary = {
        "lambda":                   LAMBDA,
        "n_splits":                 N_SPLITS,
        "n_seeds":                  len(SEEDS),
        "n_total_runs":             len(all_results),
        "train_agreement_mean":     float(np.mean(train_agrees_all)),
        "train_agreement_std":      float(np.std(train_agrees_all)),
        "test_agreement_mean":      float(np.mean(test_agrees_all)),
        "test_agreement_std":       float(np.std(test_agrees_all)),
        "test_recon_cos_mean":      float(np.mean(test_recon_all)),
        "test_recon_cos_std":       float(np.std(test_recon_all)),
        "generalization_gap":       float(np.mean(train_agrees_all) - np.mean(test_agrees_all)),
    }

    print("\n" + "="*60)
    print("GENERALIZATION RESULTS (λ=0.5)")
    print("="*60)
    print(f"Train agreement:  {summary['train_agreement_mean']:.1%} ± {summary['train_agreement_std']:.1%}")
    print(f"Test  agreement:  {summary['test_agreement_mean']:.1%} ± {summary['test_agreement_std']:.1%}")
    print(f"Generalization gap: {summary['generalization_gap']:.1%}")
    print(f"Test recon cos:   {summary['test_recon_cos_mean']:.3f} ± {summary['test_recon_cos_std']:.3f}")
    print()

    # Interpret
    gap = summary['generalization_gap']
    test_mean = summary['test_agreement_mean']
    if test_mean >= 0.80 and gap <= 0.15:
        verdict = "GENERALIZES — codebook learned real cross-modal structure, not memorization"
    elif test_mean >= 0.60 and gap <= 0.30:
        verdict = "PARTIAL GENERALIZATION — some structure learned, some memorization"
    else:
        verdict = "DOES NOT GENERALIZE — 99% train agreement was likely memorization"
    print(f"Verdict: {verdict}")
    print("="*60)

    # Save
    output = {"summary": summary, "per_run": all_results, "verdict": verdict}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
