"""
Expanded generalization test: 49 concepts (17 original + 32 new).

Tests whether 3x more training data fixes the generalization failure
observed with 17 concepts (95% train, 2% test).

Design:
  - 49 concepts total from *_expanded.npy files
  - 75/25 split: 37 train, 12 test per split
  - 10 random splits x 5 seeds = 50 runs
  - Lambda=0.5 (optimal from multi-seed experiment)
  - ST (768-dim) <-> V-JEPA 2 (1024-dim)

Also reports per-concept difficulty and checks pre-registration predictions.

Results saved to lm_output/codebook_expanded_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from collections import defaultdict

# -- Config --

LAMBDA       = 0.5
CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 300
LR           = 1e-3
SEEDS        = [42, 123, 7, 99, 2025]
N_SPLITS     = 10
N_TEST       = 12          # 12 held-out concepts per split (~25%)

OUTPUT_FILE  = "lm_output/codebook_expanded_results.json"

# Load concept index
with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    concept_index = json.load(f)

ALL_CONCEPTS = concept_index["all_concepts"]
N_CONCEPTS   = len(ALL_CONCEPTS)
assert N_CONCEPTS == 49, f"Expected 49 concepts, got {N_CONCEPTS}"

# -- Architecture (same as contrastive scripts) --

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


# -- NT-Xent Loss --

def nt_xent_loss(z_a, z_b, temperature=0.5):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    N = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, labels)


# -- RSA utility --

def spearman_r(a, b):
    from scipy.stats import spearmanr
    n = a.shape[0]
    idx = np.triu_indices(n, k=1)
    return spearmanr(a[idx], b[idx]).statistic


def build_rdm(vecs):
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    sim = vecs @ vecs.T
    return 1 - sim


# -- Training --

def train_and_evaluate(st_train, vj_train, st_test, vj_test, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CrossModalCodebook(st_train.shape[1], vj_train.shape[1],
                               CODEBOOK_DIM, NUM_CODES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    x_st = torch.tensor(st_train, dtype=torch.float32).to(device)
    x_vj = torch.tensor(vj_train, dtype=torch.float32).to(device)

    for epoch in range(EPOCHS):
        model.train()
        z_a, z_b, q_a, q_b, idx_a, idx_b, recon_a, recon_b, comm_a, comm_b = model(x_st, x_vj)
        recon_loss = F.mse_loss(recon_a, x_st) + F.mse_loss(recon_b, x_vj)
        commit_loss = comm_a + comm_b
        contrast_loss = nt_xent_loss(z_a, z_b)
        loss = recon_loss + 0.25 * commit_loss + LAMBDA * contrast_loss
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate train
    model.eval()
    with torch.no_grad():
        _, _, _, _, idx_a_tr, idx_b_tr, recon_a_tr, recon_b_tr, _, _ = model(x_st, x_vj)

    idx_a_tr = idx_a_tr.cpu().numpy()
    idx_b_tr = idx_b_tr.cpu().numpy()
    train_agreement = float(np.mean(idx_a_tr == idx_b_tr))
    train_n_codes = len(set(idx_a_tr) | set(idx_b_tr))

    # Evaluate test
    x_st_te = torch.tensor(st_test, dtype=torch.float32).to(device)
    x_vj_te = torch.tensor(vj_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        _, _, _, _, idx_a_te, idx_b_te, recon_a_te, recon_b_te, _, _ = model(x_st_te, x_vj_te)

    idx_a_te = idx_a_te.cpu().numpy()
    idx_b_te = idx_b_te.cpu().numpy()
    test_agreement = float(np.mean(idx_a_te == idx_b_te))
    test_n_codes = len(set(idx_a_te) | set(idx_b_te))

    # Per-concept agreement on test set
    per_concept_agree = (idx_a_te == idx_b_te).astype(int).tolist()

    return {
        "train_agreement": train_agreement,
        "train_n_codes": train_n_codes,
        "test_agreement": test_agreement,
        "test_n_codes": test_n_codes,
        "test_per_concept": per_concept_agree,
        "test_indices_st": idx_a_te.tolist(),
        "test_indices_vj": idx_b_te.tolist(),
    }


# -- Main --

def main():
    print("Loading expanded representations...")
    st_all = np.load("lm_output/st_hiddens_expanded.npy")      # [49, 768]
    vj_all = np.load("lm_output/vjepa2_hiddens_expanded.npy")  # [49, 1024]

    print(f"ST shape: {st_all.shape}, VJ shape: {vj_all.shape}")
    print(f"Concepts: {N_CONCEPTS}")
    print(f"Running {N_SPLITS} splits x {len(SEEDS)} seeds = {N_SPLITS * len(SEEDS)} runs")
    print(f"Split: {N_CONCEPTS - N_TEST} train / {N_TEST} test\n")

    # Check for zero vectors (failed extractions)
    st_norms = np.linalg.norm(st_all, axis=1)
    vj_norms = np.linalg.norm(vj_all, axis=1)
    valid_mask = (st_norms > 0) & (vj_norms > 0)
    n_valid = valid_mask.sum()
    if n_valid < N_CONCEPTS:
        invalid = [ALL_CONCEPTS[i] for i in range(N_CONCEPTS) if not valid_mask[i]]
        print(f"WARNING: {N_CONCEPTS - n_valid} concepts have zero vectors: {invalid}")
        print(f"Using {n_valid} valid concepts only.\n")
    valid_indices = np.where(valid_mask)[0].tolist()
    valid_concepts = [ALL_CONCEPTS[i] for i in valid_indices]

    st_valid = st_all[valid_indices]
    vj_valid = vj_all[valid_indices]
    n_valid = len(valid_indices)

    # Generate random splits
    rng = random.Random(42)
    splits = []
    all_idx = list(range(n_valid))
    seen = set()
    attempts = 0
    while len(splits) < N_SPLITS and attempts < 1000:
        test_idx = tuple(sorted(rng.sample(all_idx, N_TEST)))
        if test_idx not in seen:
            seen.add(test_idx)
            train_idx = [i for i in all_idx if i not in test_idx]
            splits.append((train_idx, list(test_idx)))
        attempts += 1

    print("Splits:")
    for i, (train_idx, test_idx) in enumerate(splits):
        test_names = [valid_concepts[j] for j in test_idx]
        print(f"  Split {i+1}: test = {test_names}")
    print()

    all_results = []
    concept_agree_counts = defaultdict(list)  # concept -> list of agree/disagree

    for split_idx, (train_idx, test_idx) in enumerate(splits):
        test_names = [valid_concepts[j] for j in test_idx]
        train_names = [valid_concepts[j] for j in train_idx]

        st_train = st_valid[train_idx]
        vj_train = vj_valid[train_idx]
        st_test = st_valid[test_idx]
        vj_test = vj_valid[test_idx]

        for seed in SEEDS:
            result = train_and_evaluate(st_train, vj_train, st_test, vj_test, seed)
            result["split"] = split_idx + 1
            result["seed"] = seed
            result["test_concepts"] = test_names

            # Track per-concept agreement
            for j, name in enumerate(test_names):
                concept_agree_counts[name].append(result["test_per_concept"][j])

            all_results.append(result)

            print(f"  Split {split_idx+1} seed={seed} | "
                  f"train={result['train_agreement']:.0%} | "
                  f"test={result['test_agreement']:.0%} | "
                  f"codes={result['test_n_codes']}")

        # Per-split summary
        train_ag = [r["train_agreement"] for r in all_results[-len(SEEDS):]]
        test_ag = [r["test_agreement"] for r in all_results[-len(SEEDS):]]
        print(f"  -> Split {split_idx+1}: train {np.mean(train_ag):.0%}+/-{np.std(train_ag):.0%} | "
              f"test {np.mean(test_ag):.0%}+/-{np.std(test_ag):.0%}\n")

    # -- Global summary --
    train_agrees = [r["train_agreement"] for r in all_results]
    test_agrees = [r["test_agreement"] for r in all_results]

    # Per-concept difficulty
    concept_difficulty = {}
    for name, agrees in concept_agree_counts.items():
        concept_difficulty[name] = {
            "agreement_rate": float(np.mean(agrees)),
            "n_appearances": len(agrees),
        }
    # Sort by difficulty (lowest agreement first)
    difficulty_ranked = sorted(concept_difficulty.items(), key=lambda x: x[1]["agreement_rate"])

    # Load pre-registration predictions
    try:
        with open("lm_output/preregistration_expanded.json", "r", encoding="utf-8") as f:
            prereg = json.load(f)
        predicted_hard = prereg["predicted_hard"]
        predicted_easy = prereg["predicted_easy"]
    except FileNotFoundError:
        predicted_hard = []
        predicted_easy = []

    # Check pre-registration: are predicted-hard concepts actually harder?
    if predicted_hard and predicted_easy:
        hard_rates = [concept_difficulty[c]["agreement_rate"]
                      for c in predicted_hard if c in concept_difficulty]
        easy_rates = [concept_difficulty[c]["agreement_rate"]
                      for c in predicted_easy if c in concept_difficulty]
        prereg_check = {
            "hard_mean_agreement": float(np.mean(hard_rates)) if hard_rates else None,
            "easy_mean_agreement": float(np.mean(easy_rates)) if easy_rates else None,
            "prediction_correct": (float(np.mean(hard_rates)) < float(np.mean(easy_rates)))
                                  if hard_rates and easy_rates else None,
        }
    else:
        prereg_check = None

    summary = {
        "lambda": LAMBDA,
        "n_concepts": n_valid,
        "n_train": n_valid - N_TEST,
        "n_test": N_TEST,
        "n_splits": N_SPLITS,
        "n_seeds": len(SEEDS),
        "n_total_runs": len(all_results),
        "train_agreement_mean": float(np.mean(train_agrees)),
        "train_agreement_std": float(np.std(train_agrees)),
        "test_agreement_mean": float(np.mean(test_agrees)),
        "test_agreement_std": float(np.std(test_agrees)),
        "generalization_gap": float(np.mean(train_agrees) - np.mean(test_agrees)),
        "preregistration_check": prereg_check,
    }

    print("\n" + "=" * 60)
    print("EXPANDED GENERALIZATION RESULTS (49 concepts, lambda=0.5)")
    print("=" * 60)
    print(f"Train agreement:    {summary['train_agreement_mean']:.1%} +/- {summary['train_agreement_std']:.1%}")
    print(f"Test  agreement:    {summary['test_agreement_mean']:.1%} +/- {summary['test_agreement_std']:.1%}")
    print(f"Generalization gap: {summary['generalization_gap']:.1%}")
    print()

    # Compare with original 17-concept result
    print("Comparison with 17-concept experiment:")
    print(f"  17-concept: train 95% / test 2%  (gap 93%)")
    print(f"  49-concept: train {summary['train_agreement_mean']:.0%} / "
          f"test {summary['test_agreement_mean']:.0%}  "
          f"(gap {summary['generalization_gap']:.0%})")
    print()

    # Per-concept difficulty
    print("Per-concept difficulty (held-out agreement rate):")
    print("-" * 50)
    for name, stats in difficulty_ranked:
        tag = ""
        if name in predicted_hard:
            tag = " [predicted HARD]"
        elif name in predicted_easy:
            tag = " [predicted EASY]"
        print(f"  {name:15s}  {stats['agreement_rate']:.0%}  "
              f"(n={stats['n_appearances']}){tag}")
    print()

    if prereg_check:
        print("Pre-registration check:")
        print(f"  Predicted-hard mean agreement: {prereg_check['hard_mean_agreement']:.0%}")
        print(f"  Predicted-easy mean agreement: {prereg_check['easy_mean_agreement']:.0%}")
        print(f"  Prediction correct (hard < easy): {prereg_check['prediction_correct']}")
    print()

    # Verdict
    gap = summary['generalization_gap']
    test_mean = summary['test_agreement_mean']
    if test_mean >= 0.80 and gap <= 0.15:
        verdict = "GENERALIZES - more data solved the memorization problem"
    elif test_mean >= 0.60 and gap <= 0.30:
        verdict = "PARTIAL GENERALIZATION - more data helps but gap remains"
    elif test_mean >= 0.30:
        verdict = "WEAK GENERALIZATION - some signal but still largely memorizing"
    else:
        verdict = "DOES NOT GENERALIZE - even 49 concepts insufficient for cross-modal structure"
    print(f"Verdict: {verdict}")
    print("=" * 60)

    # Save
    output = {
        "summary": summary,
        "concept_difficulty": {name: stats for name, stats in difficulty_ranked},
        "per_run": all_results,
        "verdict": verdict,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
