"""
No-VQ baseline: does the VQ bottleneck actually add anything?

Compares three conditions:
  1. Raw cosine similarity (no training)
  2. Contrastive projection only (no VQ) — linear encoder + NT-Xent
  3. Contrastive + VQ (our full system, lambda=0.5)

For each condition, measures:
  - Cross-modal agreement (same-concept similarity rank)
  - RSA: Spearman correlation of similarity matrices

If VQ adds nothing, condition 2 ≈ condition 3.
If contrastive alone fixes the gap, condition 1 already works.

10 seeds each, 49 concepts.
Outputs: lm_output/novq_baseline_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from scipy.stats import spearmanr

# ── Config ────────────────────────────────────────────────────────────────────
CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 300
LR           = 1e-3
LAMBDA       = 0.5
SEEDS        = [42, 123, 7, 99, 2025, 11, 31, 55, 77, 88]

# ── Load ──────────────────────────────────────────────────────────────────────
st_all = np.load("lm_output/st_hiddens_expanded.npy")    # [49, 768]
vj_all = np.load("lm_output/vjepa2_hiddens_expanded.npy") # [49, 1024]
N = st_all.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, N={N} concepts")

# ── RSA utility ───────────────────────────────────────────────────────────────
def spearman_sim(vecs_a, vecs_b):
    """RSA: Spearman r between flattened upper-triangle of two cosine sim matrices."""
    def cosine_sim(v):
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return v @ v.T
    sim_a = cosine_sim(vecs_a)
    sim_b = cosine_sim(vecs_b)
    n = sim_a.shape[0]
    idx = np.triu_indices(n, k=1)
    r, p = spearmanr(sim_a[idx], sim_b[idx])
    return float(r)

# ── Cross-modal agreement ─────────────────────────────────────────────────────
def cross_modal_agreement(vecs_a, vecs_b):
    """Fraction of concepts where argmax cosine(a_i, b_j) = i (nearest neighbour)."""
    a = vecs_a / (np.linalg.norm(vecs_a, axis=1, keepdims=True) + 1e-8)
    b = vecs_b / (np.linalg.norm(vecs_b, axis=1, keepdims=True) + 1e-8)
    sim = a @ b.T  # [N, N]
    preds = sim.argmax(axis=1)
    return float(np.mean(preds == np.arange(N)))

# ── Condition 1: Raw (no training) ────────────────────────────────────────────
raw_rsa = spearman_sim(st_all, vj_all)
# NN agreement requires same dimensionality — project to shared dim via SVD
from sklearn.decomposition import TruncatedSVD
shared_dim = min(st_all.shape[1], vj_all.shape[1])
svd_a = TruncatedSVD(n_components=CODEBOOK_DIM, random_state=42).fit_transform(st_all)
svd_b = TruncatedSVD(n_components=CODEBOOK_DIM, random_state=42).fit_transform(vj_all)
raw_agreement = cross_modal_agreement(svd_a, svd_b)
print(f"\nCondition 1 — Raw (no training):")
print(f"  RSA (ST vs VJ): r = {raw_rsa:+.3f}")
print(f"  Cross-modal agreement (NN after SVD to {CODEBOOK_DIM}-dim): {raw_agreement:.1%}")

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


class Encoder(nn.Module):
    def __init__(self, input_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, out_dim), nn.LayerNorm(out_dim),
        )
    def forward(self, x): return self.net(x)


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


# ── Condition 2: Contrastive, no VQ ──────────────────────────────────────────
print(f"\nCondition 2 — Contrastive projection (no VQ), {len(SEEDS)} seeds:")
x_st = torch.tensor(st_all, dtype=torch.float32).to(device)
x_vj = torch.tensor(vj_all, dtype=torch.float32).to(device)

novq_agreements = []
novq_rsas = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    enc_a = Encoder(768,  CODEBOOK_DIM).to(device)
    enc_b = Encoder(1024, CODEBOOK_DIM).to(device)
    opt = torch.optim.Adam(list(enc_a.parameters()) + list(enc_b.parameters()), lr=LR)

    for epoch in range(EPOCHS):
        enc_a.train(); enc_b.train()
        z_a = enc_a(x_st)
        z_b = enc_b(x_vj)
        loss = LAMBDA * nt_xent_loss(z_a, z_b)
        opt.zero_grad(); loss.backward(); opt.step()

    enc_a.eval(); enc_b.eval()
    with torch.no_grad():
        z_a = enc_a(x_st).cpu().numpy()
        z_b = enc_b(x_vj).cpu().numpy()

    rsa = spearman_sim(z_a, z_b)
    agree = cross_modal_agreement(z_a, z_b)
    novq_agreements.append(agree)
    novq_rsas.append(rsa)
    print(f"  Seed {seed}: agreement={agree:.1%}, RSA={rsa:+.3f}")

print(f"  Mean: agreement={np.mean(novq_agreements):.1%}±{np.std(novq_agreements):.1%}, "
      f"RSA={np.mean(novq_rsas):+.3f}±{np.std(novq_rsas):.3f}")

# ── Condition 3: Full system (contrastive + VQ) ───────────────────────────────
print(f"\nCondition 3 — Contrastive + VQ (full system), {len(SEEDS)} seeds:")

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


vq_agreements = []
vq_rsas = []
vq_code_agreements = []

for seed in SEEDS:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CrossModalCodebook(768, 1024, CODEBOOK_DIM, NUM_CODES).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        z_a, z_b, q_a, q_b, idx_a, idx_b, ra, rb, ca, cb = model(x_st, x_vj)
        recon_loss = F.mse_loss(ra, x_st) + F.mse_loss(rb, x_vj)
        commit_loss = ca + cb
        contrast_loss = nt_xent_loss(z_a, z_b)
        loss = recon_loss + 0.25 * commit_loss + LAMBDA * contrast_loss
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z_a, z_b, q_a, q_b, idx_a, idx_b, _, _, _, _ = model(x_st, x_vj)

    # Pre-VQ continuous representations
    z_a_np = z_a.cpu().numpy()
    z_b_np = z_b.cpu().numpy()
    rsa = spearman_sim(z_a_np, z_b_np)
    agree = cross_modal_agreement(z_a_np, z_b_np)

    # Post-VQ code agreement
    idx_a_np = idx_a.cpu().numpy()
    idx_b_np = idx_b.cpu().numpy()
    code_agree = float(np.mean(idx_a_np == idx_b_np))

    vq_agreements.append(agree)
    vq_rsas.append(rsa)
    vq_code_agreements.append(code_agree)
    print(f"  Seed {seed}: nn_agree={agree:.1%}, RSA={rsa:+.3f}, code_agree={code_agree:.1%}")

print(f"  Mean: nn_agree={np.mean(vq_agreements):.1%}±{np.std(vq_agreements):.1%}, "
      f"RSA={np.mean(vq_rsas):+.3f}±{np.std(vq_rsas):.3f}, "
      f"code_agree={np.mean(vq_code_agreements):.1%}±{np.std(vq_code_agreements):.1%}")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SUMMARY: Does VQ add anything over contrastive-only?")
print("=" * 65)
print(f"{'Condition':<35} {'NN agree':>10} {'RSA':>8}")
print("-" * 55)
print(f"  Raw (no training){'':>18} {raw_agreement:>9.1%} {raw_rsa:>+8.3f}")
print(f"  Contrastive only (no VQ){'':>11} {np.mean(novq_agreements):>9.1%} {np.mean(novq_rsas):>+8.3f}")
print(f"  Contrastive + VQ (full){'':>12} {np.mean(vq_agreements):>9.1%} {np.mean(vq_rsas):>+8.3f}")
print()

vq_better_agree = np.mean(vq_agreements) > np.mean(novq_agreements) + 0.05
vq_better_rsa = np.mean(vq_rsas) > np.mean(novq_rsas) + 0.05

if vq_better_agree or vq_better_rsa:
    print("Verdict: VQ adds measurable value over contrastive-only projection")
elif abs(np.mean(vq_agreements) - np.mean(novq_agreements)) < 0.05:
    print("Verdict: VQ adds minimal value — contrastive loss does the heavy lifting")
else:
    print("Verdict: Contrastive-only outperforms full VQ system")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "raw": {"nn_agreement": raw_agreement, "rsa": raw_rsa},
    "contrastive_only": {
        "nn_agreement_mean": float(np.mean(novq_agreements)),
        "nn_agreement_std": float(np.std(novq_agreements)),
        "rsa_mean": float(np.mean(novq_rsas)),
        "rsa_std": float(np.std(novq_rsas)),
        "per_seed": [{"seed": s, "agreement": a, "rsa": r}
                     for s, a, r in zip(SEEDS, novq_agreements, novq_rsas)],
    },
    "full_vq": {
        "nn_agreement_mean": float(np.mean(vq_agreements)),
        "nn_agreement_std": float(np.std(vq_agreements)),
        "rsa_mean": float(np.mean(vq_rsas)),
        "rsa_std": float(np.std(vq_rsas)),
        "code_agreement_mean": float(np.mean(vq_code_agreements)),
        "code_agreement_std": float(np.std(vq_code_agreements)),
    },
}
with open("lm_output/novq_baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/novq_baseline_results.json")
