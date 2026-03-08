"""
Code analysis: what do the 2 active codes represent?

Trains the codebook 5 times (5 seeds) on all 49 concepts,
then analyses which concepts go to code A vs code B.

Questions:
  1. Is the binary partition consistent across seeds?
  2. Does the partition correlate with any known semantic dimension?
     - animate vs inanimate
     - manipulable vs non-manipulable
     - dynamic vs static (high sensorimotor vs low)
     - polysemy (high WordNet senses vs low)
  3. What do the two codes actually mean intuitively?

Outputs: lm_output/code_analysis_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from collections import defaultdict, Counter

# ── Config ────────────────────────────────────────────────────────────────────
LAMBDA       = 0.5
CODEBOOK_DIM = 256
NUM_CODES    = 64
EPOCHS       = 300
LR           = 1e-3
SEEDS        = [42, 123, 7, 99, 2025]

# ── Load ──────────────────────────────────────────────────────────────────────
st_all = np.load("lm_output/st_hiddens_expanded.npy")   # [49, 768]
vj_all = np.load("lm_output/vjepa2_hiddens_expanded.npy") # [49, 1024]

with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)
concepts = idx["all_concepts"]
N = len(concepts)

# ── Semantic annotations ──────────────────────────────────────────────────────
# Hand-coded dimensions for the 49 concepts

animate = {
    "hand", "feather", "leaf",
}

# Manipulable: can be picked up and used as a tool
manipulable = {
    "apple", "knife", "rope", "ladder", "hammer", "scissors", "bowl",
    "bucket", "needle", "drum", "clock", "telescope", "coin", "shelf",
    "pipe", "chain", "thread", "glass", "feather", "leaf", "ice",
    "hand",
}

# Dynamic: inherently involves motion or change (high sensorimotor grounding)
dynamic = {
    "water", "fire", "wave", "spring", "run", "shoot", "strike", "press",
    "charge", "bark", "light", "wheel", "hand",
}

# Abstract/polysemous: primarily defined by metaphor or relation, hard to ground visually
abstract_poly = {
    "field", "net", "hole", "shadow", "charge", "light", "spring",
    "run", "press", "bark", "wave", "strike",
}

# ── Architecture (same as training scripts) ───────────────────────────────────

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
    N = z_a.shape[0]
    z = torch.cat([z_a, z_b], dim=0)
    sim = z @ z.T / temperature
    mask = torch.eye(2*N, device=z.device).bool()
    sim.masked_fill_(mask, float('-inf'))
    labels = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, labels)


# ── Train and collect codes ───────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_st = torch.tensor(st_all, dtype=torch.float32).to(device)
x_vj = torch.tensor(vj_all, dtype=torch.float32).to(device)

# For each seed: record which code each concept gets (both modalities)
# Then map to canonical 0/1 (the two active codes)
all_assignments = []  # [seed, concept, modality] -> canonical code (0 or 1)

print("Training 5 seeds to collect code assignments...")
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
        _, _, _, _, idx_a, idx_b, _, _, _, _ = model(x_st, x_vj)
    idx_a = idx_a.cpu().numpy()
    idx_b = idx_b.cpu().numpy()

    # Find the two active codes
    active = sorted(set(idx_a.tolist()) | set(idx_b.tolist()))
    code_map = {c: i for i, c in enumerate(active)}
    mapped_a = np.array([code_map[c] for c in idx_a])
    mapped_b = np.array([code_map[c] for c in idx_b])

    agreement = float(np.mean(idx_a == idx_b))
    n_active = len(active)
    print(f"  Seed {seed}: agreement={agreement:.0%}, active_codes={n_active}, codes={active[:5]}")

    all_assignments.append({"st": mapped_a.tolist(), "vj": mapped_b.tolist(),
                             "raw_st": idx_a.tolist(), "raw_vj": idx_b.tolist()})


# ── Analyse consistency ───────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("CODE ASSIGNMENT ANALYSIS")
print("=" * 65)

# For each concept: what fraction of seeds assign it to code 0 (vs 1)?
# (code 0 = the majority code for that concept across seeds)
concept_code_votes = []  # [N] -> fraction of seeds that put concept in group 0

for ci in range(N):
    votes = []
    for seed_data in all_assignments:
        # use ST assignment (both should agree at 97%+)
        votes.append(seed_data["st"][ci])
    # Majority vote determines group
    majority = Counter(votes).most_common(1)[0][0]
    fraction_majority = votes.count(majority) / len(votes)
    concept_code_votes.append((concepts[ci], majority, fraction_majority, votes))

# Sort by consistency
concept_code_votes.sort(key=lambda x: -x[2])

# Split into two groups based on majority assignment
group0 = [(c, frac) for c, code, frac, _ in concept_code_votes if code == 0]
group1 = [(c, frac) for c, code, frac, _ in concept_code_votes if code == 1]

# Make group0 the larger group
if len(group1) > len(group0):
    group0, group1 = group1, group0

print(f"\nGroup A ({len(group0)} concepts):")
for c, frac in sorted(group0):
    consistency = f"{frac:.0%}" if frac < 1.0 else "100%"
    ann = []
    if c in dynamic: ann.append("dynamic")
    if c in manipulable: ann.append("manip")
    if c in abstract_poly: ann.append("abstract")
    ann_str = ", ".join(ann) if ann else ""
    print(f"  {c:<15} {consistency:>5}  {ann_str}")

print(f"\nGroup B ({len(group1)} concepts):")
for c, frac in sorted(group1):
    consistency = f"{frac:.0%}" if frac < 1.0 else "100%"
    ann = []
    if c in dynamic: ann.append("dynamic")
    if c in manipulable: ann.append("manip")
    if c in abstract_poly: ann.append("abstract")
    ann_str = ", ".join(ann) if ann else ""
    print(f"  {c:<15} {consistency:>5}  {ann_str}")

# ── Check semantic dimensions ─────────────────────────────────────────────────

print("\n" + "─" * 65)
print("SEMANTIC DIMENSION ANALYSIS")
print("─" * 65)

group0_names = {c for c, _ in group0}
group1_names = {c for c, _ in group1}

for dim_name, dim_set in [
    ("dynamic", dynamic),
    ("manipulable", manipulable),
    ("abstract/polysemous", abstract_poly),
]:
    in_g0 = len(dim_set & group0_names)
    in_g1 = len(dim_set & group1_names)
    total = len(dim_set)
    print(f"\n  {dim_name} ({total} concepts):")
    print(f"    Group A: {in_g0}/{total} ({in_g0/total:.0%})")
    print(f"    Group B: {in_g1}/{total} ({in_g1/total:.0%})")
    if in_g0 > in_g1 * 1.5:
        print(f"    → Group A is more {dim_name}")
    elif in_g1 > in_g0 * 1.5:
        print(f"    → Group B is more {dim_name}")
    else:
        print(f"    → No clear association")

# ── Cross-seed consistency overall ───────────────────────────────────────────

consistencies = [frac for _, _, frac, _ in concept_code_votes]
print(f"\n  Mean cross-seed consistency: {np.mean(consistencies):.2f}")
print(f"  Concepts with 100% consistency: {sum(1 for f in consistencies if f == 1.0)}/{N}")
print(f"  Concepts with <80% consistency: {sum(1 for f in consistencies if f < 0.8)}/{N}")

# Most ambiguous concepts
ambiguous = [(c, frac) for c, _, frac, _ in concept_code_votes if frac < 1.0]
if ambiguous:
    print(f"\n  Ambiguous assignments (< 100% consistent):")
    for c, frac in sorted(ambiguous, key=lambda x: x[1]):
        print(f"    {c:<15} {frac:.0%}")

# ── Save ──────────────────────────────────────────────────────────────────────

output = {
    "group_a": [c for c, _ in group0],
    "group_b": [c for c, _ in group1],
    "concept_consistency": {c: frac for c, _, frac, _ in concept_code_votes},
    "semantic_analysis": {
        "dynamic_in_a": list(dynamic & group0_names),
        "dynamic_in_b": list(dynamic & group1_names),
        "manipulable_in_a": list(manipulable & group0_names),
        "manipulable_in_b": list(manipulable & group1_names),
        "abstract_in_a": list(abstract_poly & group0_names),
        "abstract_in_b": list(abstract_poly & group1_names),
    }
}
with open("lm_output/code_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/code_analysis_results.json")
