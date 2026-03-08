"""
Shared VQ Codebook: V-JEPA 2 <-> Mistral
-----------------------------------------
Diagnostic experiment: can a discrete codebook bridge two representational
spaces that RSA shows are geometrically orthogonal (r = -0.036)?

Architecture (minimal by design):
    V-JEPA2 [1024] -> Linear+LN -> [256] -> VQ(64x256) -> [256] -> Linear -> [1024]
    Mistral [4096] -> Linear+LN -> [256] -> VQ(64x256) -> [256] -> Linear -> [4096]

No cross-modal alignment loss. The codebook must discover shared structure
through reconstruction pressure alone.

Training data: 17 physical concepts x 30 augmented copies = 510 per modality.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

from extract_lm_standalone import ALL_CONCEPTS, CONCEPTS
from rsa import cosine_similarity_matrix, rsa_score

output_dir = Path("lm_output")
PHYSICAL = CONCEPTS["physical"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(n_augment=30, sigma=0.1):
    """Load real embeddings and generate augmented training set."""
    lm = np.load(output_dir / "lm_hiddens.npy")       # [71, 4096]
    vjepa2 = np.load(output_dir / "vjepa2_hiddens.npy")  # [71, 1024]

    # Get valid physical concept indices (non-zero in both)
    lm_norms = np.linalg.norm(lm, axis=-1)
    vj_norms = np.linalg.norm(vjepa2, axis=-1)
    valid_phys = [c for c in PHYSICAL
                  if lm_norms[ALL_CONCEPTS.index(c)] > 1e-8
                  and vj_norms[ALL_CONCEPTS.index(c)] > 1e-8]
    phys_idx = [ALL_CONCEPTS.index(c) for c in valid_phys]

    # Extract and L2-normalize base embeddings
    lm_base = lm[phys_idx]
    vj_base = vjepa2[phys_idx]
    lm_base = lm_base / np.linalg.norm(lm_base, axis=-1, keepdims=True)
    vj_base = vj_base / np.linalg.norm(vj_base, axis=-1, keepdims=True)

    n_concepts = len(valid_phys)

    # Augment: Gaussian noise + renormalize
    rng = np.random.default_rng(42)

    lm_aug = []
    vj_aug = []
    labels = []  # concept index for each sample

    for ci in range(n_concepts):
        for _ in range(n_augment):
            # Mistral
            noise_lm = rng.normal(0, sigma, lm_base.shape[1])
            v_lm = lm_base[ci] + noise_lm
            v_lm = v_lm / np.linalg.norm(v_lm)
            lm_aug.append(v_lm)

            # V-JEPA 2
            noise_vj = rng.normal(0, sigma, vj_base.shape[1])
            v_vj = vj_base[ci] + noise_vj
            v_vj = v_vj / np.linalg.norm(v_vj)
            vj_aug.append(v_vj)

            labels.append(ci)

    lm_aug = np.array(lm_aug, dtype=np.float32)
    vj_aug = np.array(vj_aug, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"Data: {n_concepts} concepts, {len(lm_aug)} samples per modality")
    print(f"  Mistral: {lm_aug.shape}, V-JEPA2: {vj_aug.shape}")

    return lm_aug, vj_aug, labels, lm_base, vj_base, valid_phys


# ── VQ Layer (EMA) ───────────────────────────────────────────────────────────

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        # Codebook
        self.register_buffer("embeddings", torch.randn(n_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(n_embeddings))
        self.register_buffer("ema_embed_sum", torch.randn(n_embeddings, embedding_dim))

        self._initialized = False

    def _init_from_data(self, z):
        """Initialize codebook from first batch of data (k-means++ style)."""
        if self._initialized:
            return
        n = min(z.shape[0], self.n_embeddings)
        indices = torch.randperm(z.shape[0])[:n]
        self.embeddings[:n] = z[indices].detach()
        self.ema_embed_sum[:n] = z[indices].detach()
        self.cluster_size[:n] = 1.0
        self._initialized = True

    def forward(self, z):
        # z: (batch, embedding_dim)
        self._init_from_data(z)

        # Distances to codebook entries
        d = (z.unsqueeze(1) - self.embeddings.unsqueeze(0)).pow(2).sum(-1)  # (B, K)
        encoding_indices = d.argmin(dim=1)  # (B,)

        # Quantized vectors (straight-through)
        z_q = self.embeddings[encoding_indices]

        # EMA update (training only)
        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()  # (B, K)
            self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            embed_sum = one_hot.T @ z.detach()  # (K, D)
            self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            # Laplace smoothing
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            self.embeddings.copy_(self.ema_embed_sum / cluster_size.unsqueeze(1))

        # Commitment loss
        commitment_loss = F.mse_loss(z, z_q.detach())

        # Straight-through estimator
        z_q_st = z + (z_q - z).detach()

        return z_q_st, commitment_loss, encoding_indices


# ── Shared Codebook Model ───────────────────────────────────────────────────

class SharedCodebook(nn.Module):
    def __init__(self, dim_vjepa=1024, dim_mistral=4096, codebook_dim=256,
                 n_codes=64):
        super().__init__()

        # Encoders: project to shared latent space
        self.enc_vjepa = nn.Sequential(
            nn.Linear(dim_vjepa, codebook_dim),
            nn.LayerNorm(codebook_dim),
        )
        self.enc_mistral = nn.Sequential(
            nn.Linear(dim_mistral, codebook_dim),
            nn.LayerNorm(codebook_dim),
        )

        # Shared VQ codebook
        self.vq = VectorQuantizerEMA(n_codes, codebook_dim)

        # Decoders: reconstruct from quantized codes
        self.dec_vjepa = nn.Linear(codebook_dim, dim_vjepa)
        self.dec_mistral = nn.Linear(codebook_dim, dim_mistral)

    def encode(self, x, modality):
        if modality == "vjepa":
            return self.enc_vjepa(x)
        else:
            return self.enc_mistral(x)

    def decode(self, z_q, modality):
        if modality == "vjepa":
            return self.dec_vjepa(z_q)
        else:
            return self.dec_mistral(z_q)

    def forward(self, x, modality):
        z_e = self.encode(x, modality)
        z_q, commit_loss, indices = self.vq(z_e)
        x_recon = self.decode(z_q, modality)
        return x_recon, commit_loss, indices, z_e, z_q


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, lm_data, vj_data, labels, n_epochs=200, batch_size=32,
          lr=1e-3, commitment_weight=0.25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lm_t = torch.tensor(lm_data, device=DEVICE)
    vj_t = torch.tensor(vj_data, device=DEVICE)
    labels_t = torch.tensor(labels, device=DEVICE)

    n_samples = lm_t.shape[0]
    rng = np.random.default_rng(123)

    history = {"epoch": [], "loss": [], "recon_lm": [], "recon_vj": [], "commit": []}

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)

        epoch_loss = 0.0
        epoch_recon_lm = 0.0
        epoch_recon_vj = 0.0
        epoch_commit = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            batch_lm = lm_t[idx]
            batch_vj = vj_t[idx]

            # Forward both modalities
            recon_lm, commit_lm, idx_lm, _, _ = model(batch_lm, "mistral")
            recon_vj, commit_vj, idx_vj, _, _ = model(batch_vj, "vjepa")

            # Losses
            loss_recon_lm = F.mse_loss(recon_lm, batch_lm)
            loss_recon_vj = F.mse_loss(recon_vj, batch_vj)
            loss_commit = (commit_lm + commit_vj) / 2

            loss = loss_recon_lm + loss_recon_vj + commitment_weight * loss_commit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_recon_lm += loss_recon_lm.item()
            epoch_recon_vj += loss_recon_vj.item()
            epoch_commit += loss_commit.item()
            n_batches += 1

        if (epoch + 1) % 20 == 0 or epoch == 0:
            avg = epoch_loss / n_batches
            avg_lm = epoch_recon_lm / n_batches
            avg_vj = epoch_recon_vj / n_batches
            avg_cm = epoch_commit / n_batches
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={avg:.4f} "
                  f"recon_lm={avg_lm:.4f} recon_vj={avg_vj:.4f} commit={avg_cm:.4f}")

        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss / n_batches)
        history["recon_lm"].append(epoch_recon_lm / n_batches)
        history["recon_vj"].append(epoch_recon_vj / n_batches)
        history["commit"].append(epoch_commit / n_batches)

    return history


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, lm_base, vj_base, valid_phys):
    """Evaluate on the 17 original (non-augmented) concept embeddings."""
    model.eval()

    lm_t = torch.tensor(lm_base, device=DEVICE)
    vj_t = torch.tensor(vj_base, device=DEVICE)

    with torch.no_grad():
        recon_lm, _, idx_lm, ze_lm, zq_lm = model(lm_t, "mistral")
        recon_vj, _, idx_vj, ze_vj, zq_vj = model(vj_t, "vjepa")

    idx_lm = idx_lm.cpu().numpy()
    idx_vj = idx_vj.cpu().numpy()

    n_concepts = len(valid_phys)
    results = {}

    # 1. Reconstruction quality
    mse_lm = F.mse_loss(recon_lm, lm_t).item()
    mse_vj = F.mse_loss(recon_vj, vj_t).item()

    # Cosine similarity between original and reconstructed
    cos_lm = F.cosine_similarity(recon_lm, lm_t, dim=1).mean().item()
    cos_vj = F.cosine_similarity(recon_vj, vj_t, dim=1).mean().item()

    print(f"\n  Reconstruction quality:")
    print(f"    Mistral: MSE={mse_lm:.4f}, cos_sim={cos_lm:.4f}")
    print(f"    V-JEPA2: MSE={mse_vj:.4f}, cos_sim={cos_vj:.4f}")

    results["reconstruction"] = {
        "mistral_mse": mse_lm, "mistral_cos": cos_lm,
        "vjepa2_mse": mse_vj, "vjepa2_cos": cos_vj,
    }

    # 2. Code utilization
    all_codes = np.concatenate([idx_lm, idx_vj])
    unique_codes = len(np.unique(all_codes))
    unique_lm = len(np.unique(idx_lm))
    unique_vj = len(np.unique(idx_vj))

    print(f"\n  Code utilization (of 64 total):")
    print(f"    Mistral uses: {unique_lm} codes")
    print(f"    V-JEPA2 uses: {unique_vj} codes")
    print(f"    Combined:     {unique_codes} codes")

    results["code_utilization"] = {
        "total_codes": 64,
        "mistral_unique": int(unique_lm),
        "vjepa2_unique": int(unique_vj),
        "combined_unique": int(unique_codes),
    }

    # 3. Cross-modal code agreement
    n_agree = int(np.sum(idx_lm == idx_vj))
    agree_rate = n_agree / n_concepts

    # Expected agreement by chance (given code distributions)
    from collections import Counter
    freq_lm = Counter(idx_lm)
    freq_vj = Counter(idx_vj)
    chance_agree = sum(
        (freq_lm.get(k, 0) / n_concepts) * (freq_vj.get(k, 0) / n_concepts)
        for k in set(list(freq_lm.keys()) + list(freq_vj.keys()))
    )

    print(f"\n  Cross-modal code agreement:")
    print(f"    Same code for same concept: {n_agree}/{n_concepts} ({agree_rate:.1%})")
    print(f"    Expected by chance:         {chance_agree:.1%}")

    # Per-concept breakdown
    print(f"\n    {'Concept':12s} | {'Mistral':>7s} | {'V-JEPA2':>7s} | {'Match':>5s}")
    print(f"    {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}")
    per_concept = {}
    for ci, concept in enumerate(valid_phys):
        match = "Y" if idx_lm[ci] == idx_vj[ci] else " "
        print(f"    {concept:12s} | {idx_lm[ci]:7d} | {idx_vj[ci]:7d} | {match:>5s}")
        per_concept[concept] = {
            "mistral_code": int(idx_lm[ci]),
            "vjepa2_code": int(idx_vj[ci]),
            "match": bool(idx_lm[ci] == idx_vj[ci]),
        }

    results["cross_modal_agreement"] = {
        "n_agree": n_agree,
        "n_total": n_concepts,
        "agreement_rate": agree_rate,
        "chance_agreement": chance_agree,
        "per_concept": per_concept,
    }

    # 4. RSA after quantization
    # Use the quantized vectors (in shared 256-dim space)
    zq_lm_np = zq_lm.cpu().float().numpy()
    zq_vj_np = zq_vj.cpu().float().numpy()

    rsm_lm_q = cosine_similarity_matrix(zq_lm_np)
    rsm_vj_q = cosine_similarity_matrix(zq_vj_np)

    r_post, p_post = rsa_score(rsm_lm_q, rsm_vj_q, "spearman")

    # Compare to pre-quantization (in shared space before VQ)
    ze_lm_np = ze_lm.cpu().float().numpy()
    ze_vj_np = ze_vj.cpu().float().numpy()

    rsm_lm_pre = cosine_similarity_matrix(ze_lm_np)
    rsm_vj_pre = cosine_similarity_matrix(ze_vj_np)

    r_pre, p_pre = rsa_score(rsm_lm_pre, rsm_vj_pre, "spearman")

    # Original space RSA (baseline)
    rsm_lm_orig = cosine_similarity_matrix(lm_base)
    rsm_vj_orig = cosine_similarity_matrix(vj_base)
    r_orig, p_orig = rsa_score(rsm_lm_orig, rsm_vj_orig, "spearman")

    print(f"\n  RSA (Mistral vs V-JEPA2 similarity structure):")
    print(f"    Original space:     r = {r_orig:+.4f}  (p = {p_orig:.2e})")
    print(f"    Pre-VQ (shared):    r = {r_pre:+.4f}  (p = {p_pre:.2e})")
    print(f"    Post-VQ (quantized): r = {r_post:+.4f}  (p = {p_post:.2e})")

    results["rsa"] = {
        "original": {"r": r_orig, "p": p_orig},
        "pre_vq_shared_space": {"r": r_pre, "p": p_pre},
        "post_vq_quantized": {"r": r_post, "p": p_post},
    }

    # 5. Within-modality RSA preservation
    # Does quantization preserve each modality's internal structure?
    r_lm_preserve, _ = rsa_score(rsm_lm_orig, rsm_lm_q, "spearman")
    r_vj_preserve, _ = rsa_score(rsm_vj_orig, rsm_vj_q, "spearman")

    # Pre-VQ preservation
    r_lm_pre_preserve, _ = rsa_score(rsm_lm_orig, rsm_lm_pre, "spearman")
    r_vj_pre_preserve, _ = rsa_score(rsm_vj_orig, rsm_vj_pre, "spearman")

    print(f"\n  Within-modality structure preservation:")
    print(f"    Mistral: original->pre_VQ r={r_lm_pre_preserve:+.4f}, original->post_VQ r={r_lm_preserve:+.4f}")
    print(f"    V-JEPA2: original->pre_VQ r={r_vj_pre_preserve:+.4f}, original->post_VQ r={r_vj_preserve:+.4f}")

    results["structure_preservation"] = {
        "mistral_pre_vq": r_lm_pre_preserve,
        "mistral_post_vq": r_lm_preserve,
        "vjepa2_pre_vq": r_vj_pre_preserve,
        "vjepa2_post_vq": r_vj_preserve,
    }

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("SHARED VQ CODEBOOK: V-JEPA 2 <-> Mistral 7B")
    print("=" * 65)

    # Prepare data
    lm_aug, vj_aug, labels, lm_base, vj_base, valid_phys = prepare_data(
        n_augment=30, sigma=0.1
    )

    # Build model
    model = SharedCodebook(
        dim_vjepa=1024,
        dim_mistral=4096,
        codebook_dim=256,
        n_codes=64,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(f"  Encoders: V-JEPA2 1024->256, Mistral 4096->256")
    print(f"  Codebook: 64 entries x 256-dim (EMA update)")
    print(f"  Decoders: 256->1024, 256->4096")

    # Train
    print(f"\nTraining (200 epochs, batch_size=32, lr=1e-3, commitment=0.25):")
    history = train(model, lm_aug, vj_aug, labels,
                    n_epochs=200, batch_size=32, lr=1e-3, commitment_weight=0.25)

    # Evaluate on original 17 concepts
    print(f"\n{'='*65}")
    print("EVALUATION ON ORIGINAL 17 CONCEPTS")
    print("=" * 65)
    results = evaluate(model, lm_base, vj_base, valid_phys)
    results["training_history"] = {
        "final_loss": history["loss"][-1],
        "final_recon_lm": history["recon_lm"][-1],
        "final_recon_vj": history["recon_vj"][-1],
        "final_commit": history["commit"][-1],
    }
    results["architecture"] = {
        "codebook_dim": 256,
        "n_codes": 64,
        "n_concepts": len(valid_phys),
        "concepts": valid_phys,
        "n_augment": 30,
        "sigma": 0.1,
        "n_epochs": 200,
        "lr": 1e-3,
        "commitment_weight": 0.25,
    }

    # Save
    def to_python(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_python(v) for v in obj]
        return obj

    with open(output_dir / "codebook_results.json", "w", encoding="utf-8") as f:
        json.dump(to_python(results), f, indent=2)

    # Save model checkpoint
    torch.save(model.state_dict(), output_dir / "codebook_model.pt")

    print(f"\n{'='*65}")
    print("SAVED")
    print("=" * 65)
    print(f"  {output_dir}/codebook_results.json")
    print(f"  {output_dir}/codebook_model.pt")

    # Final verdict
    agree = results["cross_modal_agreement"]["agreement_rate"]
    chance = results["cross_modal_agreement"]["chance_agreement"]
    r_post = results["rsa"]["post_vq_quantized"]["r"]
    r_orig = results["rsa"]["original"]["r"]

    print(f"\n{'='*65}")
    print("VERDICT")
    print("=" * 65)
    if agree > 0.5 and agree > chance * 2:
        print(f"  Cross-modal agreement {agree:.0%} >> chance {chance:.0%}")
        print(f"  SURPRISING: The codebook found shared structure despite orthogonal spaces.")
        print(f"  This suggests the linear projections can align the spaces before quantization.")
    elif agree > chance * 1.5:
        print(f"  Cross-modal agreement {agree:.0%} > chance {chance:.0%} (weak signal)")
        print(f"  Some concepts share codes, but most don't. Partial bridging at best.")
    else:
        print(f"  Cross-modal agreement {agree:.0%} ~ chance {chance:.0%}")
        print(f"  EXPECTED NULL: The codebook assigns different codes to same concepts.")
        print(f"  The orthogonal geometry (RSA r={r_orig:+.3f}) is real and unbridgeable")
        print(f"  by a simple shared bottleneck.")

    if abs(r_post) > abs(r_orig) + 0.1:
        print(f"\n  Post-VQ RSA ({r_post:+.3f}) > original ({r_orig:+.3f})")
        print(f"  Quantization *increased* cross-modal structural similarity.")
    elif abs(r_post) < abs(r_orig) - 0.1:
        print(f"\n  Post-VQ RSA ({r_post:+.3f}) < original ({r_orig:+.3f})")
        print(f"  Quantization degraded what little alignment existed.")
    else:
        print(f"\n  Post-VQ RSA ({r_post:+.3f}) ~ original ({r_orig:+.3f})")
        print(f"  Quantization neither helped nor hurt cross-modal alignment.")
