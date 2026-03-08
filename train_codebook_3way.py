"""
Three-Way VQ Codebook: V-JEPA 2 + CLIP + Mistral
--------------------------------------------------
Hypothesis: CLIP bridges V-JEPA 2 and Mistral because it was trained on
language-image pairs — it partially encodes both visual and linguistic structure.

V-JEPA2 [1024] -> Linear+LN -> [256] -> |
CLIP    [768]  -> Linear+LN -> [256] -> |-> VQ(64x256) -> |-> Linear -> [1024]
Mistral [4096] -> Linear+LN -> [256] -> |                 |-> Linear -> [768]
                                                           |-> Linear -> [4096]

No cross-modal alignment loss. All three modalities in every batch.
Compare against two-way (V-JEPA2 <-> Mistral) results.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter

from extract_lm_standalone import ALL_CONCEPTS, CONCEPTS
from rsa import cosine_similarity_matrix, rsa_score

output_dir = Path("lm_output")
PHYSICAL = CONCEPTS["physical"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── Data preparation ─────────────────────────────────────────────────────────

def prepare_data(n_augment=30, sigma=0.1):
    lm = np.load(output_dir / "lm_hiddens.npy")
    vjepa2 = np.load(output_dir / "vjepa2_hiddens.npy")
    clip = np.load(output_dir / "clip_hiddens.npy")

    # Valid physical: non-zero in all three
    lm_n = np.linalg.norm(lm, axis=-1)
    vj_n = np.linalg.norm(vjepa2, axis=-1)
    cl_n = np.linalg.norm(clip, axis=-1)

    valid_phys = [c for c in PHYSICAL
                  if lm_n[ALL_CONCEPTS.index(c)] > 1e-8
                  and vj_n[ALL_CONCEPTS.index(c)] > 1e-8
                  and cl_n[ALL_CONCEPTS.index(c)] > 1e-8]
    phys_idx = [ALL_CONCEPTS.index(c) for c in valid_phys]

    # Extract and L2-normalize
    lm_base = lm[phys_idx]
    vj_base = vjepa2[phys_idx]
    cl_base = clip[phys_idx]
    lm_base = lm_base / np.linalg.norm(lm_base, axis=-1, keepdims=True)
    vj_base = vj_base / np.linalg.norm(vj_base, axis=-1, keepdims=True)
    cl_base = cl_base / np.linalg.norm(cl_base, axis=-1, keepdims=True)

    n_concepts = len(valid_phys)

    # Geometry diagnostic: mean pairwise cosine similarity
    print(f"\n  GEOMETRY DIAGNOSTIC (mean pairwise cosine similarity):")
    for name, base in [("Mistral", lm_base), ("V-JEPA2", vj_base), ("CLIP", cl_base)]:
        rsm = base @ base.T
        n = rsm.shape[0]
        triu = rsm[np.triu_indices(n, k=1)]
        print(f"    {name:8s}: mean={triu.mean():.4f}, std={triu.std():.4f}, "
              f"min={triu.min():.4f}, max={triu.max():.4f}")
        # Effective spread: how far apart are the most distant concepts?
        off_diag = rsm[np.triu_indices(n, k=1)]
        spread = 1.0 - off_diag.mean()  # 0 = all identical, 1 = orthogonal
        print(f"             spread (1 - mean_sim) = {spread:.4f}")
    print()

    # Augment
    rng = np.random.default_rng(42)

    lm_aug, vj_aug, cl_aug, labels = [], [], [], []

    for ci in range(n_concepts):
        for _ in range(n_augment):
            noise_lm = rng.normal(0, sigma, lm_base.shape[1])
            v_lm = lm_base[ci] + noise_lm
            lm_aug.append(v_lm / np.linalg.norm(v_lm))

            noise_vj = rng.normal(0, sigma, vj_base.shape[1])
            v_vj = vj_base[ci] + noise_vj
            vj_aug.append(v_vj / np.linalg.norm(v_vj))

            noise_cl = rng.normal(0, sigma, cl_base.shape[1])
            v_cl = cl_base[ci] + noise_cl
            cl_aug.append(v_cl / np.linalg.norm(v_cl))

            labels.append(ci)

    lm_aug = np.array(lm_aug, dtype=np.float32)
    vj_aug = np.array(vj_aug, dtype=np.float32)
    cl_aug = np.array(cl_aug, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    print(f"  Data: {n_concepts} concepts, {len(lm_aug)} samples per modality")

    return lm_aug, vj_aug, cl_aug, labels, lm_base, vj_base, cl_base, valid_phys


# ── VQ Layer (EMA) ───────────────────────────────────────────────────────────

class VectorQuantizerEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, decay=0.99, epsilon=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.epsilon = epsilon

        self.register_buffer("embeddings", torch.randn(n_embeddings, embedding_dim))
        self.register_buffer("cluster_size", torch.zeros(n_embeddings))
        self.register_buffer("ema_embed_sum", torch.randn(n_embeddings, embedding_dim))
        self._initialized = False

    def _init_from_data(self, z):
        if self._initialized:
            return
        n = min(z.shape[0], self.n_embeddings)
        indices = torch.randperm(z.shape[0])[:n]
        self.embeddings[:n] = z[indices].detach()
        self.ema_embed_sum[:n] = z[indices].detach()
        self.cluster_size[:n] = 1.0
        self._initialized = True

    def forward(self, z):
        self._init_from_data(z)

        d = (z.unsqueeze(1) - self.embeddings.unsqueeze(0)).pow(2).sum(-1)
        encoding_indices = d.argmin(dim=1)

        z_q = self.embeddings[encoding_indices]

        if self.training:
            one_hot = F.one_hot(encoding_indices, self.n_embeddings).float()
            self.cluster_size.mul_(self.decay).add_(one_hot.sum(0), alpha=1 - self.decay)
            embed_sum = one_hot.T @ z.detach()
            self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.epsilon) / (n + self.n_embeddings * self.epsilon) * n
            self.embeddings.copy_(self.ema_embed_sum / cluster_size.unsqueeze(1))

        commitment_loss = F.mse_loss(z, z_q.detach())
        z_q_st = z + (z_q - z).detach()

        return z_q_st, commitment_loss, encoding_indices


# ── Three-Way Shared Codebook ───────────────────────────────────────────────

class SharedCodebook3Way(nn.Module):
    def __init__(self, dim_vjepa=1024, dim_clip=768, dim_mistral=4096,
                 codebook_dim=256, n_codes=64):
        super().__init__()

        self.enc_vjepa = nn.Sequential(
            nn.Linear(dim_vjepa, codebook_dim), nn.LayerNorm(codebook_dim))
        self.enc_clip = nn.Sequential(
            nn.Linear(dim_clip, codebook_dim), nn.LayerNorm(codebook_dim))
        self.enc_mistral = nn.Sequential(
            nn.Linear(dim_mistral, codebook_dim), nn.LayerNorm(codebook_dim))

        self.vq = VectorQuantizerEMA(n_codes, codebook_dim)

        self.dec_vjepa = nn.Linear(codebook_dim, dim_vjepa)
        self.dec_clip = nn.Linear(codebook_dim, dim_clip)
        self.dec_mistral = nn.Linear(codebook_dim, dim_mistral)

    def forward(self, x, modality):
        if modality == "vjepa":
            z_e = self.enc_vjepa(x)
        elif modality == "clip":
            z_e = self.enc_clip(x)
        else:
            z_e = self.enc_mistral(x)

        z_q, commit_loss, indices = self.vq(z_e)

        if modality == "vjepa":
            x_recon = self.dec_vjepa(z_q)
        elif modality == "clip":
            x_recon = self.dec_clip(z_q)
        else:
            x_recon = self.dec_mistral(z_q)

        return x_recon, commit_loss, indices, z_e, z_q


# ── Training ─────────────────────────────────────────────────────────────────

def train(model, lm_data, vj_data, cl_data, labels, n_epochs=200,
          batch_size=32, lr=1e-3, commitment_weight=0.25):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    lm_t = torch.tensor(lm_data, device=DEVICE)
    vj_t = torch.tensor(vj_data, device=DEVICE)
    cl_t = torch.tensor(cl_data, device=DEVICE)

    n_samples = lm_t.shape[0]
    rng = np.random.default_rng(123)

    history = []

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)

        ep = {"recon_lm": 0, "recon_vj": 0, "recon_cl": 0, "commit": 0, "loss": 0}
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]

            recon_lm, cm_lm, _, _, _ = model(lm_t[idx], "mistral")
            recon_vj, cm_vj, _, _, _ = model(vj_t[idx], "vjepa")
            recon_cl, cm_cl, _, _, _ = model(cl_t[idx], "clip")

            loss_lm = F.mse_loss(recon_lm, lm_t[idx])
            loss_vj = F.mse_loss(recon_vj, vj_t[idx])
            loss_cl = F.mse_loss(recon_cl, cl_t[idx])
            loss_commit = (cm_lm + cm_vj + cm_cl) / 3

            loss = loss_lm + loss_vj + loss_cl + commitment_weight * loss_commit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ep["recon_lm"] += loss_lm.item()
            ep["recon_vj"] += loss_vj.item()
            ep["recon_cl"] += loss_cl.item()
            ep["commit"] += loss_commit.item()
            ep["loss"] += loss.item()
            n_batches += 1

        for k in ep:
            ep[k] /= n_batches
        history.append(ep)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: loss={ep['loss']:.4f} "
                  f"lm={ep['recon_lm']:.4f} vj={ep['recon_vj']:.4f} "
                  f"cl={ep['recon_cl']:.4f} commit={ep['commit']:.4f}")

    return history


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, lm_base, vj_base, cl_base, valid_phys):
    model.eval()

    lm_t = torch.tensor(lm_base, device=DEVICE, dtype=torch.float32)
    vj_t = torch.tensor(vj_base, device=DEVICE, dtype=torch.float32)
    cl_t = torch.tensor(cl_base, device=DEVICE, dtype=torch.float32)

    with torch.no_grad():
        recon_lm, _, idx_lm, ze_lm, zq_lm = model(lm_t, "mistral")
        recon_vj, _, idx_vj, ze_vj, zq_vj = model(vj_t, "vjepa")
        recon_cl, _, idx_cl, ze_cl, zq_cl = model(cl_t, "clip")

    idx_lm = idx_lm.cpu().numpy()
    idx_vj = idx_vj.cpu().numpy()
    idx_cl = idx_cl.cpu().numpy()

    n = len(valid_phys)
    results = {}

    # 1. Reconstruction quality
    mse_lm = F.mse_loss(recon_lm, lm_t).item()
    mse_vj = F.mse_loss(recon_vj, vj_t).item()
    mse_cl = F.mse_loss(recon_cl, cl_t).item()
    cos_lm = F.cosine_similarity(recon_lm, lm_t, dim=1).mean().item()
    cos_vj = F.cosine_similarity(recon_vj, vj_t, dim=1).mean().item()
    cos_cl = F.cosine_similarity(recon_cl, cl_t, dim=1).mean().item()

    print(f"\n  Reconstruction quality:")
    print(f"    Mistral: MSE={mse_lm:.4f}, cos_sim={cos_lm:.4f}")
    print(f"    V-JEPA2: MSE={mse_vj:.4f}, cos_sim={cos_vj:.4f}")
    print(f"    CLIP:    MSE={mse_cl:.4f}, cos_sim={cos_cl:.4f}")

    results["reconstruction"] = {
        "mistral": {"mse": mse_lm, "cos": cos_lm},
        "vjepa2": {"mse": mse_vj, "cos": cos_vj},
        "clip": {"mse": mse_cl, "cos": cos_cl},
    }

    # 2. Code utilization
    unique_lm = len(np.unique(idx_lm))
    unique_vj = len(np.unique(idx_vj))
    unique_cl = len(np.unique(idx_cl))
    all_codes = np.concatenate([idx_lm, idx_vj, idx_cl])
    unique_all = len(np.unique(all_codes))

    print(f"\n  Code utilization (of 64 total):")
    print(f"    Mistral: {unique_lm} codes")
    print(f"    V-JEPA2: {unique_vj} codes")
    print(f"    CLIP:    {unique_cl} codes")
    print(f"    Combined: {unique_all} codes")

    results["code_utilization"] = {
        "total_codes": 64,
        "mistral": int(unique_lm), "vjepa2": int(unique_vj),
        "clip": int(unique_cl), "combined": int(unique_all),
    }

    # 3. Cross-modal code agreement
    pairs = [("V-JEPA2", "CLIP", idx_vj, idx_cl),
             ("Mistral", "CLIP", idx_lm, idx_cl),
             ("V-JEPA2", "Mistral", idx_vj, idx_lm)]

    print(f"\n  Cross-modal code agreement:")
    results["cross_modal_agreement"] = {}

    for name_a, name_b, codes_a, codes_b in pairs:
        n_agree = int(np.sum(codes_a == codes_b))
        rate = n_agree / n

        # Chance agreement
        freq_a = Counter(codes_a)
        freq_b = Counter(codes_b)
        chance = sum(
            (freq_a.get(k, 0) / n) * (freq_b.get(k, 0) / n)
            for k in set(list(freq_a.keys()) + list(freq_b.keys()))
        )

        print(f"    {name_a:8s} <-> {name_b:8s}: {n_agree}/{n} ({rate:.1%}), "
              f"chance={chance:.1%}")

        results["cross_modal_agreement"][f"{name_a}_vs_{name_b}"] = {
            "n_agree": n_agree, "n_total": n,
            "rate": rate, "chance": chance,
        }

    # Per-concept code table
    print(f"\n    {'Concept':12s} | {'Mistral':>7s} | {'CLIP':>7s} | {'V-JEPA2':>7s} | VJ=CL | LM=CL | VJ=LM")
    print(f"    {'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-------+-------+------")
    per_concept = {}
    for ci, concept in enumerate(valid_phys):
        vc = "Y" if idx_vj[ci] == idx_cl[ci] else " "
        lc = "Y" if idx_lm[ci] == idx_cl[ci] else " "
        vl = "Y" if idx_vj[ci] == idx_lm[ci] else " "
        print(f"    {concept:12s} | {idx_lm[ci]:7d} | {idx_cl[ci]:7d} | {idx_vj[ci]:7d} |"
              f"   {vc}   |   {lc}   |   {vl}")
        per_concept[concept] = {
            "mistral": int(idx_lm[ci]), "clip": int(idx_cl[ci]),
            "vjepa2": int(idx_vj[ci]),
            "vjepa2_clip_match": bool(idx_vj[ci] == idx_cl[ci]),
            "mistral_clip_match": bool(idx_lm[ci] == idx_cl[ci]),
            "vjepa2_mistral_match": bool(idx_vj[ci] == idx_lm[ci]),
        }
    results["per_concept"] = per_concept

    # 4. RSA after quantization — all three pairs
    zq_lm_np = zq_lm.cpu().float().numpy()
    zq_vj_np = zq_vj.cpu().float().numpy()
    zq_cl_np = zq_cl.cpu().float().numpy()
    ze_lm_np = ze_lm.cpu().float().numpy()
    ze_vj_np = ze_vj.cpu().float().numpy()
    ze_cl_np = ze_cl.cpu().float().numpy()

    # RSMs
    rsm_lm_orig = cosine_similarity_matrix(lm_base)
    rsm_vj_orig = cosine_similarity_matrix(vj_base)
    rsm_cl_orig = cosine_similarity_matrix(cl_base)
    rsm_lm_pre = cosine_similarity_matrix(ze_lm_np)
    rsm_vj_pre = cosine_similarity_matrix(ze_vj_np)
    rsm_cl_pre = cosine_similarity_matrix(ze_cl_np)
    rsm_lm_q = cosine_similarity_matrix(zq_lm_np)
    rsm_vj_q = cosine_similarity_matrix(zq_vj_np)
    rsm_cl_q = cosine_similarity_matrix(zq_cl_np)

    def safe_rsa(a, b):
        """RSA that handles constant inputs (from codebook collapse)."""
        try:
            r, p = rsa_score(a, b, "spearman")
            if np.isnan(r):
                return 0.0, 1.0
            return r, p
        except Exception:
            return 0.0, 1.0

    rsa_pairs = [
        ("V-JEPA2", "CLIP", rsm_vj_orig, rsm_cl_orig, rsm_vj_pre, rsm_cl_pre, rsm_vj_q, rsm_cl_q),
        ("Mistral", "CLIP", rsm_lm_orig, rsm_cl_orig, rsm_lm_pre, rsm_cl_pre, rsm_lm_q, rsm_cl_q),
        ("V-JEPA2", "Mistral", rsm_vj_orig, rsm_lm_orig, rsm_vj_pre, rsm_lm_pre, rsm_vj_q, rsm_lm_q),
    ]

    print(f"\n  RSA across stages:")
    print(f"    {'Pair':22s} | {'Original':>10s} | {'Pre-VQ':>10s} | {'Post-VQ':>10s}")
    print(f"    {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results["rsa"] = {}
    for name_a, name_b, orig_a, orig_b, pre_a, pre_b, q_a, q_b in rsa_pairs:
        r_orig, p_orig = safe_rsa(orig_a, orig_b)
        r_pre, p_pre = safe_rsa(pre_a, pre_b)
        r_post, p_post = safe_rsa(q_a, q_b)

        sig_orig = "*" if p_orig < 0.05 else " "
        sig_pre = "*" if p_pre < 0.05 else " "
        sig_post = "*" if p_post < 0.05 else " "

        pair_name = f"{name_a} vs {name_b}"
        print(f"    {pair_name:22s} | {r_orig:+.4f}{sig_orig}   | {r_pre:+.4f}{sig_pre}   | {r_post:+.4f}{sig_post}")

        results["rsa"][f"{name_a}_vs_{name_b}"] = {
            "original": {"r": float(r_orig), "p": float(p_orig)},
            "pre_vq": {"r": float(r_pre), "p": float(p_pre)},
            "post_vq": {"r": float(r_post), "p": float(p_post)},
        }

    # 5. Within-modality structure preservation
    print(f"\n  Within-modality structure preservation (original -> post_VQ):")
    results["structure_preservation"] = {}
    for name, rsm_o, rsm_p, rsm_q in [
        ("Mistral", rsm_lm_orig, rsm_lm_pre, rsm_lm_q),
        ("V-JEPA2", rsm_vj_orig, rsm_vj_pre, rsm_vj_q),
        ("CLIP",    rsm_cl_orig, rsm_cl_pre, rsm_cl_q),
    ]:
        r_pre, _ = safe_rsa(rsm_o, rsm_p)
        r_post, _ = safe_rsa(rsm_o, rsm_q)
        print(f"    {name:8s}: pre_VQ r={r_pre:+.4f}, post_VQ r={r_post:+.4f}")
        results["structure_preservation"][name] = {
            "pre_vq": float(r_pre), "post_vq": float(r_post),
        }

    return results


# ── Comparison with two-way ──────────────────────────────────────────────────

def print_comparison(results_3way):
    """Print side-by-side comparison with two-way results."""
    try:
        with open(output_dir / "codebook_results.json", "r") as f:
            results_2way = json.load(f)
    except FileNotFoundError:
        print("\n  (No two-way results found for comparison)")
        return

    print(f"\n{'='*72}")
    print("COMPARISON: TWO-WAY vs THREE-WAY CODEBOOK")
    print("=" * 72)

    # Code utilization
    u2_lm = results_2way.get("code_utilization", {}).get("mistral_unique", "?")
    u2_vj = results_2way.get("code_utilization", {}).get("vjepa2_unique", "?")
    u3_lm = results_3way["code_utilization"]["mistral"]
    u3_vj = results_3way["code_utilization"]["vjepa2"]
    u3_cl = results_3way["code_utilization"]["clip"]

    print(f"\n  Code utilization (of 64):")
    print(f"    {'':12s} | {'Two-way':>8s} | {'Three-way':>9s}")
    print(f"    {'-'*12}-+-{'-'*8}-+-{'-'*9}")
    print(f"    {'Mistral':12s} | {u2_lm:>8} | {u3_lm:>9}")
    print(f"    {'V-JEPA2':12s} | {u2_vj:>8} | {u3_vj:>9}")
    print(f"    {'CLIP':12s} | {'n/a':>8} | {u3_cl:>9}")

    # Cross-modal agreement
    a2 = results_2way.get("cross_modal_agreement", {}).get("agreement_rate", 0)
    a3_vc = results_3way["cross_modal_agreement"]["V-JEPA2_vs_CLIP"]["rate"]
    a3_lc = results_3way["cross_modal_agreement"]["Mistral_vs_CLIP"]["rate"]
    a3_vl = results_3way["cross_modal_agreement"]["V-JEPA2_vs_Mistral"]["rate"]

    print(f"\n  Cross-modal code agreement:")
    print(f"    V-JEPA2 <-> Mistral: two-way={a2:.1%}, three-way={a3_vl:.1%}")
    print(f"    V-JEPA2 <-> CLIP:    three-way={a3_vc:.1%}")
    print(f"    Mistral <-> CLIP:    three-way={a3_lc:.1%}")

    # RSA
    r2_orig = results_2way.get("rsa", {}).get("original", {}).get("r", "?")
    r2_post = results_2way.get("rsa", {}).get("post_vq_quantized", {}).get("r", "?")
    r3 = results_3way["rsa"]

    print(f"\n  RSA (Spearman r):")
    print(f"    {'Pair':22s} | {'Orig':>8s} | {'2way post':>9s} | {'3way post':>9s}")
    print(f"    {'-'*22}-+-{'-'*8}-+-{'-'*9}-+-{'-'*9}")

    r_vj_lm_orig = r3["V-JEPA2_vs_Mistral"]["original"]["r"]
    r_vj_lm_post = r3["V-JEPA2_vs_Mistral"]["post_vq"]["r"]
    r_vj_cl_orig = r3["V-JEPA2_vs_CLIP"]["original"]["r"]
    r_vj_cl_post = r3["V-JEPA2_vs_CLIP"]["post_vq"]["r"]
    r_lm_cl_orig = r3["Mistral_vs_CLIP"]["original"]["r"]
    r_lm_cl_post = r3["Mistral_vs_CLIP"]["post_vq"]["r"]

    r2_post_str = f"{r2_post:+.4f}" if isinstance(r2_post, float) else str(r2_post)

    print(f"    {'V-JEPA2 vs Mistral':22s} | {r_vj_lm_orig:+.4f} | {r2_post_str:>9s} | {r_vj_lm_post:+.4f}")
    print(f"    {'V-JEPA2 vs CLIP':22s} | {r_vj_cl_orig:+.4f} | {'n/a':>9s} | {r_vj_cl_post:+.4f}")
    print(f"    {'Mistral vs CLIP':22s} | {r_lm_cl_orig:+.4f} | {'n/a':>9s} | {r_lm_cl_post:+.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 72)
    print("THREE-WAY VQ CODEBOOK: V-JEPA 2 + CLIP + Mistral 7B")
    print("=" * 72)

    lm_aug, vj_aug, cl_aug, labels, lm_base, vj_base, cl_base, valid_phys = \
        prepare_data(n_augment=30, sigma=0.1)

    model = SharedCodebook3Way(
        dim_vjepa=1024, dim_clip=768, dim_mistral=4096,
        codebook_dim=256, n_codes=64,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} parameters")
    print(f"    Encoders: V-JEPA2 1024->256, CLIP 768->256, Mistral 4096->256")
    print(f"    Codebook: 64 entries x 256-dim (EMA)")
    print(f"    Decoders: 256->1024, 256->768, 256->4096")

    print(f"\n  Training (200 epochs, batch=32, lr=1e-3, commitment=0.25):")
    history = train(model, lm_aug, vj_aug, cl_aug, labels,
                    n_epochs=200, batch_size=32, lr=1e-3, commitment_weight=0.25)

    print(f"\n{'='*72}")
    print("EVALUATION ON ORIGINAL 17 CONCEPTS")
    print("=" * 72)
    results = evaluate(model, lm_base, vj_base, cl_base, valid_phys)

    results["architecture"] = {
        "type": "three_way",
        "codebook_dim": 256, "n_codes": 64,
        "n_concepts": len(valid_phys), "concepts": valid_phys,
        "n_augment": 30, "sigma": 0.1,
        "n_epochs": 200, "lr": 1e-3, "commitment_weight": 0.25,
    }
    results["training_final"] = history[-1]

    # Comparison
    print_comparison(results)

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

    with open(output_dir / "codebook3way_results.json", "w", encoding="utf-8") as f:
        json.dump(to_python(results), f, indent=2)

    torch.save(model.state_dict(), output_dir / "codebook3way_model.pt")

    print(f"\n{'='*72}")
    print("SAVED")
    print("=" * 72)
    print(f"  {output_dir}/codebook3way_results.json")
    print(f"  {output_dir}/codebook3way_model.pt")

    # Verdict
    a_vc = results["cross_modal_agreement"]["V-JEPA2_vs_CLIP"]["rate"]
    a_lc = results["cross_modal_agreement"]["Mistral_vs_CLIP"]["rate"]
    a_vl = results["cross_modal_agreement"]["V-JEPA2_vs_Mistral"]["rate"]
    n_codes_used = results["code_utilization"]["combined"]
    r_vl_post = results["rsa"]["V-JEPA2_vs_Mistral"]["post_vq"]["r"]
    r_vl_orig = results["rsa"]["V-JEPA2_vs_Mistral"]["original"]["r"]

    print(f"\n{'='*72}")
    print("VERDICT")
    print("=" * 72)

    if n_codes_used <= 3:
        print(f"  COLLAPSE: Only {n_codes_used} codes used (one per modality).")
        print(f"  CLIP geometry insufficient to bridge the modality gap.")
    elif n_codes_used <= 10:
        print(f"  PARTIAL: {n_codes_used} codes used — some differentiation but limited.")
    else:
        print(f"  GOOD UTILIZATION: {n_codes_used} codes used.")

    if a_vc > 0.15:
        print(f"  V-JEPA2 <-> CLIP agreement: {a_vc:.0%} — CLIP bridges to visual world model.")
    if a_lc > 0.15:
        print(f"  Mistral <-> CLIP agreement: {a_lc:.0%} — CLIP bridges to language model.")
    if a_vl > 0.0:
        print(f"  V-JEPA2 <-> Mistral agreement: {a_vl:.0%} (was 0% in two-way).")
        if a_vl > 0.15:
            print(f"  CLIP IS BRIDGING: indirect agreement via shared CLIP codes!")
    else:
        print(f"  V-JEPA2 <-> Mistral agreement: 0% — same as two-way, CLIP doesn't bridge.")

    if r_vl_post > r_vl_orig + 0.1:
        print(f"\n  Post-VQ RSA V-JEPA2 vs Mistral: {r_vl_post:+.3f} (was {r_vl_orig:+.3f})")
        print(f"  Quantization INCREASED cross-modal alignment!")
    else:
        print(f"\n  Post-VQ RSA V-JEPA2 vs Mistral: {r_vl_post:+.3f} (was {r_vl_orig:+.3f})")
        print(f"  No improvement in cross-modal alignment from CLIP bridging.")
