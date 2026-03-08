"""
Contrastive Codebook: NT-Xent loss to bridge visual/linguistic modalities
-------------------------------------------------------------------------
Previous experiments showed VQ collapse to 2 codes (1 per modality) because
reconstruction loss alone can't bridge the visual/linguistic gap.

Fix: Add contrastive loss (NT-Xent) on pre-VQ projections that explicitly
pulls same-concept embeddings from different modalities toward each other.

Architecture: Three-way ST [768] + CLIP [768] + V-JEPA 2 [1024]
  - Same as previous: Linear+LayerNorm encoders -> VQ(64x256) -> Linear decoders
  - NEW: NT-Xent contrastive loss on z_e (pre-VQ projections)

Runs three lambda values: 0.1, 0.5, 1.0
Total loss: L_recon + 0.25 * L_commit + lambda * L_contrastive

Output: lm_output/codebook_contrastive_results.json
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


# ── Data ─────────────────────────────────────────────────────────────────────

def load_physical_bases():
    """Load and L2-normalize physical concept embeddings for all models."""
    st = np.load(output_dir / "st_hiddens.npy")
    vjepa2 = np.load(output_dir / "vjepa2_hiddens.npy")
    clip = np.load(output_dir / "clip_hiddens.npy")

    st_n = np.linalg.norm(st, axis=-1)
    vj_n = np.linalg.norm(vjepa2, axis=-1)
    cl_n = np.linalg.norm(clip, axis=-1)

    valid_phys = [c for c in PHYSICAL
                  if st_n[ALL_CONCEPTS.index(c)] > 1e-8
                  and vj_n[ALL_CONCEPTS.index(c)] > 1e-8
                  and cl_n[ALL_CONCEPTS.index(c)] > 1e-8]
    phys_idx = [ALL_CONCEPTS.index(c) for c in valid_phys]

    st_base = st[phys_idx]
    vj_base = vjepa2[phys_idx]
    cl_base = clip[phys_idx]
    st_base = st_base / np.linalg.norm(st_base, axis=-1, keepdims=True)
    vj_base = vj_base / np.linalg.norm(vj_base, axis=-1, keepdims=True)
    cl_base = cl_base / np.linalg.norm(cl_base, axis=-1, keepdims=True)

    return st_base, vj_base, cl_base, valid_phys


def augment(bases_dict, n_augment=30, sigma=0.1):
    """Generate augmented training data with concept labels."""
    rng = np.random.default_rng(42)
    result = {}
    labels = None

    for name, base in bases_dict.items():
        n_concepts = base.shape[0]
        aug = []
        lab = []
        for ci in range(n_concepts):
            for _ in range(n_augment):
                noise = rng.normal(0, sigma, base.shape[1])
                v = base[ci] + noise
                aug.append(v / np.linalg.norm(v))
                lab.append(ci)
        result[name] = np.array(aug, dtype=np.float32)
        if labels is None:
            labels = np.array(lab, dtype=np.int64)

    return result, labels


# ── VQ Layer ─────────────────────────────────────────────────────────────────

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


# ── Generic N-way Codebook ───────────────────────────────────────────────────

class SharedCodebookNWay(nn.Module):
    def __init__(self, modality_dims, codebook_dim=256, n_codes=64):
        super().__init__()
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()

        for name, dim in modality_dims.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, codebook_dim), nn.LayerNorm(codebook_dim))
            self.decoders[name] = nn.Linear(codebook_dim, dim)

        self.vq = VectorQuantizerEMA(n_codes, codebook_dim)

    def forward(self, x, modality):
        z_e = self.encoders[modality](x)
        z_q, commit_loss, indices = self.vq(z_e)
        x_recon = self.decoders[modality](z_q)
        return x_recon, commit_loss, indices, z_e, z_q


# ── NT-Xent Contrastive Loss ────────────────────────────────────────────────

def nt_xent_loss(z_a, z_b, labels_a, labels_b, temperature=0.1):
    """
    NT-Xent (normalized temperature-scaled cross entropy) contrastive loss.

    For each sample in z_a, find samples in z_b with the same concept label
    (positives) and push apart samples with different labels (negatives).

    Args:
        z_a: [B, D] pre-VQ projections from modality A
        z_b: [B, D] pre-VQ projections from modality B
        labels_a: [B] concept labels for modality A
        labels_b: [B] concept labels for modality B
        temperature: scaling temperature (0.1 = sharp, like CLIP)

    Returns:
        scalar loss
    """
    # L2 normalize
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)

    # Similarity matrix: [B_a, B_b]
    sim = z_a @ z_b.T / temperature

    # Positive mask: same concept label across modalities
    pos_mask = (labels_a.unsqueeze(1) == labels_b.unsqueeze(0)).float()

    # For each row in z_a, we want the log-softmax over z_b entries,
    # weighted by which are positives
    # Loss = -mean over positives of log(exp(sim_pos) / sum(exp(sim_all)))

    # Check if there are any positives
    n_pos = pos_mask.sum()
    if n_pos == 0:
        return torch.tensor(0.0, device=z_a.device)

    # Log-softmax over columns for each row
    log_prob = F.log_softmax(sim, dim=1)

    # Mean of log_prob at positive positions
    loss = -(pos_mask * log_prob).sum() / n_pos

    return loss


# ── Train with contrastive loss ──────────────────────────────────────────────

def train_model_contrastive(model, data_dict, labels, n_epochs=200,
                             batch_size=32, lr=1e-3, commitment_weight=0.25,
                             contrastive_lambda=0.5, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    modalities = list(data_dict.keys())
    tensors = {k: torch.tensor(v, device=DEVICE) for k, v in data_dict.items()}
    labels_t = torch.tensor(labels, device=DEVICE)
    n_samples = tensors[modalities[0]].shape[0]
    rng = np.random.default_rng(123)

    # Identify visual vs language modalities for contrastive pairs
    visual_mods = [m for m in modalities if m in ("vjepa", "clip")]
    lang_mods = [m for m in modalities if m in ("st",)]

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)
        ep_loss = 0.0
        ep_recon = 0.0
        ep_contrastive = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            total_recon = 0.0
            total_commit = 0.0
            batch_labels = labels_t[idx]

            # Forward pass all modalities, collect z_e
            z_e_dict = {}
            for mod in modalities:
                recon, commit, _, z_e, _ = model(tensors[mod][idx], mod)
                total_recon += F.mse_loss(recon, tensors[mod][idx])
                total_commit += commit
                z_e_dict[mod] = z_e

            # Contrastive loss: all language <-> visual pairs
            contrastive = torch.tensor(0.0, device=DEVICE)
            n_pairs = 0
            for lang_mod in lang_mods:
                for vis_mod in visual_mods:
                    contrastive = contrastive + nt_xent_loss(
                        z_e_dict[lang_mod], z_e_dict[vis_mod],
                        batch_labels, batch_labels
                    )
                    n_pairs += 1
            if n_pairs > 0:
                contrastive = contrastive / n_pairs

            loss = (total_recon
                    + commitment_weight * (total_commit / len(modalities))
                    + contrastive_lambda * contrastive)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            ep_recon += total_recon.item()
            ep_contrastive += contrastive.item()
            n_batches += 1

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: loss={ep_loss/n_batches:.4f}"
                  f"  recon={ep_recon/n_batches:.4f}"
                  f"  contrastive={ep_contrastive/n_batches:.4f}")

    return ep_loss / n_batches


# ── Evaluate ─────────────────────────────────────────────────────────────────

def safe_rsa(a, b):
    try:
        r, p = rsa_score(a, b, "spearman")
        if np.isnan(r):
            return 0.0, 1.0
        return r, p
    except Exception:
        return 0.0, 1.0


def evaluate_codebook(model, bases_dict, valid_phys, label=""):
    """Evaluate codebook on original (non-augmented) concept embeddings."""
    model.eval()
    modalities = list(bases_dict.keys())
    n = len(valid_phys)

    tensors = {k: torch.tensor(v, device=DEVICE, dtype=torch.float32)
               for k, v in bases_dict.items()}

    results_per_mod = {}
    with torch.no_grad():
        for mod in modalities:
            recon, _, indices, z_e, z_q = model(tensors[mod], mod)
            results_per_mod[mod] = {
                "recon": recon, "indices": indices.cpu().numpy(),
                "z_e": z_e.cpu().float().numpy(), "z_q": z_q.cpu().float().numpy(),
                "mse": F.mse_loss(recon, tensors[mod]).item(),
                "cos": F.cosine_similarity(recon, tensors[mod], dim=1).mean().item(),
            }

    results = {}

    # 1. Reconstruction
    print(f"\n  Reconstruction:")
    for mod in modalities:
        r = results_per_mod[mod]
        print(f"    {mod:8s}: MSE={r['mse']:.4f}, cos_sim={r['cos']:.4f}")
    results["reconstruction"] = {mod: {"mse": results_per_mod[mod]["mse"],
                                        "cos": results_per_mod[mod]["cos"]}
                                  for mod in modalities}

    # 2. Code utilization
    all_indices = np.concatenate([results_per_mod[m]["indices"] for m in modalities])
    unique_all = len(np.unique(all_indices))
    print(f"\n  Code utilization (of 64):")
    for mod in modalities:
        u = len(np.unique(results_per_mod[mod]["indices"]))
        print(f"    {mod:8s}: {u} codes")
        results.setdefault("code_utilization", {})[mod] = int(u)
    print(f"    Combined: {unique_all} codes")
    results["code_utilization"]["combined"] = int(unique_all)

    # 3. Cross-modal agreement
    print(f"\n  Cross-modal agreement:")
    results["cross_modal"] = {}
    for i, mod_a in enumerate(modalities):
        for mod_b in modalities[i+1:]:
            idx_a = results_per_mod[mod_a]["indices"]
            idx_b = results_per_mod[mod_b]["indices"]
            n_agree = int(np.sum(idx_a == idx_b))
            rate = n_agree / n
            freq_a = Counter(idx_a)
            freq_b = Counter(idx_b)
            chance = sum((freq_a.get(k, 0)/n) * (freq_b.get(k, 0)/n)
                         for k in set(list(freq_a) + list(freq_b)))
            print(f"    {mod_a:8s} <-> {mod_b:8s}: {n_agree}/{n} ({rate:.1%}), chance={chance:.1%}")
            results["cross_modal"][f"{mod_a}_vs_{mod_b}"] = {
                "n_agree": n_agree, "rate": rate, "chance": chance}

    # 4. Per-concept code table
    print(f"\n    {'Concept':12s}", end="")
    for mod in modalities:
        print(f" | {mod:>7s}", end="")
    print()
    print(f"    {'-'*12}", end="")
    for _ in modalities:
        print(f"-+-{'-'*7}", end="")
    print()

    per_concept = {}
    for ci, concept in enumerate(valid_phys):
        print(f"    {concept:12s}", end="")
        codes = {}
        for mod in modalities:
            c = int(results_per_mod[mod]["indices"][ci])
            codes[mod] = c
            print(f" | {c:7d}", end="")
        print()
        per_concept[concept] = codes
    results["per_concept"] = per_concept

    # 5. Pre-VQ cross-modal RSA (key diagnostic for contrastive loss)
    print(f"\n  Pre-VQ cross-modal RSA (shared 256-dim space):")
    print(f"    {'Pair':22s} | {'Pre-VQ r':>10s} | {'p':>8s} | sig")
    print(f"    {'-'*22}-+-{'-'*10}-+-{'-'*8}-+----")
    results["pre_vq_rsa"] = {}
    for i, mod_a in enumerate(modalities):
        for mod_b in modalities[i+1:]:
            rsm_pre_a = cosine_similarity_matrix(results_per_mod[mod_a]["z_e"])
            rsm_pre_b = cosine_similarity_matrix(results_per_mod[mod_b]["z_e"])
            r_pre, p_pre = safe_rsa(rsm_pre_a, rsm_pre_b)
            sig = "*" if p_pre < 0.05 else " "
            pair = f"{mod_a} vs {mod_b}"
            print(f"    {pair:22s} | {r_pre:+.4f}     | {p_pre:.4f}  | {sig}")
            results["pre_vq_rsa"][f"{mod_a}_vs_{mod_b}"] = {
                "r": float(r_pre), "p": float(p_pre)}

    # 6. RSA across stages (original -> pre-VQ -> post-VQ)
    print(f"\n  RSA across stages:")
    print(f"    {'Pair':22s} | {'Original':>10s} | {'Pre-VQ':>10s} | {'Post-VQ':>10s}")
    print(f"    {'-'*22}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    results["rsa"] = {}
    for i, mod_a in enumerate(modalities):
        for mod_b in modalities[i+1:]:
            rsm_orig_a = cosine_similarity_matrix(bases_dict[mod_a])
            rsm_orig_b = cosine_similarity_matrix(bases_dict[mod_b])
            rsm_pre_a = cosine_similarity_matrix(results_per_mod[mod_a]["z_e"])
            rsm_pre_b = cosine_similarity_matrix(results_per_mod[mod_b]["z_e"])
            rsm_q_a = cosine_similarity_matrix(results_per_mod[mod_a]["z_q"])
            rsm_q_b = cosine_similarity_matrix(results_per_mod[mod_b]["z_q"])

            r_orig, p_orig = safe_rsa(rsm_orig_a, rsm_orig_b)
            r_pre, p_pre = safe_rsa(rsm_pre_a, rsm_pre_b)
            r_post, p_post = safe_rsa(rsm_q_a, rsm_q_b)

            s_o = "*" if p_orig < 0.05 else " "
            s_p = "*" if p_pre < 0.05 else " "
            s_q = "*" if p_post < 0.05 else " "

            pair = f"{mod_a} vs {mod_b}"
            print(f"    {pair:22s} | {r_orig:+.4f}{s_o}   | {r_pre:+.4f}{s_p}   | {r_post:+.4f}{s_q}")
            results["rsa"][f"{mod_a}_vs_{mod_b}"] = {
                "original": {"r": float(r_orig), "p": float(p_orig)},
                "pre_vq": {"r": float(r_pre), "p": float(p_pre)},
                "post_vq": {"r": float(r_post), "p": float(p_post)},
            }

    # 7. Within-modality preservation
    print(f"\n  Structure preservation (original -> post_VQ):")
    results["preservation"] = {}
    for mod in modalities:
        rsm_orig = cosine_similarity_matrix(bases_dict[mod])
        rsm_pre = cosine_similarity_matrix(results_per_mod[mod]["z_e"])
        rsm_q = cosine_similarity_matrix(results_per_mod[mod]["z_q"])
        r_pre, _ = safe_rsa(rsm_orig, rsm_pre)
        r_post, _ = safe_rsa(rsm_orig, rsm_q)
        print(f"    {mod:8s}: pre_VQ r={r_pre:+.4f}, post_VQ r={r_post:+.4f}")
        results["preservation"][mod] = {"pre_vq": float(r_pre), "post_vq": float(r_post)}

    return results


# ── Comparison table ─────────────────────────────────────────────────────────

def print_comparison_table(all_results):
    """Print comparison across lambda=0 (baseline) and all tested values."""
    print(f"\n{'='*90}")
    print("COMPARISON TABLE: CONTRASTIVE LOSS SWEEP")
    print("=" * 90)

    lambdas = sorted(all_results.keys())

    # Code utilization
    print(f"\n  Code utilization (of 64):")
    print(f"    {'':8s}", end="")
    for lam in lambdas:
        print(f" | lambda={lam:<5s}", end="")
    print()
    print(f"    {'':8s}", end="")
    for _ in lambdas:
        print(f"-+-{'-'*11}", end="")
    print()

    for mod in ["st", "clip", "vjepa", "combined"]:
        print(f"    {mod:8s}", end="")
        for lam in lambdas:
            r = all_results[lam]
            u = r.get("code_utilization", {}).get(mod, "?")
            print(f" | {str(u):>11s}", end="")
        print()

    # Cross-modal agreement
    print(f"\n  Cross-modal agreement:")
    pairs = ["st_vs_clip", "st_vs_vjepa", "clip_vs_vjepa"]
    pair_labels = ["ST <-> CLIP", "ST <-> VJEPA", "CLIP <-> VJEPA"]
    print(f"    {'Pair':14s}", end="")
    for lam in lambdas:
        print(f" | lambda={lam:<5s}", end="")
    print()
    print(f"    {'-'*14}", end="")
    for _ in lambdas:
        print(f"-+-{'-'*11}", end="")
    print()
    for pair, plabel in zip(pairs, pair_labels):
        print(f"    {plabel:14s}", end="")
        for lam in lambdas:
            r = all_results[lam]
            cm = r.get("cross_modal", {}).get(pair, {})
            rate = cm.get("rate")
            if rate is not None:
                print(f" | {rate:>10.0%}", end="")
            else:
                print(f" | {'n/a':>11s}", end="")
        print()

    # Pre-VQ cross-modal RSA (key signal)
    print(f"\n  Pre-VQ cross-modal RSA (earliest signal of contrastive effect):")
    print(f"    {'Pair':14s}", end="")
    for lam in lambdas:
        print(f" | lambda={lam:<5s}", end="")
    print()
    print(f"    {'-'*14}", end="")
    for _ in lambdas:
        print(f"-+-{'-'*11}", end="")
    print()
    for pair, plabel in zip(pairs, pair_labels):
        print(f"    {plabel:14s}", end="")
        for lam in lambdas:
            r = all_results[lam]
            pre = r.get("pre_vq_rsa", {}).get(pair, {})
            rv = pre.get("r")
            if rv is not None:
                sig = "*" if pre.get("p", 1.0) < 0.05 else " "
                print(f" | {rv:+.4f}{sig}    ", end="")
            else:
                print(f" | {'n/a':>11s}", end="")
        print()

    # Post-VQ RSA
    print(f"\n  Post-VQ RSA:")
    print(f"    {'Pair':14s}", end="")
    for lam in lambdas:
        print(f" | lambda={lam:<5s}", end="")
    print()
    print(f"    {'-'*14}", end="")
    for _ in lambdas:
        print(f"-+-{'-'*11}", end="")
    print()
    for pair, plabel in zip(pairs, pair_labels):
        print(f"    {plabel:14s}", end="")
        for lam in lambdas:
            r = all_results[lam]
            rsa = r.get("rsa", {}).get(pair, {}).get("post_vq", {})
            rv = rsa.get("r")
            if rv is not None:
                sig = "*" if rsa.get("p", 1.0) < 0.05 else " "
                print(f" | {rv:+.4f}{sig}    ", end="")
            else:
                print(f" | {'n/a':>11s}", end="")
        print()

    # Reconstruction quality
    print(f"\n  Reconstruction (cos_sim):")
    print(f"    {'Modality':14s}", end="")
    for lam in lambdas:
        print(f" | lambda={lam:<5s}", end="")
    print()
    print(f"    {'-'*14}", end="")
    for _ in lambdas:
        print(f"-+-{'-'*11}", end="")
    print()
    for mod in ["st", "clip", "vjepa"]:
        print(f"    {mod:14s}", end="")
        for lam in lambdas:
            r = all_results[lam]
            cos = r.get("reconstruction", {}).get(mod, {}).get("cos")
            if cos is not None:
                print(f" | {cos:>10.4f}", end="")
            else:
                print(f" | {'n/a':>11s}", end="")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st_base, vj_base, cl_base, valid_phys = load_physical_bases()
    n = len(valid_phys)

    print("=" * 90)
    print(f"CONTRASTIVE CODEBOOK EXPERIMENT ({n} physical concepts)")
    print("=" * 90)

    # Geometry diagnostic
    print(f"\n  GEOMETRY DIAGNOSTIC:")
    for name, base in [("ST", st_base), ("V-JEPA2", vj_base), ("CLIP", cl_base)]:
        rsm = base @ base.T
        nc = rsm.shape[0]
        triu = rsm[np.triu_indices(nc, k=1)]
        spread = 1.0 - triu.mean()
        print(f"    {name:8s}: mean_sim={triu.mean():.4f}, spread={spread:.4f}")

    bases = {"st": st_base, "clip": cl_base, "vjepa": vj_base}

    # Load lambda=0 baseline from previous results
    all_results = {}
    try:
        with open(output_dir / "codebook_st3way_results.json") as f:
            prev = json.load(f)
        baseline = prev.get("three_way_st_clip_vjepa", {})
        if baseline:
            all_results["0.0"] = baseline
            print(f"\n  Loaded lambda=0 baseline from codebook_st3way_results.json")
    except FileNotFoundError:
        print(f"\n  No lambda=0 baseline found, will skip in comparison")

    # Run three lambda values
    lambda_values = [0.1, 0.5, 1.0]

    for lam in lambda_values:
        print(f"\n{'='*90}")
        print(f"LAMBDA = {lam} (contrastive weight)")
        print("=" * 90)

        data, labels = augment(bases, n_augment=30, sigma=0.1)

        model = SharedCodebookNWay(
            {"st": 768, "clip": 768, "vjepa": 1024},
            codebook_dim=256, n_codes=64
        ).to(DEVICE)
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        print(f"  Training (200 epochs, lambda_contrastive={lam}):")
        train_model_contrastive(
            model, data, labels,
            n_epochs=200, batch_size=32, lr=1e-3,
            commitment_weight=0.25, contrastive_lambda=lam
        )

        print(f"\n  EVALUATION (lambda={lam}):")
        results = evaluate_codebook(model, bases, valid_phys, f"lam{lam}")

        all_results[str(lam)] = results

        del model
        torch.cuda.empty_cache()

    # ── Comparison table ─────────────────────────────────────────────────────

    print_comparison_table(all_results)

    # ── Save results ─────────────────────────────────────────────────────────

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

    with open(output_dir / "codebook_contrastive_results.json", "w", encoding="utf-8") as f:
        json.dump(to_python(all_results), f, indent=2)

    print(f"\n{'='*90}")
    print("SAVED: lm_output/codebook_contrastive_results.json")
    print("=" * 90)

    # ── Verdict ──────────────────────────────────────────────────────────────

    print(f"\n{'='*90}")
    print("VERDICT")
    print("=" * 90)

    best_lam = None
    best_util = 0
    for lam in lambda_values:
        r = all_results[str(lam)]
        u = r.get("code_utilization", {}).get("combined", 0)
        if u > best_util:
            best_util = u
            best_lam = lam

    baseline_util = all_results.get("0.0", {}).get("code_utilization", {}).get("combined", 2)

    if best_util > 10:
        print(f"  COLLAPSE PREVENTED at lambda={best_lam}: {best_util} codes (vs {baseline_util} at lambda=0)")
        # Check cross-modal agreement
        r = all_results[str(best_lam)]
        sv = r.get("cross_modal", {}).get("st_vs_vjepa", {}).get("rate", 0)
        print(f"  ST <-> V-JEPA2 agreement: {sv:.0%}")
        pre_rsa = r.get("pre_vq_rsa", {}).get("st_vs_vjepa", {}).get("r", 0)
        print(f"  Pre-VQ ST vs V-JEPA2 RSA: {pre_rsa:+.4f}")
        print(f"  Contrastive loss successfully bridges the visual/linguistic gap.")
    elif best_util > baseline_util:
        print(f"  PARTIAL IMPROVEMENT at lambda={best_lam}: {best_util} codes (vs {baseline_util} at lambda=0)")
        print(f"  Contrastive pressure helps but doesn't fully prevent collapse.")
        print(f"  May need stronger lambda, more epochs, or different architecture.")
    else:
        print(f"  NO IMPROVEMENT: {best_util} codes at best (lambda={best_lam})")
        print(f"  Even explicit cross-modal alignment can't bridge the gap.")
        print(f"  The modality difference may be too fundamental for this architecture.")
