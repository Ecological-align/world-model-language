"""
ST Codebook Experiments: Does sentence-transformer geometry prevent collapse?
-----------------------------------------------------------------------------
Two runs in one script:

1. Two-way: ST [768] <-> V-JEPA 2 [1024]
   - Direct comparison: can ST find shared structure with V-JEPA2 without CLIP?
   - RSA baseline: ST vs V-JEPA2 r=-0.025 (ns) — expect weak/no bridging

2. Three-way: ST [768] + CLIP [768] + V-JEPA 2 [1024]
   - Same as train_codebook_3way.py but ST replaces Mistral
   - ST spread ~0.36 vs Mistral's 0.074 — 5x more variance
   - Tests: is collapse caused by Mistral's geometry or visual/linguistic incompatibility?

Same architecture and hyperparameters as all previous runs.
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


def print_geometry(bases_dict):
    """Print mean pairwise cosine similarity for each modality."""
    print(f"\n  GEOMETRY DIAGNOSTIC:")
    for name, base in bases_dict.items():
        rsm = base @ base.T
        n = rsm.shape[0]
        triu = rsm[np.triu_indices(n, k=1)]
        spread = 1.0 - triu.mean()
        print(f"    {name:8s}: mean_sim={triu.mean():.4f}, std={triu.std():.4f}, "
              f"spread={spread:.4f}, min_sim={triu.min():.4f}")


def augment(bases_dict, n_augment=30, sigma=0.1):
    """Generate augmented training data from base embeddings."""
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
    """Flexible N-modality codebook."""
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


# ── Train ────────────────────────────────────────────────────────────────────

def train_model(model, data_dict, n_epochs=200, batch_size=32, lr=1e-3,
                commitment_weight=0.25, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    modalities = list(data_dict.keys())
    tensors = {k: torch.tensor(v, device=DEVICE) for k, v in data_dict.items()}
    n_samples = tensors[modalities[0]].shape[0]
    rng = np.random.default_rng(123)

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            total_recon = 0.0
            total_commit = 0.0

            for mod in modalities:
                recon, commit, _, _, _ = model(tensors[mod][idx], mod)
                total_recon += F.mse_loss(recon, tensors[mod][idx])
                total_commit += commit

            loss = total_recon + commitment_weight * (total_commit / len(modalities))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{n_epochs}: loss={ep_loss/n_batches:.4f}")

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

    # Forward pass
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

    # 5. RSA across stages
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

    # 6. Within-modality preservation
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


# ── Final comparison table ───────────────────────────────────────────────────

def print_grand_comparison(r_2way_st, r_3way_st):
    """Print comparison across all codebook experiments."""
    # Load previous results
    try:
        with open(output_dir / "codebook_results.json") as f:
            r_2way_lm = json.load(f)
    except FileNotFoundError:
        r_2way_lm = None
    try:
        with open(output_dir / "codebook3way_results.json") as f:
            r_3way_lm = json.load(f)
    except FileNotFoundError:
        r_3way_lm = None

    print(f"\n{'='*80}")
    print("GRAND COMPARISON: ALL CODEBOOK EXPERIMENTS")
    print("=" * 80)

    def get_util(r, key="combined"):
        if r is None:
            return "n/a"
        u = r.get("code_utilization", {})
        return str(u.get(key, u.get("combined_unique", "?")))

    def get_agree(r, pair):
        if r is None:
            return "n/a"
        cm = r.get("cross_modal_agreement", r.get("cross_modal", {}))
        entry = cm.get(pair, {})
        rate = entry.get("rate", entry.get("agreement_rate"))
        if rate is None:
            return "n/a"
        return f"{rate:.0%}"

    def get_rsa_post(r, pair):
        if r is None:
            return "n/a"
        rsa = r.get("rsa", {})
        entry = rsa.get(pair, {})
        post = entry.get("post_vq", entry.get("post_vq_quantized", {}))
        rv = post.get("r")
        if rv is None:
            return "n/a"
        return f"{rv:+.3f}"

    experiments = [
        ("2way Mistral+VJ", r_2way_lm),
        ("3way Mistral+CL+VJ", r_3way_lm),
        ("2way ST+VJ", r_2way_st),
        ("3way ST+CL+VJ", r_3way_st),
    ]

    # Code utilization
    print(f"\n  Code utilization (of 64):")
    print(f"    {'Experiment':22s} | {'Combined':>8s}")
    print(f"    {'-'*22}-+-{'-'*8}")
    for name, r in experiments:
        print(f"    {name:22s} | {get_util(r):>8s}")

    # Cross-modal agreement (language model <-> visual)
    print(f"\n  Language <-> V-JEPA2 agreement:")
    print(f"    {'Experiment':22s} | {'Rate':>8s}")
    print(f"    {'-'*22}-+-{'-'*8}")
    # 2way Mistral
    if r_2way_lm:
        print(f"    {'2way Mistral+VJ':22s} | {get_agree(r_2way_lm, 'V-JEPA2_vs_Mistral'):>8s}")
    # 3way Mistral
    if r_3way_lm:
        print(f"    {'3way Mistral+CL+VJ':22s} | {get_agree(r_3way_lm, 'V-JEPA2_vs_Mistral'):>8s}")
    # 2way ST
    print(f"    {'2way ST+VJ':22s} | {get_agree(r_2way_st, 'st_vs_vjepa'):>8s}")
    # 3way ST
    print(f"    {'3way ST+CL+VJ':22s} | {get_agree(r_3way_st, 'st_vs_vjepa'):>8s}")

    # Visual <-> Visual
    print(f"\n  V-JEPA2 <-> CLIP agreement:")
    print(f"    {'Experiment':22s} | {'Rate':>8s}")
    print(f"    {'-'*22}-+-{'-'*8}")
    if r_3way_lm:
        print(f"    {'3way Mistral+CL+VJ':22s} | {get_agree(r_3way_lm, 'V-JEPA2_vs_CLIP'):>8s}")
    print(f"    {'3way ST+CL+VJ':22s} | {get_agree(r_3way_st, 'vjepa_vs_clip'):>8s}")

    # Post-VQ RSA
    print(f"\n  Post-VQ RSA (language <-> V-JEPA2):")
    print(f"    {'Experiment':22s} | {'Post-VQ r':>9s} | note")
    print(f"    {'-'*22}-+-{'-'*9}-+------")
    if r_2way_lm:
        print(f"    {'2way Mistral+VJ':22s} | {get_rsa_post(r_2way_lm, 'V-JEPA2_vs_Mistral'):>9s} | original r=-0.036")
    if r_3way_lm:
        print(f"    {'3way Mistral+CL+VJ':22s} | {get_rsa_post(r_3way_lm, 'V-JEPA2_vs_Mistral'):>9s} |")
    print(f"    {'2way ST+VJ':22s} | {get_rsa_post(r_2way_st, 'st_vs_vjepa'):>9s} | original r=-0.025")
    print(f"    {'3way ST+CL+VJ':22s} | {get_rsa_post(r_3way_st, 'st_vs_vjepa'):>9s} |")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st_base, vj_base, cl_base, valid_phys = load_physical_bases()
    n = len(valid_phys)

    print("=" * 80)
    print(f"ST CODEBOOK EXPERIMENTS ({n} physical concepts)")
    print("=" * 80)

    print_geometry({"ST": st_base, "V-JEPA2": vj_base, "CLIP": cl_base})

    # Also show Mistral for comparison
    lm = np.load(output_dir / "lm_hiddens.npy")
    phys_idx = [ALL_CONCEPTS.index(c) for c in valid_phys]
    lm_base = lm[phys_idx]
    lm_base = lm_base / np.linalg.norm(lm_base, axis=-1, keepdims=True)
    print_geometry({"Mistral": lm_base})
    del lm, lm_base

    # ── Run 1: Two-way ST <-> V-JEPA 2 ──────────────────────────────────────

    print(f"\n{'='*80}")
    print("RUN 1: TWO-WAY CODEBOOK (ST <-> V-JEPA 2)")
    print("=" * 80)

    bases_2way = {"st": st_base, "vjepa": vj_base}
    data_2way, labels = augment(bases_2way, n_augment=30, sigma=0.1)

    model_2way = SharedCodebookNWay(
        {"st": 768, "vjepa": 1024}, codebook_dim=256, n_codes=64
    ).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model_2way.parameters()):,}")

    print(f"  Training (200 epochs):")
    train_model(model_2way, data_2way, n_epochs=200)

    print(f"\n  EVALUATION:")
    results_2way = evaluate_codebook(model_2way, bases_2way, valid_phys, "2way_st")

    del model_2way
    torch.cuda.empty_cache()

    # ── Run 2: Three-way ST + CLIP + V-JEPA 2 ───────────────────────────────

    print(f"\n{'='*80}")
    print("RUN 2: THREE-WAY CODEBOOK (ST + CLIP + V-JEPA 2)")
    print("=" * 80)

    bases_3way = {"st": st_base, "clip": cl_base, "vjepa": vj_base}
    data_3way, labels = augment(bases_3way, n_augment=30, sigma=0.1)

    model_3way = SharedCodebookNWay(
        {"st": 768, "clip": 768, "vjepa": 1024}, codebook_dim=256, n_codes=64
    ).to(DEVICE)
    print(f"  Params: {sum(p.numel() for p in model_3way.parameters()):,}")

    print(f"  Training (200 epochs):")
    train_model(model_3way, data_3way, n_epochs=200)

    print(f"\n  EVALUATION:")
    results_3way = evaluate_codebook(model_3way, bases_3way, valid_phys, "3way_st")

    del model_3way
    torch.cuda.empty_cache()

    # ── Grand comparison ─────────────────────────────────────────────────────

    print_grand_comparison(results_2way, results_3way)

    # ── Save ─────────────────────────────────────────────────────────────────

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

    combined = {
        "two_way_st_vjepa": to_python(results_2way),
        "three_way_st_clip_vjepa": to_python(results_3way),
    }

    with open(output_dir / "codebook_st3way_results.json", "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{'='*80}")
    print("SAVED: lm_output/codebook_st3way_results.json")
    print("=" * 80)

    # Verdict
    u_2way = results_2way["code_utilization"]["combined"]
    u_3way = results_3way["code_utilization"]["combined"]

    a_2way = results_2way["cross_modal"]["st_vs_vjepa"]["rate"]
    a_3way_sv = results_3way["cross_modal"]["st_vs_vjepa"]["rate"]
    a_3way_vc = results_3way["cross_modal"].get("vjepa_vs_clip",
                    results_3way["cross_modal"].get("clip_vs_vjepa", {})).get("rate", 0)

    print(f"\n{'='*80}")
    print("VERDICT")
    print("=" * 80)

    if u_3way > 10:
        print(f"  ST geometry PREVENTS collapse: {u_3way} codes used (vs 2 with Mistral)")
        print(f"  The problem was Mistral's compressed geometry, not visual/linguistic incompatibility.")
    elif u_3way > 3:
        print(f"  PARTIAL improvement: {u_3way} codes (vs 2 with Mistral)")
        print(f"  More spread helps but doesn't fully resolve the modality gap.")
    else:
        print(f"  STILL COLLAPSED: {u_3way} codes despite 5x more spread.")
        print(f"  The problem is fundamental visual/linguistic incompatibility,")
        print(f"  not Mistral's specific geometry.")

    if a_3way_sv > 0:
        print(f"\n  ST <-> V-JEPA2 agreement: {a_3way_sv:.0%} (was 0% with Mistral)")
    if a_3way_vc > 0:
        print(f"  V-JEPA2 <-> CLIP agreement: {a_3way_vc:.0%}")
