"""
Multi-seed Contrastive Codebook: Variance estimates across 5 seeds
------------------------------------------------------------------
Runs the contrastive codebook (ST + CLIP + V-JEPA 2, NT-Xent loss)
at lambda={0.1, 0.5, 1.0} x seeds={42, 123, 7, 99, 2025} = 15 runs.

Collects per run:
  - active_codes (combined code utilization)
  - st_vj_agreement_rate
  - post_vq_rsa_st_vj (r value)
  - recon_cos_st
  - preservation_st_post_vq

Output:
  lm_output/codebook_multiseed_results.json   -- full per-run data
  lm_output/codebook_multiseed_summary.json   -- mean/std table + verdict
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time
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

SEEDS = [42, 123, 7, 99, 2025]
LAMBDA_VALUES = [0.1, 0.5, 1.0]


# ── Data ─────────────────────────────────────────────────────────────────────

def load_physical_bases():
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


def augment(bases_dict, n_augment=30, sigma=0.1, seed=42):
    rng = np.random.default_rng(seed)
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


# ── N-way Codebook ───────────────────────────────────────────────────────────

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


# ── NT-Xent ──────────────────────────────────────────────────────────────────

def nt_xent_loss(z_a, z_b, labels_a, labels_b, temperature=0.1):
    z_a = F.normalize(z_a, dim=1)
    z_b = F.normalize(z_b, dim=1)
    sim = z_a @ z_b.T / temperature
    pos_mask = (labels_a.unsqueeze(1) == labels_b.unsqueeze(0)).float()
    n_pos = pos_mask.sum()
    if n_pos == 0:
        return torch.tensor(0.0, device=z_a.device)
    log_prob = F.log_softmax(sim, dim=1)
    loss = -(pos_mask * log_prob).sum() / n_pos
    return loss


# ── Train ────────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_model_contrastive(model, data_dict, labels, seed=42,
                             n_epochs=200, batch_size=32, lr=1e-3,
                             commitment_weight=0.25, contrastive_lambda=0.5,
                             verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    modalities = list(data_dict.keys())
    tensors = {k: torch.tensor(v, device=DEVICE) for k, v in data_dict.items()}
    labels_t = torch.tensor(labels, device=DEVICE)
    n_samples = tensors[modalities[0]].shape[0]
    rng = np.random.default_rng(seed)

    visual_mods = [m for m in modalities if m in ("vjepa", "clip")]
    lang_mods = [m for m in modalities if m in ("st",)]

    for epoch in range(n_epochs):
        model.train()
        perm = rng.permutation(n_samples)
        ep_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            total_recon = 0.0
            total_commit = 0.0
            batch_labels = labels_t[idx]

            z_e_dict = {}
            for mod in modalities:
                recon, commit, _, z_e, _ = model(tensors[mod][idx], mod)
                total_recon += F.mse_loss(recon, tensors[mod][idx])
                total_commit += commit
                z_e_dict[mod] = z_e

            contrastive = torch.tensor(0.0, device=DEVICE)
            n_pairs = 0
            for lang_mod in lang_mods:
                for vis_mod in visual_mods:
                    contrastive = contrastive + nt_xent_loss(
                        z_e_dict[lang_mod], z_e_dict[vis_mod],
                        batch_labels, batch_labels)
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
            n_batches += 1

        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
            print(f"      Epoch {epoch+1:3d}/{n_epochs}: loss={ep_loss/n_batches:.4f}")

    return ep_loss / n_batches


# ── Evaluate (compact — returns only the 5 metrics) ─────────────────────────

def safe_rsa(a, b):
    try:
        r, p = rsa_score(a, b, "spearman")
        if np.isnan(r):
            return 0.0, 1.0
        return r, p
    except Exception:
        return 0.0, 1.0


def evaluate_compact(model, bases_dict, valid_phys):
    """Return dict with the 5 key metrics + per-concept codes."""
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
                "indices": indices.cpu().numpy(),
                "z_e": z_e.cpu().float().numpy(),
                "z_q": z_q.cpu().float().numpy(),
                "cos": F.cosine_similarity(recon, tensors[mod], dim=1).mean().item(),
            }

    # 1. Active codes (combined)
    all_indices = np.concatenate([results_per_mod[m]["indices"] for m in modalities])
    active_codes = int(len(np.unique(all_indices)))

    # 2. ST <-> VJEPA agreement rate
    idx_st = results_per_mod["st"]["indices"]
    idx_vj = results_per_mod["vjepa"]["indices"]
    st_vj_agreement = float(np.sum(idx_st == idx_vj) / n)

    # 3. Post-VQ RSA ST vs VJEPA
    rsm_q_st = cosine_similarity_matrix(results_per_mod["st"]["z_q"])
    rsm_q_vj = cosine_similarity_matrix(results_per_mod["vjepa"]["z_q"])
    post_vq_rsa_st_vj, _ = safe_rsa(rsm_q_st, rsm_q_vj)

    # 4. Reconstruction cos_sim for ST
    recon_cos_st = results_per_mod["st"]["cos"]

    # 5. Preservation: original ST structure -> post-VQ
    rsm_orig_st = cosine_similarity_matrix(bases_dict["st"])
    preservation_st, _ = safe_rsa(rsm_orig_st, rsm_q_st)

    # Per-concept codes (for later analysis)
    per_concept = {}
    for ci, concept in enumerate(valid_phys):
        per_concept[concept] = {
            mod: int(results_per_mod[mod]["indices"][ci]) for mod in modalities
        }

    return {
        "active_codes": active_codes,
        "st_vj_agreement_rate": st_vj_agreement,
        "post_vq_rsa_st_vj": post_vq_rsa_st_vj,
        "recon_cos_st": recon_cos_st,
        "preservation_st_post_vq": preservation_st,
        "per_concept": per_concept,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    st_base, vj_base, cl_base, valid_phys = load_physical_bases()
    n = len(valid_phys)

    print("=" * 85)
    print(f"MULTI-SEED CONTRASTIVE CODEBOOK ({n} concepts, "
          f"{len(SEEDS)} seeds x {len(LAMBDA_VALUES)} lambdas = "
          f"{len(SEEDS) * len(LAMBDA_VALUES)} runs)")
    print("=" * 85)

    bases = {"st": st_base, "clip": cl_base, "vjepa": vj_base}

    # Load lambda=0 baseline
    baseline = None
    try:
        with open(output_dir / "codebook_st3way_results.json") as f:
            prev = json.load(f)
        baseline = prev.get("three_way_st_clip_vjepa", {})
        if baseline:
            print(f"  Loaded lambda=0 baseline from codebook_st3way_results.json")
    except FileNotFoundError:
        pass

    # Collect all results: {lambda_str: {seed_str: metrics_dict}}
    all_runs = {}
    total_runs = len(LAMBDA_VALUES) * len(SEEDS)
    run_i = 0

    for lam in LAMBDA_VALUES:
        lam_key = str(lam)
        all_runs[lam_key] = {}

        for seed in SEEDS:
            run_i += 1
            t0 = time.time()
            print(f"\n  [{run_i:2d}/{total_runs}] lambda={lam}, seed={seed}")

            # Set all seeds
            set_all_seeds(seed)

            # Augment with this seed
            data, labels = augment(bases, n_augment=30, sigma=0.1, seed=seed)

            # Fresh model
            model = SharedCodebookNWay(
                {"st": 768, "clip": 768, "vjepa": 1024},
                codebook_dim=256, n_codes=64
            ).to(DEVICE)

            # Train
            train_model_contrastive(
                model, data, labels, seed=seed,
                n_epochs=200, batch_size=32, lr=1e-3,
                commitment_weight=0.25, contrastive_lambda=lam,
                verbose=True
            )

            # Evaluate
            metrics = evaluate_compact(model, bases, valid_phys)
            elapsed = time.time() - t0

            print(f"    -> codes={metrics['active_codes']}, "
                  f"agree={metrics['st_vj_agreement_rate']:.0%}, "
                  f"rsa={metrics['post_vq_rsa_st_vj']:+.3f}, "
                  f"recon={metrics['recon_cos_st']:.3f}, "
                  f"pres={metrics['preservation_st_post_vq']:+.3f}  "
                  f"({elapsed:.1f}s)")

            all_runs[lam_key][str(seed)] = metrics

            del model
            torch.cuda.empty_cache()

    # ── Compute summaries ────────────────────────────────────────────────────

    metric_keys = ["active_codes", "st_vj_agreement_rate", "post_vq_rsa_st_vj",
                   "recon_cos_st", "preservation_st_post_vq"]

    summary = {}
    for lam_key in all_runs:
        seed_results = all_runs[lam_key]
        stats = {}
        for mk in metric_keys:
            vals = [seed_results[s][mk] for s in seed_results]
            stats[mk] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "values": vals,
            }
        summary[lam_key] = stats

    # Check if lambda=0.5 wins agreement in majority of seeds
    if "0.5" in all_runs and "0.1" in all_runs and "1.0" in all_runs:
        wins = 0
        for seed in SEEDS:
            s = str(seed)
            a_05 = all_runs["0.5"][s]["st_vj_agreement_rate"]
            a_01 = all_runs["0.1"][s]["st_vj_agreement_rate"]
            a_10 = all_runs["1.0"][s]["st_vj_agreement_rate"]
            if a_05 >= a_01 and a_05 >= a_10:
                wins += 1
        lambda_05_robust = wins >= 4
    else:
        wins = 0
        lambda_05_robust = False

    # ── Print summary table ──────────────────────────────────────────────────

    print(f"\n{'='*85}")
    print("SUMMARY TABLE: MEAN +/- STD ACROSS 5 SEEDS")
    print("=" * 85)

    header_labels = {
        "active_codes": "codes",
        "st_vj_agreement_rate": "ST-VJ agree",
        "post_vq_rsa_st_vj": "post-VQ RSA",
        "recon_cos_st": "recon cos",
        "preservation_st_post_vq": "pres ST",
    }

    # Header
    print(f"  {'lambda':>8s}", end="")
    for mk in metric_keys:
        print(f" | {header_labels[mk]:>15s}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in metric_keys:
        print(f"-+-{'-'*15}", end="")
    print()

    # Baseline row
    if baseline:
        bl_codes = baseline.get("code_utilization", {}).get("combined", "?")
        bl_agree = baseline.get("cross_modal", {}).get("st_vs_vjepa", {}).get("rate", 0)
        bl_rsa = baseline.get("rsa", {}).get("st_vs_vjepa", {}).get("post_vq", {}).get("r", 0)
        bl_recon = baseline.get("reconstruction", {}).get("st", {}).get("cos", 0)
        bl_pres = baseline.get("preservation", {}).get("st", {}).get("post_vq", 0)
        print(f"  {'0.0':>8s}"
              f" | {str(bl_codes):>15s}"
              f" | {bl_agree:>15.1%}"
              f" | {bl_rsa:>+14.3f}"
              f" | {bl_recon:>14.3f}"
              f" | {bl_pres:>+14.3f}")

    # Lambda rows
    for lam_key in sorted(summary.keys(), key=float):
        s = summary[lam_key]
        print(f"  {lam_key:>8s}", end="")
        for mk in metric_keys:
            m = s[mk]["mean"]
            sd = s[mk]["std"]
            if mk == "active_codes":
                print(f" | {m:>6.1f} +/- {sd:<5.1f}", end="")
            elif mk == "st_vj_agreement_rate":
                print(f" | {m:>5.0%} +/- {sd:<5.0%}", end="")
            else:
                print(f" | {m:>+5.3f} +/- {sd:<5.3f}", end="")
        print()

    # Per-seed detail for ST-VJ agreement
    print(f"\n  Per-seed ST <-> VJEPA agreement:")
    print(f"  {'seed':>8s}", end="")
    for lam_key in sorted(all_runs.keys(), key=float):
        print(f" | lam={lam_key:>4s}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in all_runs:
        print(f"-+-{'-'*9}", end="")
    print()
    for seed in SEEDS:
        s = str(seed)
        print(f"  {s:>8s}", end="")
        for lam_key in sorted(all_runs.keys(), key=float):
            rate = all_runs[lam_key][s]["st_vj_agreement_rate"]
            print(f" | {rate:>8.0%}", end="")
        print()

    print(f"\n  Lambda=0.5 wins agreement in {wins}/{len(SEEDS)} seeds "
          f"-> {'ROBUST' if lambda_05_robust else 'NOT ROBUST'}")

    # ── Save full results ────────────────────────────────────────────────────

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

    full_output = {
        "config": {
            "seeds": SEEDS,
            "lambda_values": LAMBDA_VALUES,
            "n_epochs": 200,
            "batch_size": 32,
            "n_augment": 30,
            "sigma": 0.1,
            "n_codes": 64,
            "codebook_dim": 256,
        },
        "runs": to_python(all_runs),
    }

    with open(output_dir / "codebook_multiseed_results.json", "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    summary_output = {
        "summary": to_python(summary),
        "lambda_0.5_wins": wins,
        "lambda_0.5_robust": lambda_05_robust,
        "verdict": (
            f"Lambda=0.5 wins ST-VJ agreement in {wins}/{len(SEEDS)} seeds. "
            f"{'Result is robust.' if lambda_05_robust else 'Result is NOT robust across seeds.'}"
        ),
    }
    if baseline:
        summary_output["baseline_lambda_0"] = {
            "active_codes": bl_codes,
            "st_vj_agreement_rate": bl_agree,
            "post_vq_rsa_st_vj": bl_rsa,
            "recon_cos_st": bl_recon,
            "preservation_st_post_vq": bl_pres,
        }

    with open(output_dir / "codebook_multiseed_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_python(summary_output), f, indent=2)

    print(f"\n{'='*85}")
    print("SAVED:")
    print(f"  lm_output/codebook_multiseed_results.json  (full per-run data)")
    print(f"  lm_output/codebook_multiseed_summary.json  (mean/std + verdict)")
    print("=" * 85)
