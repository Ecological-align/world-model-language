"""
finetune_adapter.py
===================

Experiment 16: Adapter fine-tuning — measuring the cost of bridging
visual encoders to language model alignment.

WHAT THIS EXPERIMENT TESTS
===========================
All previous experiments measured alignment as a fixed property of
pre-trained models. This experiment asks: can alignment be induced
through training, and at what cost?

For each visual encoder, we train a small MLP adapter on top of the
frozen embeddings using the codebook alignment loss against frozen
Mistral embeddings. We then measure:

  1. Alignment ceiling: what is the maximum alignment each model can reach?
  2. Convergence speed: how many steps to reach a target alignment level?
  3. Bridging cost: how many adapter parameters are needed?

HYPOTHESIS
==========
If the 2×2 finding is real (image × reconstruction = structurally similar
to LM), then:

  MAE: adapter needs few steps, reaches high ceiling easily
  DINOv2: moderate steps, moderate ceiling
  VideoMAE: many steps, lower ceiling — geometry is more distant
  V-JEPA 2: hardest — most structurally distant from LM

The "cost of bridging" quantifies in training terms what the alignment
probe measures in representational terms.

ARCHITECTURE
============
Adapter: frozen_embedding → LayerNorm → Linear → GELU → Linear → adapted_embedding
The adapter is small (2 linear layers) to avoid overfitting the alignment signal.
Three adapter sizes tested: Small (64-dim), Medium (256-dim), Large (512-dim).

The codebook + contrastive loss is identical to existing experiments.
Adapter is trained on top of frozen embeddings — no backprop into the encoder.

KEY OUTPUTS
===========
  1. Convergence curves: alignment vs training epoch for each visual model
  2. Ceiling table: final alignment for each model × adapter size
  3. Steps-to-threshold: epochs to reach MAE's natural alignment (~24%)
  4. Adapter efficiency: alignment gain per parameter

Results saved to: lm_output/adapter_finetune_results.json
Curves saved to: lm_output/adapter_curves.json

Usage:
  python finetune_adapter.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR      = "lm_output/phrase_level"
RESULTS_FILE  = "lm_output/adapter_finetune_results.json"
CURVES_FILE   = "lm_output/adapter_curves.json"

# Codebook (same as all prior experiments)
CODEBOOK_DIM  = 64
N_CODES       = 16
LAMBDA_CM     = 0.5
LAMBDA_DIV    = 0.1

# Training
LR            = 1e-3
EPOCHS        = 600          # enough to see convergence plateau
LOG_EVERY     = 20           # record alignment every N epochs
N_RUNS        = 5            # seeds for stability

# Adapter sizes to test (hidden dim of the 2-layer MLP)
ADAPTER_SIZES = {
    "small":  64,
    "medium": 256,
    "large":  512,
}

# Target alignment: MAE's natural level (~24% avg across LLMs, ~17.7% for Mistral)
# Use Mistral-specific value as the threshold
MAE_NATURAL_ALIGNMENT = 17.7   # Mistral-7B LM↔MAE from Experiment 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Visual models to test ──────────────────────────────────────────────────────
# Ordered by expected difficulty (easiest to hardest to align)
VISUAL_MODELS = {
    "mae":          ("mae_hiddens_phrase.npy",            "MAE-Large",      "image", "reconstruction"),
    "dinov2":       ("dinov2_hiddens_phrase.npy",          "DINOv2",         "image", "distillation"),
    "clip":         ("clip_hiddens_phrase.npy",            "CLIP",           "image", "contrastive"),
    "vmae_k400":    ("videomae_hiddens_phrase.npy",        "VideoMAE-K400",  "video", "reconstruction"),
    "vmae_ssv2":    ("videomae_ssv2_hiddens_phrase.npy",   "VideoMAE-SSv2",  "video", "reconstruction"),
    "vjepa2":       ("vjepa2_hiddens_phrase.npy",          "V-JEPA2",        "video", "temporal_pred"),
}

LM_FILE = "lm_hiddens_phrase.npy"   # Mistral-7B (reference LM)


# ── Adapter ────────────────────────────────────────────────────────────────────

class Adapter(nn.Module):
    """
    2-layer MLP adapter on top of frozen visual embeddings.
    Maps: frozen_dim → hidden_dim → frozen_dim
    Residual connection preserves original structure.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm  = nn.LayerNorm(input_dim)
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, input_dim)
        self.n_params = sum(p.numel() for p in self.parameters())
        # Init near identity — start from natural alignment, then improve
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = F.gelu(self.fc1(out))
        out = self.fc2(out)
        return residual + out   # residual: adapter starts as identity


# ── VQ Codebook ────────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    def __init__(self, dim_a, dim_b, embed_dim, n_codes):
        super().__init__()
        self.proj_a   = nn.Linear(dim_a, embed_dim)
        self.proj_b   = nn.Linear(dim_b, embed_dim)
        self.codebook = nn.Embedding(n_codes, embed_dim)
        nn.init.orthogonal_(self.codebook.weight)

    def quantize(self, z):
        d   = torch.cdist(z, self.codebook.weight)
        idx = d.argmin(dim=-1)
        return idx, self.codebook(idx)

    def forward(self, a, b):
        za = F.normalize(self.proj_a(a), dim=-1)
        zb = F.normalize(self.proj_b(b), dim=-1)
        idx_a, qa = self.quantize(za)
        idx_b, qb = self.quantize(zb)

        rec = (F.mse_loss(qa, za.detach()) + F.mse_loss(qb, zb.detach()) +
               0.25 * (F.mse_loss(za, qa.detach()) + F.mse_loss(zb, qb.detach())))

        sim    = torch.mm(za, zb.T) / 0.07
        labels = torch.arange(len(za), device=a.device)
        cm     = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

        avg_a  = F.softmax(-torch.cdist(za, self.codebook.weight) * 5, dim=-1).mean(0)
        avg_b  = F.softmax(-torch.cdist(zb, self.codebook.weight) * 5, dim=-1).mean(0)
        div    = (-(avg_a * (avg_a + 1e-8).log()).sum()
                  - (avg_b * (avg_b + 1e-8).log()).sum()) / 2

        loss      = rec + LAMBDA_CM * cm - LAMBDA_DIV * div
        agreement = (idx_a == idx_b).float().mean().item()
        active    = len(idx_a.unique())
        return loss, agreement, active


# ── Training ───────────────────────────────────────────────────────────────────

def train_adapter(lm_concept, vis_concept, vis_dim, lm_dim,
                  adapter_hidden, seed, epochs=EPOCHS):
    """
    Train an adapter on vis_concept to align with lm_concept.

    Returns:
      curve: list of (epoch, train_agreement, test_agreement) tuples
      final_train: final training agreement
      final_test: final test agreement
      steps_to_threshold: epochs to first reach MAE_NATURAL_ALIGNMENT on test
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N     = len(lm_concept)
    perm  = np.random.permutation(N)
    split = int(0.8 * N)
    tr_i, te_i = perm[:split], perm[split:]

    lm_tr = torch.tensor(lm_concept[tr_i], dtype=torch.float32)
    lm_te = torch.tensor(lm_concept[te_i], dtype=torch.float32)
    vis_tr = torch.tensor(vis_concept[tr_i], dtype=torch.float32)
    vis_te = torch.tensor(vis_concept[te_i], dtype=torch.float32)

    adapter  = Adapter(vis_dim, adapter_hidden)
    codebook = VQCodebook(lm_dim, vis_dim, CODEBOOK_DIM, N_CODES)
    params   = list(adapter.parameters()) + list(codebook.parameters())
    opt      = torch.optim.Adam(params, lr=LR)

    curve              = []
    steps_to_threshold = None

    for epoch in range(1, epochs + 1):
        adapter.train(); codebook.train()
        opt.zero_grad()
        adapted = adapter(vis_tr)
        loss, tr_agr, _ = codebook(lm_tr, adapted)
        loss.backward()
        opt.step()

        if epoch % LOG_EVERY == 0 or epoch == epochs:
            adapter.eval(); codebook.eval()
            with torch.no_grad():
                adapted_te = adapter(vis_te)
                _, te_agr, active = codebook(lm_te, adapted_te)
            te_pct = te_agr * 100
            tr_pct = tr_agr * 100
            curve.append((epoch, tr_pct, te_pct))

            # Track first epoch to reach MAE natural alignment on test
            if steps_to_threshold is None and te_pct >= MAE_NATURAL_ALIGNMENT:
                steps_to_threshold = epoch

    final_train = curve[-1][1]
    final_test  = curve[-1][2]

    return curve, final_train, final_test, steps_to_threshold


# ── Baseline (no adapter) ──────────────────────────────────────────────────────

def baseline_alignment(lm_concept, vis_concept, lm_dim, vis_dim, seed):
    """Alignment with no adapter — frozen embeddings only."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    N     = len(lm_concept)
    perm  = np.random.permutation(N)
    split = int(0.8 * N)
    tr_i, te_i = perm[:split], perm[split:]

    lm_tr  = torch.tensor(lm_concept[tr_i], dtype=torch.float32)
    lm_te  = torch.tensor(lm_concept[te_i], dtype=torch.float32)
    vis_tr = torch.tensor(vis_concept[tr_i], dtype=torch.float32)
    vis_te = torch.tensor(vis_concept[te_i], dtype=torch.float32)

    codebook = VQCodebook(lm_dim, vis_dim, CODEBOOK_DIM, N_CODES)
    opt      = torch.optim.Adam(codebook.parameters(), lr=LR)

    for _ in range(300):
        codebook.train()
        opt.zero_grad()
        loss, _, _ = codebook(lm_tr, vis_tr)
        loss.backward()
        opt.step()

    codebook.eval()
    with torch.no_grad():
        _, te_agr, _ = codebook(lm_te, vis_te)
    return te_agr * 100


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("EXPERIMENT 16: ADAPTER FINE-TUNING")
    print("Cost of bridging visual encoders to LM alignment")
    print("=" * 72)
    print(f"MAE natural alignment threshold: {MAE_NATURAL_ALIGNMENT}%")
    print(f"Adapter sizes: {ADAPTER_SIZES}")
    print(f"Epochs: {EPOCHS}  |  Log every: {LOG_EVERY}  |  Runs: {N_RUNS}")
    print(f"Device: {DEVICE}\n")

    # Load event index
    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    concepts    = list(dict.fromkeys(e["concept"] for e in event_index))

    def concept_means(arr):
        out = []
        for c in concepts:
            idxs = [i for i, e in enumerate(event_index) if e["concept"] == c]
            out.append(arr[idxs].mean(axis=0))
        return np.array(out)

    # Load LM (reference target — frozen)
    lm_raw     = np.load(os.path.join(DATA_DIR, LM_FILE))
    lm_concept = concept_means(lm_raw)
    lm_dim     = lm_raw.shape[1]
    print(f"LM (Mistral-7B): {lm_raw.shape}  concept-level: {lm_concept.shape}\n")

    # Load visual models
    vis_data = {}
    for key, (fname, label, dtype, obj) in VISUAL_MODELS.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            arr = np.load(path)
            vis_data[key] = (concept_means(arr), label, arr.shape[1], dtype, obj)
            print(f"  Loaded {label:<20} shape={arr.shape}")
        else:
            print(f"  [SKIP] {label}: {path} not found")
    print()

    all_results = {}
    all_curves  = {}

    for vis_key, (vis_concept, vis_label, vis_dim, dtype, obj) in vis_data.items():
        print(f"{'='*72}")
        print(f"Visual model: {vis_label}  [{dtype}, {obj}]  dim={vis_dim}")
        print(f"{'='*72}")

        model_results = {"label": vis_label, "dtype": dtype, "obj": obj}
        model_curves  = {}

        # ── Baseline (no adapter) ───────────────────────────────────────────
        base_scores = []
        for seed in range(N_RUNS):
            b = baseline_alignment(lm_concept, vis_concept, lm_dim, vis_dim, seed)
            base_scores.append(b)
        base_mean = np.mean(base_scores)
        base_std  = np.std(base_scores)
        print(f"\n  Baseline (no adapter): {base_mean:.1f}% ± {base_std:.1f}%")
        model_results["baseline"] = {"mean": base_mean, "std": base_std}

        # ── Adapter sizes ────────────────────────────────────────────────────
        for size_name, hidden_dim in ADAPTER_SIZES.items():
            n_params = Adapter(vis_dim, hidden_dim).n_params
            print(f"\n  Adapter [{size_name}, hidden={hidden_dim}, params={n_params:,}]")

            run_finals   = []
            run_ceilings = []
            run_thresholds = []
            size_curves  = []

            for seed in range(N_RUNS):
                curve, final_tr, final_te, thresh = train_adapter(
                    lm_concept, vis_concept, vis_dim, lm_dim,
                    hidden_dim, seed
                )
                run_finals.append(final_te)
                run_ceilings.append(final_te)
                run_thresholds.append(thresh)
                size_curves.append(curve)

                thresh_str = f"{thresh}" if thresh else "never"
                print(f"    seed={seed}  final_test={final_te:.1f}%  "
                      f"steps_to_threshold={thresh_str}")

            mean_final   = np.mean(run_finals)
            std_final    = np.std(run_finals)
            mean_thresh  = np.mean([t for t in run_thresholds if t is not None]) \
                           if any(t for t in run_thresholds) else None
            reached_pct  = sum(1 for t in run_thresholds if t is not None) / N_RUNS * 100

            gain         = mean_final - base_mean
            print(f"    → Final: {mean_final:.1f}% ± {std_final:.1f}%  "
                  f"(+{gain:.1f}% vs baseline)")
            if mean_thresh:
                print(f"    → Steps to MAE threshold ({MAE_NATURAL_ALIGNMENT}%): "
                      f"{mean_thresh:.0f} epochs  ({reached_pct:.0f}% of runs reached it)")
            else:
                print(f"    → Never reached MAE threshold ({MAE_NATURAL_ALIGNMENT}%)")

            model_results[size_name] = {
                "hidden_dim":      hidden_dim,
                "n_params":        n_params,
                "baseline":        base_mean,
                "final_mean":      mean_final,
                "final_std":       std_final,
                "gain":            gain,
                "steps_to_threshold": mean_thresh,
                "reached_threshold_pct": reached_pct,
            }
            # Average curve across runs for plotting
            avg_curve = []
            n_points  = len(size_curves[0])
            for pt_i in range(n_points):
                ep    = size_curves[0][pt_i][0]
                tr_m  = np.mean([c[pt_i][1] for c in size_curves])
                te_m  = np.mean([c[pt_i][2] for c in size_curves])
                avg_curve.append([ep, tr_m, te_m])
            model_curves[size_name] = avg_curve

        all_results[vis_key] = model_results
        all_curves[vis_key]  = model_curves

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("SUMMARY — Bridging cost comparison")
    print(f"{'='*72}")
    print(f"\n  {'Model':<20}  {'Baseline':>9}  {'Medium adapter':>15}  "
          f"{'Gain':>6}  {'Steps to MAE':>14}  {'Reached?':>9}")
    print(f"  {'-'*74}")

    for vis_key, res in all_results.items():
        base    = res["baseline"]["mean"]
        med     = res.get("medium", {})
        final   = med.get("final_mean", float("nan"))
        gain    = med.get("gain", float("nan"))
        steps   = med.get("steps_to_threshold")
        reached = med.get("reached_threshold_pct", 0)
        steps_s = f"{steps:.0f}" if steps else "never"
        print(f"  {res['label']:<20}  {base:>8.1f}%  {final:>14.1f}%  "
              f"{gain:>+5.1f}%  {steps_s:>14}  {reached:>8.0f}%")

    print(f"\n  MAE natural alignment threshold: {MAE_NATURAL_ALIGNMENT}%")
    print(f"  Interpretation:")
    print(f"    Models that reach the threshold with fewer steps are")
    print(f"    structurally closer to LM geometry.")
    print(f"    Models that never reach it have a fundamentally different")
    print(f"    representational structure that adapters cannot bridge.")

    # ── Key comparison: MAE vs VideoMAE ───────────────────────────────────────
    print(f"\n{'='*72}")
    print("KEY COMPARISON: MAE vs VideoMAE-K400 (same objective, different data)")
    print(f"{'='*72}")
    if "mae" in all_results and "vmae_k400" in all_results:
        mae_res  = all_results["mae"]["medium"]
        vmae_res = all_results["vmae_k400"]["medium"]
        if mae_res and vmae_res:
            print(f"  MAE baseline → ceiling:  {all_results['mae']['baseline']['mean']:.1f}% → {mae_res['final_mean']:.1f}%  "
                  f"(+{mae_res['gain']:.1f}%)")
            print(f"  VideoMAE baseline → ceiling: {all_results['vmae_k400']['baseline']['mean']:.1f}% → {vmae_res['final_mean']:.1f}%  "
                  f"(+{vmae_res['gain']:.1f}%)")
            ceiling_gap = mae_res['final_mean'] - vmae_res['final_mean']
            print(f"  Ceiling gap after adaptation: {ceiling_gap:+.1f}%")
            if ceiling_gap > 3:
                print(f"  → Even after fine-tuning, video-trained models cannot fully")
                print(f"    close the gap to image-reconstruction alignment.")
                print(f"    The geometry difference is structural, not just surface-level.")
            else:
                print(f"  → Adapter bridges the gap — visual geometry is flexible enough")
                print(f"    to be adapted toward language alignment regardless of training data.")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(CURVES_FILE, "w") as f:
        json.dump(all_curves, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Curves saved to {CURVES_FILE}")
    print(f"\nNext: run plot_adapter_curves.py to visualize convergence")


if __name__ == "__main__":
    main()
