"""
gemma2_arch_control.py
======================

Architecture control experiment for Gemma-2-9B.

Tests:
  Gemma2 ↔ MAE-Base (ViT-B, 768-dim, image, reconstruction)
  Gemma2 ↔ VideoMAE-K400 (ViT-B, 768-dim, video, reconstruction)
  Gemma2 ↔ VideoMAE-SSv2 (ViT-B, 768-dim, video, reconstruction)

All three visual models are identical ViT-Base architectures.
This rules out model capacity as the explanation for any alignment gap.

Also runs MAE-Large, CLIP, V-JEPA2 for a full 6-model comparison
consistent with the cross-LLM table in the paper.

Adds a 5th row to the architecture control table:
  Mistral-7B:   MAE higher (3/4 → resolved with Gemma2)
  Qwen2.5-7B:   MAE higher ✓
  Llama-3.1-8B: Marginal Δ=2.3%  ← the disputed exception
  Qwen2.5-32B:  MAE higher ✓
  Gemma-2-9B:   ?

Output: lm_output/gemma2_arch_control_results.json
        Prints the full results table

Run from repo root:
  PYTHONPATH=. .venv/Scripts/python.exe gemma2_arch_control.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

DATA_DIR    = "lm_output/phrase_level"
OUT_FILE    = "lm_output/gemma2_arch_control_results.json"

CODEBOOK_DIM = 64
N_CODES      = 16
LAMBDA_CM    = 0.5
LAMBDA_DIV   = 0.1
N_RUNS       = 50
N_SPLITS     = 10
N_BOOTSTRAP  = 2000
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LM_FILE = "gemma2_hiddens_phrase.npy"   # extract_gemma2.py output

VISUAL_MODELS = {
    "mae_large":   ("mae_hiddens_phrase.npy",          "MAE-Large"),
    "mae_base":    ("mae_base_hiddens_phrase.npy",     "MAE-Base"),
    "dinov2":      ("dinov2_hiddens_phrase.npy",       "DINOv2"),
    "clip":        ("clip_hiddens_phrase.npy",         "CLIP"),
    "vmae_k400":   ("videomae_hiddens_phrase.npy",     "VideoMAE-K400"),
    "vmae_ssv2":   ("videomae_ssv2_hiddens_phrase.npy","VideoMAE-SSv2"),
    "vjepa2":      ("vjepa2_hiddens_phrase.npy",       "V-JEPA2"),
}


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
               0.25*(F.mse_loss(za, qa.detach()) + F.mse_loss(zb, qb.detach())))
        sim    = torch.mm(za, zb.T) / 0.07
        labels = torch.arange(len(za), device=a.device)
        cm     = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        avg_a  = F.softmax(-torch.cdist(za, self.codebook.weight)*5, dim=-1).mean(0)
        avg_b  = F.softmax(-torch.cdist(zb, self.codebook.weight)*5, dim=-1).mean(0)
        div    = (-(avg_a*(avg_a+1e-8).log()).sum()
                  -(avg_b*(avg_b+1e-8).log()).sum()) / 2
        return rec + LAMBDA_CM*cm - LAMBDA_DIV*div, (idx_a==idx_b).float().mean().item()


def run_one(lm_e, vis_e, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cb  = VQCodebook(lm_e.shape[1], vis_e.shape[1], CODEBOOK_DIM, N_CODES)
    opt = torch.optim.Adam(cb.parameters(), lr=1e-3)
    lm_t  = torch.tensor(lm_e,  dtype=torch.float32)
    vis_t = torch.tensor(vis_e, dtype=torch.float32)
    for _ in range(300):
        cb.train(); opt.zero_grad()
        loss, _ = cb(lm_t, vis_t)
        loss.backward(); opt.step()
    cb.eval()
    with torch.no_grad():
        _, agr = cb(lm_t, vis_t)
    return agr * 100


def concept_means(arr, event_index):
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    out = []
    for c in concepts:
        idxs = [i for i,e in enumerate(event_index) if e["concept"]==c]
        out.append(arr[idxs].mean(axis=0))
    return np.array(out)


def bootstrap_ci(scores, n_boot=N_BOOTSTRAP):
    rng = np.random.default_rng(42)
    boot = [rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])


def main():
    print("="*72)
    print("ARCHITECTURE CONTROL — GEMMA-2-9B")
    print("="*72)

    lm_path = os.path.join(DATA_DIR, LM_FILE)
    if not os.path.exists(lm_path):
        print(f"\nERROR: {lm_path} not found.")
        print("Run extract_gemma2.py first.")
        sys.exit(1)

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]

    lm_raw  = np.load(lm_path)
    lm_conc = concept_means(lm_raw, event_index)
    n_conc  = len(lm_conc)
    print(f"\nGemma-2-9B: {lm_raw.shape}  →  {n_conc} concept means")

    results = {}

    for key, (fname, label) in VISUAL_MODELS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {label}")
            continue

        vis_raw  = np.load(path)
        vis_conc = concept_means(vis_raw, event_index)
        print(f"\n  {label}  (lm_dim={lm_raw.shape[1]}, vis_dim={vis_raw.shape[1]})")

        scores = [run_one(lm_conc, vis_conc, seed) for seed in range(N_RUNS)]
        mean   = np.mean(scores)
        ci     = bootstrap_ci(np.array(scores))
        print(f"    Agreement: {mean:.1f}%  95% CI [{ci[0]:.1f}, {ci[1]:.1f}]")

        results[key] = {
            "label":    label,
            "mean":     float(mean),
            "ci_lo":    float(ci[0]),
            "ci_hi":    float(ci[1]),
            "scores":   [float(s) for s in scores],
        }

    # Print architecture control summary
    print(f"\n\n{'='*72}")
    print("ARCHITECTURE CONTROL SUMMARY  (Gemma-2-9B)")
    print("All ViT-Base models: same architecture, 768-dim, 86M params")
    print("="*72)
    print(f"\n  {'Model':<22}  {'Agreement':>10}  {'95% CI':>20}  {'Notes'}")
    print(f"  {'-'*72}")
    for key in ["mae_base","vmae_k400","vmae_ssv2","mae_large","dinov2","clip","vjepa2"]:
        if key not in results: continue
        r = results[key]
        note = ""
        if key == "mae_base":    note = "← architecture-matched"
        if key == "vmae_k400":  note = "← architecture-matched"
        if key == "vmae_ssv2":  note = "← architecture-matched"
        print(f"  {r['label']:<22}  {r['mean']:>9.1f}%  [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]  {note}")

    # Verdict
    if "mae_base" in results and "vmae_k400" in results:
        mae_m  = results["mae_base"]["mean"]
        vmae_m = results["vmae_k400"]["mean"]
        gap    = mae_m - vmae_m
        print(f"\n  MAE-Base vs VideoMAE-K400 (arch-controlled): Δ = {gap:+.1f}%")
        mae_ci  = (results["mae_base"]["ci_lo"],  results["mae_base"]["ci_hi"])
        vmae_ci = (results["vmae_k400"]["ci_lo"], results["vmae_k400"]["ci_hi"])
        if mae_m > vmae_m:
            print(f"  MAE-Base HIGHER  [{mae_ci[0]:.1f},{mae_ci[1]:.1f}] vs [{vmae_ci[0]:.1f},{vmae_ci[1]:.1f}]")
            ci_overlap = mae_ci[0] < vmae_ci[1]
            print(f"  CI overlap: {'yes — directional only' if ci_overlap else 'no — statistically separable'}")
            print(f"  Verdict: Architecture confound ruled out ✓" if gap > 2 else "  Verdict: Marginal")
        else:
            print(f"  VideoMAE HIGHER or equal — architecture confound NOT ruled out ✗")

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")

    # Cross-LLM summary for paper table
    print(f"\n\n{'='*72}")
    print("ADD TO PAPER — Architecture Control Table (5th row)")
    print("="*72)
    if "mae_base" in results and "vmae_k400" in results and "vmae_ssv2" in results:
        r_m = results["mae_base"]
        r_k = results["vmae_k400"]
        r_s = results["vmae_ssv2"]
        gap = r_m["mean"] - r_k["mean"]
        verdict = "Ruled out ✓" if gap > 2 else "Marginal"
        print(f"\n  | Gemma-2-9B | {r_m['mean']:.1f}% [{r_m['ci_lo']:.1f}, {r_m['ci_hi']:.1f}] | "
              f"{r_k['mean']:.1f}% [{r_k['ci_lo']:.1f}, {r_k['ci_hi']:.1f}] | "
              f"{r_s['mean']:.1f}% [{r_s['ci_lo']:.1f}, {r_s['ci_hi']:.1f}] | {verdict} |")


if __name__ == "__main__":
    main()
