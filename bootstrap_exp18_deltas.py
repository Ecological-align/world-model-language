"""
bootstrap_exp18_deltas.py
=========================

Computes 95% bootstrap confidence intervals on the process-object delta
values from Experiment 18 (process_object_split_results.json), and tests
whether the key pairwise comparisons are statistically separable:

  1. MAE delta vs VideoMAE-K400 delta  (9.4% gap — the factorial claim)
  2. MAE delta vs CLIP delta           (8.1% gap — cross-objective)
  3. MAE delta vs DINOv2 delta         (6.5% gap — same data, different obj)

The bootstrap re-samples across the N_RUNS codebook training seeds and
computes a delta distribution for each model, then derives CIs and
one-tailed p-values for each comparison.

Writes: lm_output/bootstrap_exp18_results.json
Prints: human-readable summary table

Run from repo root:
  PYTHONPATH=. .venv/Scripts/python.exe bootstrap_exp18_deltas.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

DATA_DIR    = "lm_output/phrase_level"
SPLIT_FILE  = "lm_output/process_object_split_results.json"
OUTPUT_FILE = "lm_output/bootstrap_exp18_results.json"

CODEBOOK_DIM = 64
N_CODES      = 16
LAMBDA_CM    = 0.5
LAMBDA_DIV   = 0.1
N_RUNS       = 50          # seeds per subset
N_BOOTSTRAP  = 2000        # bootstrap resamples
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VISUAL_MODELS = {
    "mae":       "mae_hiddens_phrase.npy",
    "dinov2":    "dinov2_hiddens_phrase.npy",
    "clip":      "clip_hiddens_phrase.npy",
    "vmae_k400": "videomae_hiddens_phrase.npy",
    "vjepa2":    "vjepa2_hiddens_phrase.npy",
}
LM_FILE = "lm_hiddens_phrase.npy"

PROCESS_CONCEPTS = {
    "fall","bounce","collision","spill","slide","dissolve",
    "shatter","ignite","rust","vibration","compression","flow",
    "melt","burn","freeze","crack","break","splash","drip",
    "scatter","bend","stretch","pour","boil"
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
        loss = rec + LAMBDA_CM*cm - LAMBDA_DIV*div
        return loss, (idx_a == idx_b).float().mean().item()


def run_codebook(lm_e, vis_e, seed):
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


def concept_means(arr, event_index, concept_list):
    out = []
    for c in concept_list:
        idxs = [i for i,e in enumerate(event_index) if e["concept"]==c]
        if idxs:
            out.append(arr[idxs].mean(axis=0))
    return np.array(out)


def main():
    print("="*72)
    print("BOOTSTRAP CIs — EXPERIMENT 18 PROCESS/OBJECT DELTAS")
    print("="*72)

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    all_concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    proc_concepts = [c for c in all_concepts if c.lower() in PROCESS_CONCEPTS]
    obj_concepts  = [c for c in all_concepts if c.lower() not in PROCESS_CONCEPTS]

    print(f"\nSubsets: {len(proc_concepts)} process, {len(obj_concepts)} object concepts")

    lm_raw  = np.load(os.path.join(DATA_DIR, LM_FILE))
    lm_proc = concept_means(lm_raw, event_index, proc_concepts)
    lm_obj  = concept_means(lm_raw, event_index, obj_concepts)

    # Collect per-seed scores for each model × subset
    raw_scores = {}  # key -> {"proc": [s0..sN], "obj": [s0..sN]}

    for key, fname in VISUAL_MODELS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"  [SKIP] {key}: not found")
            continue
        vis_raw  = np.load(path)
        vis_proc = concept_means(vis_raw, event_index, proc_concepts)
        vis_obj  = concept_means(vis_raw, event_index, obj_concepts)

        print(f"\n  Collecting {N_RUNS} seeds for {key}...")
        proc_scores, obj_scores = [], []
        for seed in range(N_RUNS):
            proc_scores.append(run_codebook(lm_proc, vis_proc, seed))
            obj_scores.append(run_codebook(lm_obj,  vis_obj,  seed))
            if (seed+1) % 10 == 0:
                print(f"    seed {seed+1}/{N_RUNS} — proc {np.mean(proc_scores):.1f}% | obj {np.mean(obj_scores):.1f}%")

        raw_scores[key] = {"proc": proc_scores, "obj": obj_scores}

    print(f"\n\n{'='*72}")
    print("BOOTSTRAP CIs ON DELTAS  (2000 resamples)")
    print("="*72)

    delta_distributions = {}
    summary = {}

    for key, scores in raw_scores.items():
        proc = np.array(scores["proc"])
        obj  = np.array(scores["obj"])

        # Bootstrap: resample seeds with replacement, compute delta each time
        rng = np.random.default_rng(42)
        boot_deltas = []
        for _ in range(N_BOOTSTRAP):
            idx_p = rng.integers(0, len(proc), size=len(proc))
            idx_o = rng.integers(0, len(obj),  size=len(obj))
            boot_deltas.append(proc[idx_p].mean() - obj[idx_o].mean())

        boot_deltas = np.array(boot_deltas)
        delta_mean = proc.mean() - obj.mean()
        ci_lo, ci_hi = np.percentile(boot_deltas, [2.5, 97.5])

        delta_distributions[key] = boot_deltas.tolist()
        summary[key] = {
            "proc_mean": float(proc.mean()),
            "obj_mean":  float(obj.mean()),
            "delta_mean": float(delta_mean),
            "ci_95_lo": float(ci_lo),
            "ci_95_hi": float(ci_hi),
        }

    labels = {
        "mae":       "MAE-Large",
        "dinov2":    "DINOv2",
        "vjepa2":    "V-JEPA2",
        "clip":      "CLIP",
        "vmae_k400": "VideoMAE-K400",
    }

    print(f"\n  {'Model':<18}  {'Process':>9}  {'Object':>9}  {'Delta':>8}  {'95% CI':>20}")
    print(f"  {'-'*72}")
    for key in ["mae","dinov2","vjepa2","clip","vmae_k400"]:
        if key not in summary: continue
        r = summary[key]
        print(f"  {labels[key]:<18}  {r['proc_mean']:>8.1f}%  {r['obj_mean']:>8.1f}%  "
              f"{r['delta_mean']:>7.1f}%  [{r['ci_95_lo']:.1f}%, {r['ci_95_hi']:.1f}%]")

    # Pairwise tests: does MAE delta significantly exceed others?
    print(f"\n\n{'='*72}")
    print("PAIRWISE COMPARISONS — MAE delta vs others")
    print("  (one-tailed p: fraction of bootstrap resamples where comparison model delta > MAE delta)")
    print("="*72)

    mae_boots = np.array(delta_distributions.get("mae", []))
    comparisons = {}

    for key in ["vmae_k400", "clip", "dinov2", "vjepa2"]:
        if key not in delta_distributions: continue
        other_boots = np.array(delta_distributions[key])
        # Bootstrap difference distribution
        rng2 = np.random.default_rng(99)
        diff_boots = []
        for _ in range(N_BOOTSTRAP):
            m = mae_boots[rng2.integers(0, len(mae_boots), len(mae_boots))].mean()
            o = other_boots[rng2.integers(0, len(other_boots), len(other_boots))].mean()
            diff_boots.append(m - o)
        diff_boots = np.array(diff_boots)
        diff_ci = np.percentile(diff_boots, [2.5, 97.5])
        # p-value: fraction where diff <= 0 (i.e., MAE not better)
        p_val = (diff_boots <= 0).mean()

        mae_d   = summary["mae"]["delta_mean"]
        other_d = summary[key]["delta_mean"]
        gap     = mae_d - other_d

        comparisons[key] = {
            "gap": float(gap),
            "diff_ci_lo": float(diff_ci[0]),
            "diff_ci_hi": float(diff_ci[1]),
            "p_one_tailed": float(p_val),
        }

        sig = "✓ separable" if diff_ci[0] > 0 else ("marginal" if p_val < 0.10 else "overlapping")
        print(f"\n  MAE vs {labels[key]}:")
        print(f"    Gap in deltas:  {gap:+.1f}%")
        print(f"    95% CI of gap:  [{diff_ci[0]:.1f}%, {diff_ci[1]:.1f}%]")
        print(f"    p (one-tailed): {p_val:.3f}")
        print(f"    Result:         {sig}")

    # Final summary
    print(f"\n\n{'='*72}")
    print("SUMMARY FOR PAPER")
    print("="*72)
    print()
    mae_r = summary.get("mae", {})
    vm_r  = summary.get("vmae_k400", {})
    if mae_r and vm_r:
        gap_c = comparisons.get("vmae_k400", {})
        print(f"  Key factorial comparison (MAE vs VideoMAE-K400, same objective):")
        print(f"    MAE delta:       {mae_r['delta_mean']:.1f}% [{mae_r['ci_95_lo']:.1f}, {mae_r['ci_95_hi']:.1f}]")
        print(f"    VideoMAE delta:  {vm_r['delta_mean']:.1f}% [{vm_r['ci_95_lo']:.1f}, {vm_r['ci_95_hi']:.1f}]")
        print(f"    Gap:             {gap_c.get('gap', 0):+.1f}% [{gap_c.get('diff_ci_lo',0):.1f}, {gap_c.get('diff_ci_hi',0):.1f}]")
        lo = gap_c.get("diff_ci_lo", 0)
        if lo > 0:
            print(f"    → Gap CI entirely above zero: MAE-VideoMAE delta difference is statistically separable.")
        else:
            print(f"    → CI crosses zero: gap is directional but not statistically separable at 95%.")
        print(f"    → Correct framing: 'MAE shows a larger process-object alignment gap than VideoMAE'")
        print(f"      {'(statistically separable at 95% CI)' if lo>0 else '(directional, CIs overlap at 95%)'}")

    results = {
        "summary": summary,
        "comparisons": comparisons,
        "n_runs": N_RUNS,
        "n_bootstrap": N_BOOTSTRAP,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
