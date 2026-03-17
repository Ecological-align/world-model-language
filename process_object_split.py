"""
process_object_split.py
========================

Tests whether process concepts (rust, crack, spill, flow, etc.) show
different LM↔MAE vs LM↔CLIP/VideoMAE alignment patterns compared to
object/material concepts.

Hypothesis from ChatGPT review:
  MAE preserves mid-level texture/material structure (rust, crack, smoke).
  If process concepts align MORE with MAE than object concepts,
  that supports the "mid-level structure" mechanism.

  If process concepts align EQUALLY across models,
  the mid-level hypothesis is less supported.

This uses existing embeddings — no new data needed.

Output: lm_output/process_object_split_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict

DATA_DIR    = "lm_output/phrase_level"
OUTPUT_FILE = "lm_output/process_object_split_results.json"

# ── Process vs object concept classification ──────────────────────────────────
# Process concepts from Experiment 13 (dynamics/events)
PROCESS_CONCEPTS = {
    "fall", "bounce", "collision", "spill", "slide", "dissolve",
    "shatter", "ignite", "rust", "vibration", "compression", "flow",
    # Additional likely process concepts in phrase bank
    "melt", "burn", "freeze", "crack", "break", "splash", "drip",
    "scatter", "bend", "stretch", "pour", "boil"
}

CODEBOOK_DIM = 64
N_CODES      = 16
LAMBDA_CM    = 0.5
LAMBDA_DIV   = 0.1
N_RUNS       = 50
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VISUAL_MODELS = {
    "mae":       ("mae_hiddens_phrase.npy",          "MAE-Large"),
    "dinov2":    ("dinov2_hiddens_phrase.npy",         "DINOv2"),
    "clip":      ("clip_hiddens_phrase.npy",           "CLIP"),
    "vmae_k400": ("videomae_hiddens_phrase.npy",       "VideoMAE-K400"),
    "vjepa2":    ("vjepa2_hiddens_phrase.npy",         "V-JEPA2"),
}
LM_FILE = "lm_hiddens_phrase.npy"


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
        loss = rec + LAMBDA_CM * cm - LAMBDA_DIV * div
        agreement = (idx_a == idx_b).float().mean().item()
        return loss, agreement


def run_codebook(lm_emb, vis_emb, lm_dim, vis_dim, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cb  = VQCodebook(lm_dim, vis_dim, CODEBOOK_DIM, N_CODES)
    opt = torch.optim.Adam(cb.parameters(), lr=1e-3)
    lm_t  = torch.tensor(lm_emb,  dtype=torch.float32)
    vis_t = torch.tensor(vis_emb, dtype=torch.float32)
    for _ in range(300):
        cb.train(); opt.zero_grad()
        loss, _ = cb(lm_t, vis_t)
        loss.backward(); opt.step()
    cb.eval()
    with torch.no_grad():
        _, agr = cb(lm_t, vis_t)
    return agr * 100


def main():
    print("=" * 72)
    print("PROCESS vs OBJECT CONCEPT ALIGNMENT SPLIT")
    print("=" * 72)

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]

    all_concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    process_concepts = [c for c in all_concepts if c.lower() in PROCESS_CONCEPTS]
    object_concepts  = [c for c in all_concepts if c.lower() not in PROCESS_CONCEPTS]

    print(f"\nConcept split:")
    print(f"  Process concepts: {len(process_concepts)} — {process_concepts}")
    print(f"  Object/material concepts: {len(object_concepts)}")

    def concept_means(arr, concept_list):
        out = []
        for c in concept_list:
            idxs = [i for i, e in enumerate(event_index) if e["concept"] == c]
            if idxs:
                out.append(arr[idxs].mean(axis=0))
        return np.array(out)

    lm_raw  = np.load(os.path.join(DATA_DIR, LM_FILE))
    lm_dim  = lm_raw.shape[1]
    lm_proc = concept_means(lm_raw, process_concepts)
    lm_obj  = concept_means(lm_raw, object_concepts)

    results = {}

    for key, (fname, label) in VISUAL_MODELS.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            print(f"\n  [SKIP] {label}: {path} not found")
            continue

        vis_raw  = np.load(path)
        vis_dim  = vis_raw.shape[1]
        vis_proc = concept_means(vis_raw, process_concepts)
        vis_obj  = concept_means(vis_raw, object_concepts)

        print(f"\n  {label}")
        print(f"  ─────────────────────────────────────────────")

        proc_scores, obj_scores = [], []

        # Cap runs at available concepts (need >= 2)
        n_proc = len(lm_proc)
        n_obj  = len(lm_obj)

        if n_proc >= 2:
            for seed in range(N_RUNS):
                s = run_codebook(lm_proc, vis_proc, lm_dim, vis_dim, seed)
                proc_scores.append(s)
            proc_mean = np.mean(proc_scores)
            proc_std  = np.std(proc_scores)
            print(f"    Process:  {proc_mean:.1f}% ± {proc_std:.1f}")
        else:
            proc_mean = float('nan')
            print(f"    Process:  too few concepts ({n_proc})")

        if n_obj >= 2:
            for seed in range(N_RUNS):
                s = run_codebook(lm_obj, vis_obj, lm_dim, vis_dim, seed)
                obj_scores.append(s)
            obj_mean = np.mean(obj_scores)
            obj_std  = np.std(obj_scores)
            print(f"    Object:   {obj_mean:.1f}% ± {obj_std:.1f}")
        else:
            obj_mean = float('nan')
            print(f"    Object:   too few concepts ({n_obj})")

        if not (np.isnan(proc_mean) or np.isnan(obj_mean)):
            delta = proc_mean - obj_mean
            print(f"    Delta (process - object): {delta:+.1f}%")
            if delta > 3:
                print(f"    → {label} aligns MORE with process concepts")
            elif delta < -3:
                print(f"    → {label} aligns MORE with object concepts")
            else:
                print(f"    → Similar alignment across concept types")

        results[key] = {
            "label":      label,
            "process_n":  n_proc,
            "object_n":   n_obj,
            "process_mean": proc_mean,
            "object_mean":  obj_mean,
            "delta":        proc_mean - obj_mean if not (np.isnan(proc_mean) or np.isnan(obj_mean)) else None
        }

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — Process vs Object alignment delta per model")
    print(f"{'='*72}")
    print(f"\n  Hypothesis: MAE should show HIGHER alignment on process concepts")
    print(f"  (rust, crack, spill) because MAE preserves mid-level texture/")
    print(f"  material structure that language uses to describe physical processes.\n")

    print(f"  {'Model':<20}  {'Process':>9}  {'Object':>9}  {'Delta':>8}  Interpretation")
    print(f"  {'-'*70}")
    for key, r in results.items():
        pm = f"{r['process_mean']:.1f}%" if not np.isnan(r['process_mean']) else "n/a"
        om = f"{r['object_mean']:.1f}%"  if not np.isnan(r['object_mean'])  else "n/a"
        dm = f"{r['delta']:+.1f}%" if r['delta'] is not None else "n/a"
        interp = ""
        if r['delta'] is not None:
            if r['delta'] > 3:    interp = "process-favored"
            elif r['delta'] < -3: interp = "object-favored"
            else:                 interp = "similar"
        print(f"  {r['label']:<20}  {pm:>9}  {om:>9}  {dm:>8}  {interp}")

    print(f"\n  Key comparison: MAE delta vs VideoMAE delta")
    if "mae" in results and "vmae_k400" in results:
        mae_d  = results["mae"]["delta"]
        vmae_d = results["vmae_k400"]["delta"]
        if mae_d is not None and vmae_d is not None:
            diff = mae_d - vmae_d
            print(f"    MAE process-object delta:      {mae_d:+.1f}%")
            print(f"    VideoMAE process-object delta: {vmae_d:+.1f}%")
            if diff > 3:
                print(f"    → MAE preferentially aligns on process concepts (+{diff:.1f}%)")
                print(f"       Supports mid-level texture/material structure hypothesis.")
            elif diff < -3:
                print(f"    → VideoMAE preferentially aligns on process concepts ({diff:.1f}%)")
                print(f"       Counter to mid-level hypothesis — temporal model better on processes.")
            else:
                print(f"    → No differential (Δ={diff:+.1f}%). Process/object split doesn't")
                print(f"       distinguish MAE from VideoMAE.")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")
    print("Run this and paste the output back to update the paper.")


if __name__ == "__main__":
    main()
