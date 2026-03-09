"""
process_concept_probe.py
========================

Tests whether the temporal dynamics gap is concept-type-dependent.

Hypothesis (from reviewer critique):
  If the temporal gap is real, V-JEPA 2 should diverge MORE from static models
  on PROCESS concepts (fall, bounce, collision...) than on OBJECT concepts
  (apple, rope, stone...).

  Specifically we predict:
    LM↔VIS agreement:  objects > processes   (V-JEPA 2 more isolated on processes)
    LM↔MAE agreement:  objects ≈ processes   (MAE doesn't do temporal, no difference)
    VIS↔MAE agreement: objects > processes   (V-JEPA 2's temporal training matters more)

If this pattern holds, it rules out the "input domain" confound:
  The gap isn't just V-JEPA 2 struggling with static images —
  it reflects a genuine representational difference on temporally-rich concepts.

Usage:
  1. First run: python extract_process_concepts.py  (extracts embeddings for new 12 concepts)
  2. Then run:  python process_concept_probe.py      (runs the comparison)

Or run this script with --extract to do both.
"""

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

# ── CONFIG ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "lm_output/phrase_level"
RESULTS_FILE = "lm_output/process_concept_probe_results.json"

N_RUNS = 50          # codebook runs per split
N_SPLITS = 10        # train/test splits
N_SEEDS = 5          # seeds per config
EMBED_DIM = 64
N_CODES = 16
LAMBDA_CM = 0.5
LAMBDA_DIV = 0.1

# Concept type labels
OBJECT_CONCEPTS = [
    "apple", "chair", "stone", "rope", "door", "container", "mirror",
    "knife", "wheel", "wall", "bridge", "ladder", "leaf", "thread",
    "feather", "sand", "ice", "glass", "coin", "shelf", "pipe", "net",
    "chain", "bowl", "bucket", "fence", "needle", "log", "gear",
    "lens", "piston", "valve", "wedge", "pulley", "anvil", "bellows", "trough"
]

PROCESS_CONCEPTS = [
    "fall", "bounce", "collision", "spill", "slide",
    "dissolve", "shatter", "ignite", "rust",
    "vibration", "compression", "flow"
]

MATERIAL_CONCEPTS = [
    "water", "fire", "shadow", "cloud", "wave", "bark", "field"
]

MODALITY_PAIRS = [
    ("lm",   "vis",  "LM↔VIS"),
    ("lm",   "mae",  "LM↔MAE"),
    ("lm",   "clip", "LM↔CLIP"),
    ("vis",  "mae",  "VIS↔MAE"),
    ("vis",  "clip", "VIS↔CLIP"),
    ("clip", "mae",  "CLIP↔MAE"),
]

MODALITY_FILES = {
    "lm":   "lm_hiddens_phrase.npy",
    "vis":  "vjepa2_hiddens_phrase.npy",
    "mae":  "mae_hiddens_phrase.npy",
    "clip": "clip_hiddens_phrase.npy",
}

# ── VQ CODEBOOK ──────────────────────────────────────────────────────────────

class VQCodebook(nn.Module):
    def __init__(self, dim_a, embed_dim, n_codes, dim_b=None):
        super().__init__()
        self.proj_a = nn.Linear(dim_a, embed_dim)
        self.proj_b = nn.Linear(dim_b if dim_b is not None else dim_a, embed_dim)
        self.codebook = nn.Embedding(n_codes, embed_dim)
        nn.init.orthogonal_(self.codebook.weight)

    def quantize(self, z):
        d = torch.cdist(z, self.codebook.weight)
        idx = d.argmin(dim=-1)
        return idx, self.codebook(idx)

    def forward(self, a, b):
        za = F.normalize(self.proj_a(a), dim=-1)
        zb = F.normalize(self.proj_b(b), dim=-1)
        idx_a, qa = self.quantize(za)
        idx_b, qb = self.quantize(zb)

        # Reconstruction loss
        rec = (F.mse_loss(qa, za.detach()) + F.mse_loss(qb, zb.detach()) +
               0.25 * (F.mse_loss(za, qa.detach()) + F.mse_loss(zb, qb.detach())))

        # Cross-modal contrastive
        sim = torch.mm(za, zb.T) / 0.07
        labels = torch.arange(len(za), device=a.device)
        cm = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

        # Diversity
        avg_a = F.softmax(-torch.cdist(za, self.codebook.weight) * 5, dim=-1).mean(0)
        avg_b = F.softmax(-torch.cdist(zb, self.codebook.weight) * 5, dim=-1).mean(0)
        div = (-(avg_a * (avg_a + 1e-8).log()).sum() - (avg_b * (avg_b + 1e-8).log()).sum()) / 2

        loss = rec + LAMBDA_CM * cm - LAMBDA_DIV * div
        agreement = (idx_a == idx_b).float().mean().item()
        active = len(idx_a.unique())
        return loss, agreement, active


def run_codebook(a_all, b_all, concept_indices, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(concept_indices)
    idx = np.array(concept_indices)
    split = int(0.8 * n)
    perm = np.random.permutation(n)
    train_idx = idx[perm[:split]]
    test_idx  = idx[perm[split:]]

    dim_a = a_all.shape[-1]
    dim_b = b_all.shape[-1]
    # Handle phrase-level: average over phrases per concept
    a_train = torch.tensor(a_all[train_idx], dtype=torch.float32)
    b_train = torch.tensor(b_all[train_idx], dtype=torch.float32)
    a_test  = torch.tensor(a_all[test_idx],  dtype=torch.float32)
    b_test  = torch.tensor(b_all[test_idx],  dtype=torch.float32)

    model = VQCodebook(dim_a, EMBED_DIM, N_CODES, dim_b=dim_b)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(300):
        model.train()
        opt.zero_grad()
        loss, _, _ = model(a_train, b_train)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        _, train_agr, active = model(a_train, b_train)
        _, test_agr, _       = model(a_test,  b_test)

    return train_agr, test_agr, active


# ── LOAD EMBEDDINGS ──────────────────────────────────────────────────────────

def load_embeddings():
    embs = {}
    for mod, fname in MODALITY_FILES.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping {mod}")
            continue
        arr = np.load(path)
        embs[mod] = arr
        print(f"  Loaded {mod}: {arr.shape}")
    return embs


def load_event_index():
    with open(os.path.join(OUTPUT_DIR, "event_index.json")) as f:
        data = json.load(f)
        return data["events"]


def get_concept_mean_embeddings(embs, event_index):
    """Average phrase embeddings per concept."""
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    result = {}
    for mod, arr in embs.items():
        concept_embs = []
        for concept in concepts:
            idxs = [i for i, e in enumerate(event_index) if e["concept"] == concept]
            concept_embs.append(arr[idxs].mean(axis=0))
        result[mod] = np.array(concept_embs)
    return concepts, result


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PROCESS CONCEPT PROBE")
    print("Tests: does V-JEPA2 diverge more on process vs object concepts?")
    print("=" * 60)

    embs_raw = load_embeddings()
    if len(embs_raw) < 2:
        print("ERROR: Need at least 2 modalities loaded.")
        sys.exit(1)

    event_index = load_event_index()
    all_concepts, embs = get_concept_mean_embeddings(embs_raw, event_index)

    print(f"\nTotal concepts in event_index: {len(all_concepts)}")

    # Identify which concepts are present and categorize
    present_objects   = [c for c in OBJECT_CONCEPTS   if c in all_concepts]
    present_processes = [c for c in PROCESS_CONCEPTS  if c in all_concepts]
    present_materials = [c for c in MATERIAL_CONCEPTS if c in all_concepts]

    print(f"Object concepts present:   {len(present_objects)}")
    print(f"Process concepts present:  {len(present_processes)}")
    print(f"Material concepts present: {len(present_materials)}")

    if len(present_processes) == 0:
        print("\nERROR: No process concepts found in event_index.")
        print("Run extract_process_concepts.py first to add them.")
        sys.exit(1)

    concept_groups = {
        "objects":   present_objects,
        "processes": present_processes,
        "materials": present_materials,
        "all":       all_concepts,
    }

    results = {}

    for group_name, group_concepts in concept_groups.items():
        if len(group_concepts) < 5:
            print(f"\nSkipping '{group_name}' — only {len(group_concepts)} concepts")
            continue

        concept_indices = [all_concepts.index(c) for c in group_concepts]
        print(f"\n{'─'*50}")
        print(f"Group: {group_name.upper()} ({len(group_concepts)} concepts)")
        print(f"{'─'*50}")

        group_results = {}
        for mod_a, mod_b, label in MODALITY_PAIRS:
            if mod_a not in embs or mod_b not in embs:
                continue

            train_agrs, test_agrs, actives = [], [], []
            for seed in range(N_SEEDS):
                for split in range(N_SPLITS):
                    s = seed * 100 + split
                    tr, te, ac = run_codebook(
                        embs[mod_a], embs[mod_b], concept_indices, seed=s
                    )
                    train_agrs.append(tr)
                    test_agrs.append(te)
                    actives.append(ac)

            mean_train = np.mean(train_agrs) * 100
            mean_test  = np.mean(test_agrs)  * 100
            std_train  = np.std(train_agrs)  * 100
            mean_active = np.mean(actives)

            print(f"  {label:<12}  train={mean_train:.1f}%±{std_train:.1f}  "
                  f"test={mean_test:.1f}%  codes={mean_active:.1f}")

            group_results[label] = {
                "train_mean": mean_train, "train_std": std_train,
                "test_mean": mean_test,   "active_codes": mean_active,
            }

        results[group_name] = group_results

    # ── HYPOTHESIS TEST ───────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HYPOTHESIS TEST: Objects vs. Processes")
    print(f"{'='*60}")
    print("Prediction: LM↔VIS lower for processes; LM↔MAE similar across groups")
    print()

    if "objects" in results and "processes" in results:
        for label in ["LM↔VIS", "LM↔MAE", "VIS↔MAE"]:
            if label in results["objects"] and label in results["processes"]:
                obj = results["objects"][label]["test_mean"]
                proc = results["processes"][label]["test_mean"]
                diff = proc - obj
                direction = "✓ LOWER for processes" if diff < -2 else (
                            "✗ HIGHER for processes" if diff > 2 else "≈ similar")
                print(f"  {label:<12}  objects={obj:.1f}%  processes={proc:.1f}%  "
                      f"Δ={diff:+.1f}%  {direction}")

        lm_vis_obj  = results["objects"]["LM↔VIS"]["test_mean"]
        lm_vis_proc = results["processes"]["LM↔VIS"]["test_mean"]
        lm_mae_obj  = results["objects"]["LM↔MAE"]["test_mean"]
        lm_mae_proc = results["processes"]["LM↔MAE"]["test_mean"]

        hypothesis_supported = (lm_vis_proc < lm_vis_obj) and (abs(lm_mae_proc - lm_mae_obj) < 5)
        print(f"\n  Overall: hypothesis {'SUPPORTED ✓' if hypothesis_supported else 'NOT SUPPORTED ✗'}")

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
