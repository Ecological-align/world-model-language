"""
gemma2_layer_sweep_probe.py
============================

Quick experiment: extract Gemma-2-9B at layers 28 and 35 (identified as
candidates from the diagnostic sweep), then run codebook probes against
MAE, CLIP, V-JEPA2, and VideoMAE-K400 at each layer.

Goal: find the layer where VideoMAE ≈ V-JEPA2 (as seen in all other LLMs),
confirming that the V-JEPA2/VideoMAE divergence at layer 41 is an extraction
artifact rather than a genuine Gemma finding.

If VideoMAE ≈ V-JEPA2 restores at an intermediate layer, we can:
  (a) use that layer for Gemma's main results table entry
  (b) note that layer 41 is appropriate for architecture control
      (where absolute scale doesn't matter, only MAE-Base vs VideoMAE gap)

Saves per-layer embeddings as:
  lm_output/phrase_level/gemma2_L{N}_hiddens_phrase.npy

Prints codebook agreement table per layer.

Run from repo root:
  PYTHONPATH=. .venv/Scripts/python.exe gemma2_layer_sweep_probe.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DATA_DIR = "lm_output/phrase_level"
MODEL_ID = "google/gemma-2-9b-it"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Layers to sweep — diagnostic showed 28 and 35 as candidates
# Also include 41 (already run) for direct comparison in same table
SWEEP_LAYERS = [28, 35, 41]

VISUAL_MODELS = {
    "mae":       ("mae_hiddens_phrase.npy",      "MAE-Large"),
    "clip":      ("clip_hiddens_phrase.npy",     "CLIP"),
    "vjepa2":    ("vjepa2_hiddens_phrase.npy",   "V-JEPA2"),
    "vmae_k400": ("videomae_hiddens_phrase.npy", "VideoMAE-K400"),
}

CODEBOOK_DIM = 64
N_CODES      = 16
LAMBDA_CM    = 0.5
LAMBDA_DIV   = 0.1
N_RUNS       = 30          # lighter run — enough to see the pattern
N_BOOTSTRAP  = 1000
BATCH        = 8


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


def extract_layer(model, tokenizer, texts, layer_idx):
    hiddens = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)
        hiddens.append(pooled.cpu().float().numpy())
    return np.concatenate(hiddens, axis=0)


def cosine_discriminability(arr, event_index):
    """Quick check: between-concept cosine similarity std."""
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    means = []
    for c in concepts:
        idxs = [i for i,e in enumerate(event_index) if e["concept"]==c]
        means.append(arr[idxs].mean(axis=0))
    m = np.array(means)
    m_n = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    sim = m_n @ m_n.T
    np.fill_diagonal(sim, np.nan)
    return float(np.nanmean(sim)), float(np.nanstd(sim))


def concept_means(arr, event_index):
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    out = []
    for c in concepts:
        idxs = [i for i,e in enumerate(event_index) if e["concept"]==c]
        out.append(arr[idxs].mean(axis=0))
    return np.array(out)


def run_probe(lm_conc, vis_conc, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    cb  = VQCodebook(lm_conc.shape[1], vis_conc.shape[1], CODEBOOK_DIM, N_CODES)
    opt = torch.optim.Adam(cb.parameters(), lr=1e-3)
    lm_t  = torch.tensor(lm_conc,  dtype=torch.float32)
    vis_t = torch.tensor(vis_conc, dtype=torch.float32)
    for _ in range(300):
        cb.train(); opt.zero_grad()
        loss, _ = cb(lm_t, vis_t)
        loss.backward(); opt.step()
    cb.eval()
    with torch.no_grad():
        _, agr = cb(lm_t, vis_t)
    return agr * 100


def bootstrap_ci(scores):
    rng = np.random.default_rng(42)
    boot = [rng.choice(scores, size=len(scores), replace=True).mean()
            for _ in range(N_BOOTSTRAP)]
    return np.percentile(boot, [2.5, 97.5])


def main():
    print("="*72)
    print("GEMMA-2-9B LAYER SWEEP PROBE")
    print(f"Layers: {SWEEP_LAYERS}  |  Visual models: MAE, CLIP, V-JEPA2, VideoMAE-K400")
    print("Looking for the layer where VideoMAE ≈ V-JEPA2")
    print("="*72)

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    phrases = [e["phrase"] for e in event_index]

    # Load visual model concept means (reused across all layers)
    vis_data = {}
    for key, (fname, label) in VISUAL_MODELS.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            arr = np.load(path)
            vis_data[key] = (concept_means(arr, event_index), label, arr.shape[1])
        else:
            print(f"  [SKIP] {label}: {path} not found")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    hf_token = os.environ.get("HF_TOKEN")
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if hf_token: kwargs["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, **({"token": hf_token} if hf_token else {}))
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"  {n_layers} transformer layers")

    # Sweep layers
    all_results = {}

    for layer_idx in SWEEP_LAYERS:
        if layer_idx >= n_layers:
            print(f"\n[SKIP] Layer {layer_idx} — out of range")
            continue

        # Check if we already have this extraction
        cache_path = os.path.join(DATA_DIR, f"gemma2_L{layer_idx}_hiddens_phrase.npy")
        if os.path.exists(cache_path):
            print(f"\nLayer {layer_idx}: loading cached embeddings from {cache_path}")
            lm_arr = np.load(cache_path)
        else:
            print(f"\nLayer {layer_idx}: extracting {len(phrases)} phrases...")
            lm_arr = extract_layer(model, tokenizer, phrases, layer_idx)
            np.save(cache_path, lm_arr)
            print(f"  Saved to {cache_path}")

        # Discriminability check
        disc_mean, disc_std = cosine_discriminability(lm_arr, event_index)
        print(f"  Discriminability: cosine mean={disc_mean:.4f}  std={disc_std:.4f}")
        if disc_mean > 0.97:
            print(f"  ⚠️  Near-collapsed — skipping probe")
            continue

        lm_conc = concept_means(lm_arr, event_index)
        lm_dim  = lm_arr.shape[1]

        layer_results = {}
        print(f"  Running {N_RUNS} codebook seeds per visual model...")

        for key, (vis_conc, label, vis_dim) in vis_data.items():
            scores = [run_probe(lm_conc, vis_conc, seed) for seed in range(N_RUNS)]
            mean   = np.mean(scores)
            ci     = bootstrap_ci(np.array(scores))
            layer_results[key] = {"mean": float(mean), "ci": [float(ci[0]), float(ci[1])], "label": label}
            print(f"    {label:<20} {mean:.1f}%  [{ci[0]:.1f}, {ci[1]:.1f}]")

        all_results[layer_idx] = layer_results

    # Summary table
    print(f"\n\n{'='*72}")
    print("SUMMARY — Gemma-2-9B agreement by layer")
    print("Key question: at which layer does VideoMAE ≈ V-JEPA2?")
    print("="*72)

    print(f"\n  {'Model':<20}", end="")
    for layer_idx in SWEEP_LAYERS:
        if layer_idx in all_results:
            print(f"  {'L'+str(layer_idx):>12}", end="")
    print()
    print(f"  {'-'*70}")

    for key in ["mae", "clip", "vjepa2", "vmae_k400"]:
        if key not in vis_data: continue
        label = vis_data[key][1]
        print(f"  {label:<20}", end="")
        for layer_idx in SWEEP_LAYERS:
            if layer_idx in all_results and key in all_results[layer_idx]:
                r = all_results[layer_idx][key]
                print(f"  {r['mean']:>5.1f}%      ", end="")
            else:
                print(f"  {'—':>12}", end="")
        print()

    print(f"\n  V-JEPA2 - VideoMAE gap:", end="")
    for layer_idx in SWEEP_LAYERS:
        if layer_idx in all_results:
            r = all_results[layer_idx]
            if "vjepa2" in r and "vmae_k400" in r:
                gap = r["vjepa2"]["mean"] - r["vmae_k400"]["mean"]
                print(f"  {gap:>+6.1f}%      ", end="")
    print()

    print(f"\n  Interpretation:")
    print(f"  ≈ 0%  → VideoMAE ≈ V-JEPA2 (expected pattern from other LLMs)")
    print(f"  > 5%  → V-JEPA2 pulls ahead of VideoMAE (layer-41 artifact)")

    for layer_idx in SWEEP_LAYERS:
        if layer_idx in all_results:
            r = all_results[layer_idx]
            if "vjepa2" in r and "vmae_k400" in r:
                gap = r["vjepa2"]["mean"] - r["vmae_k400"]["mean"]
                pattern = "VideoMAE ≈ V-JEPA2 ✓" if abs(gap) < 3 else f"V-JEPA2 ahead by {gap:+.1f}%"
                print(f"  Layer {layer_idx}: {pattern}")

    # Save results
    import json as json_mod
    out_path = "lm_output/gemma2_layer_sweep_results.json"
    with open(out_path, "w") as f:
        json_mod.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
