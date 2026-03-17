"""
diagnose_gemma2.py
==================

Diagnoses whether gemma2_hiddens_phrase.npy contains meaningful
per-concept variance, or whether the embeddings have collapsed.

Checks:
  1. Basic stats (mean, std, per-dim variance)
  2. Concept discriminability: within-concept vs between-concept cosine similarity
  3. Compare against Mistral embeddings (known-good) on same concepts
  4. Layer sweep: re-extract from layers 10, 16, 21, 28, 35 and check variance
  5. Prints recommended layer if current layer looks collapsed

Run from repo root:
  PYTHONPATH=. .venv/Scripts/python.exe diagnose_gemma2.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

DATA_DIR = "lm_output/phrase_level"
GEMMA_FILE  = os.path.join(DATA_DIR, "gemma2_hiddens_phrase.npy")
MISTRAL_FILE = os.path.join(DATA_DIR, "lm_hiddens_phrase.npy")  # known-good

MODEL_ID = "google/gemma-2-9b-it"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

# Layers to sweep for variance check
LAYERS_TO_CHECK = [8, 16, 21, 28, 35, 41]


def concept_stats(arr, event_index):
    """Within-concept vs between-concept cosine similarity."""
    concepts = list(dict.fromkeys(e["concept"] for e in event_index))
    concept_means = []
    for c in concepts:
        idxs = [i for i,e in enumerate(event_index) if e["concept"]==c]
        concept_means.append(arr[idxs].mean(axis=0))
    cm = np.array(concept_means)
    # Normalise
    norms = np.linalg.norm(cm, axis=1, keepdims=True)
    cm_n = cm / (norms + 1e-8)
    # Pairwise cosine sim
    sim = cm_n @ cm_n.T
    np.fill_diagonal(sim, np.nan)
    between_mean = np.nanmean(sim)
    between_std  = np.nanstd(sim)
    return between_mean, between_std, cm


def extract_layer(model, tokenizer, texts, layer_idx, batch=8):
    hiddens = []
    for i in range(0, len(texts), batch):
        batch_t = texts[i:i+batch]
        enc = tokenizer(batch_t, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states[layer_idx + 1]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)
        hiddens.append(pooled.cpu().float().numpy())
    return np.concatenate(hiddens, axis=0)


def main():
    print("="*72)
    print("GEMMA-2-9B EMBEDDING DIAGNOSTICS")
    print("="*72)

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    phrases = [e["phrase"] for e in event_index]

    # ── 1. Check saved Gemma embeddings ─────────────────────────────────────
    if not os.path.exists(GEMMA_FILE):
        print(f"\nERROR: {GEMMA_FILE} not found. Run extract_gemma2.py first.")
        return

    gem = np.load(GEMMA_FILE)
    print(f"\n1. SAVED EMBEDDING STATS  ({GEMMA_FILE})")
    print(f"   Shape:  {gem.shape}")
    print(f"   Global mean:  {gem.mean():.4f}")
    print(f"   Global std:   {gem.std():.4f}")
    print(f"   Per-dim std (mean across dims):  {gem.std(axis=0).mean():.4f}")
    print(f"   Per-phrase L2 norm (mean):        {np.linalg.norm(gem, axis=1).mean():.4f}")

    b_mean, b_std, gem_concepts = concept_stats(gem, event_index)
    print(f"\n   Between-concept cosine similarity (after mean-pooling per concept):")
    print(f"     mean = {b_mean:.4f}  std = {b_std:.4f}")
    print(f"   → If mean > 0.99: embeddings are near-constant (collapsed)")
    print(f"   → If mean 0.90–0.99: very low variance (likely extraction issue)")
    print(f"   → If mean 0.50–0.85: healthy discrimination")

    # ── 2. Compare against Mistral ───────────────────────────────────────────
    if os.path.exists(MISTRAL_FILE):
        mis = np.load(MISTRAL_FILE)
        mb_mean, mb_std, _ = concept_stats(mis, event_index)
        print(f"\n2. COMPARISON — Mistral-7B (known-good):")
        print(f"   Between-concept cosine sim: mean={mb_mean:.4f}  std={mb_std:.4f}")
        print(f"   Gemma vs Mistral discriminability ratio: {b_std/mb_std:.2f}x")
        if b_std < mb_std * 0.3:
            print(f"   ⚠️  Gemma variance is <30% of Mistral → likely collapsed")
        elif b_std < mb_std * 0.6:
            print(f"   ⚠️  Gemma variance is <60% of Mistral → possible layer issue")
        else:
            print(f"   ✓  Gemma variance comparable to Mistral")

    # ── 3. Layer sweep ───────────────────────────────────────────────────────
    print(f"\n3. LAYER SWEEP — checking which layer has best concept discrimination")
    print(f"   Loading {MODEL_ID}...")

    import os as _os
    hf_token = _os.environ.get("HF_TOKEN")
    kwargs = dict(torch_dtype=torch.float16, device_map="auto")
    if hf_token: kwargs["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **({"token": hf_token} if hf_token else {}))
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()

    n_layers = model.config.num_hidden_layers
    print(f"   Model has {n_layers} transformer layers")
    print(f"   Checking layers: {LAYERS_TO_CHECK}")
    print(f"   (Using 32 phrases for speed)\n")

    sample_phrases = phrases[:32]
    best_layer, best_std = -1, -1.0

    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw2 = json.load(f)
    ei_sample = (raw2 if isinstance(raw2, list) else raw2["events"])[:32]

    print(f"   {'Layer':>6}  {'Mean cosine':>12}  {'Std cosine':>11}  {'Discriminability'}")
    print(f"   {'-'*55}")

    layer_results = {}
    for layer_idx in LAYERS_TO_CHECK:
        if layer_idx >= n_layers:
            print(f"   {layer_idx:>6}  — skipped (model only has {n_layers} layers)")
            continue
        h = extract_layer(model, tokenizer, sample_phrases, layer_idx)
        concepts_sample = list(dict.fromkeys(e["concept"] for e in ei_sample))
        if len(concepts_sample) < 2:
            continue
        # quick pairwise on raw embeddings (no concept-mean needed for layer sweep)
        h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
        sim = h_n @ h_n.T
        np.fill_diagonal(sim, np.nan)
        m, s = np.nanmean(sim), np.nanstd(sim)
        note = ""
        if s > best_std: best_std, best_layer = s, layer_idx; note = "← best so far"
        print(f"   {layer_idx:>6}  {m:>12.4f}  {s:>11.4f}  {note}")
        layer_results[layer_idx] = {"mean": float(m), "std": float(s)}

    print(f"\n   Recommended layer: {best_layer}  (highest between-phrase cosine std)")
    print(f"   Current layer used: 21")
    if best_layer != 21:
        print(f"   ⚠️  Mismatch — re-run extract_gemma2.py with --layer {best_layer}")
        print(f"      then re-run gemma2_arch_control.py")
    else:
        print(f"   ✓  Layer 21 is the best layer — extraction looks correct")
        print(f"      The high agreement values may reflect genuine Gemma behavior")
        print(f"      (different representational scale, not necessarily collapsed)")

    print(f"\n{'='*72}")
    print("DIAGNOSIS SUMMARY")
    print("="*72)
    print(f"  If between-concept cosine mean > 0.95: extraction collapsed, retry")
    print(f"  If best_layer != 21: retry extraction with recommended layer")
    print(f"  If Gemma genuinely shows reversed ordering at best layer: report as")
    print(f"  'exception requiring further investigation' rather than replication")


if __name__ == "__main__":
    main()
