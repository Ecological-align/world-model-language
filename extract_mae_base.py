"""
extract_mae_base.py
===================

Extracts MAE-Base embeddings for the architecture-controlled comparison.

WHY THIS MATTERS — ruling out the architecture confound
=======================================================
Current comparison:
  MAE ViT-Large     (1024-dim, 307M params)  →  ~24% LM alignment
  VideoMAE-SSv2-Base (768-dim, 86M params)   →  ~13% LM alignment

These differ in both training data AND architecture depth/capacity.
A reviewer could argue: MAE-Large is just a better model, and better
models happen to align more with language.

This comparison controls for architecture:
  MAE-Base          (768-dim, 86M params)    →  ???
  VideoMAE-Base/K400 (768-dim, 86M params)   →  already extracted as videomae_hiddens_phrase.npy
  VideoMAE-Base/SSv2 (768-dim, 86M params)   →  already extracted as videomae_ssv2_hiddens_phrase.npy

If MAE-Base also scores high (~20%+) while both VideoMAE-Base models
score low (~13%), the architecture confound is eliminated. The gap is
about training data + objective, not model capacity.

Model: facebook/vit-mae-base (ViT-Base, 768-dim, ~86M params)

Output: lm_output/phrase_level/mae_base_hiddens_phrase.npy  [N, 768]

Usage:
  python extract_mae_base.py

After running, add 'mae_base' to the architecture_control_probe.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import ViTMAEModel, AutoImageProcessor

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID   = "facebook/vit-mae-base"   # ViT-Base, 768-dim — same arch as VideoMAE-Base
OUTPUT_DIR = "lm_output/phrase_level"
OUT_FILE   = os.path.join(OUTPUT_DIR, "mae_base_hiddens_phrase.npy")
IMAGE_DIR  = "lm_output/phrase_images"
EVENT_IDX  = os.path.join(OUTPUT_DIR, "event_index.json")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Load event index ───────────────────────────────────────────────────────────

def load_events():
    with open(EVENT_IDX) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data["events"]


# ── Load image ─────────────────────────────────────────────────────────────────

def load_image(event):
    event_id   = event.get("event_id", f"{event['concept']}_{event['phrase'][:20]}")
    candidates = [
        os.path.join(IMAGE_DIR, f"{event_id}.jpg"),
        os.path.join(IMAGE_DIR, f"{event_id}.png"),
        os.path.join(IMAGE_DIR,
                     f"{event['concept']}_{event['phrase'][:30].replace(' ', '_')}.jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    concept_dir = os.path.join(IMAGE_DIR, event.get("concept", ""))
    if os.path.isdir(concept_dir):
        for fname in os.listdir(concept_dir):
            if fname.endswith((".jpg", ".png")):
                try:
                    return Image.open(os.path.join(concept_dir, fname)).convert("RGB")
                except Exception:
                    pass
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ── Model loading ──────────────────────────────────────────────────────────────

def load_mae_base():
    print(f"Loading {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = ViTMAEModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.config.mask_ratio = 0.0   # No masking — pure feature extraction
    model     = model.half()         # Match VideoMAE-Base dtype
    model.eval()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    embed_dim = model.config.hidden_size
    print(f"  Loaded on {DEVICE}  ({n_params:.0f}M params, embed_dim={embed_dim})")
    print(f"  mask_ratio=0.0 (feature extraction mode, same as MAE-Large)")
    return processor, model, embed_dim


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(image, processor, model):
    """
    Mean-pool last hidden state.
    Identical approach to MAE-Large extraction in extract_phrase_level.py.
    """
    inputs       = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].half().to(DEVICE)
    outputs      = model(pixel_values=pixel_values)
    hidden       = outputs.last_hidden_state[0]   # [seq_len, 768]
    return hidden.mean(0).float().cpu().numpy()


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    if os.path.exists(OUT_FILE):
        arr = np.load(OUT_FILE)
        print(f"Output already exists: {OUT_FILE}  shape={arr.shape}")
        print("Delete the file to re-extract.")
        return

    events = load_events()
    N      = len(events)
    print(f"Events: {N}")
    print(f"Output: {OUT_FILE}")
    print()
    print("Architecture comparison this enables:")
    print("  MAE-Base (768-dim)         vs  VideoMAE-Base/K400 (768-dim)")
    print("  MAE-Base (768-dim)         vs  VideoMAE-Base/SSv2 (768-dim)")
    print("  Same architecture — only training data + objective differs")
    print()

    processor, model, embed_dim = load_mae_base()

    hiddens = np.zeros((N, embed_dim), dtype=np.float32)
    failed  = 0

    for i, event in enumerate(events):
        image = load_image(event)
        try:
            hiddens[i] = extract(image, processor, model)
        except Exception as e:
            print(f"  ERROR event {i} ({event.get('concept', '?')}): {e}")
            failed += 1

        if (i + 1) % 25 == 0 or i == N - 1:
            print(f"  [{i+1:3d}/{N}]  concept={event.get('concept', '?'):<14}  "
                  f"failed={failed}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUT_FILE, hiddens)
    print(f"\nSaved: {OUT_FILE}  shape={hiddens.shape}")
    print(f"Failed events: {failed}/{N}")
    print()
    print("Key question: does MAE-Base align HIGH like MAE-Large?")
    print("  Yes → architecture capacity is not driving the result")
    print("  No  → model size / depth matters, not just objective + data")
    print()
    print("Next: run architecture_control_probe.py")


if __name__ == "__main__":
    main()
