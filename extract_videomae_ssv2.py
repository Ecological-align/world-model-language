"""
extract_videomae_ssv2.py
========================

Extracts VideoMAE-SSv2 embeddings for all phrase-level events.

WHY THIS MODEL MATTERS — controls for temporal *content* in training data
=========================================================================
Current evidence:
  VideoMAE-K400 clusters with V-JEPA2 (not MAE) → video data drives gap

But Kinetics-400 is object-centric (short action clips, low relational complexity).
Something-Something V2 (SSv2) is explicitly about temporal/relational dynamics:
  "putting X on top of Y", "pushing X until it falls off", "pretending to pour X into Y"

Hypothesis: if temporal *content* within video training matters,
  VideoMAE-SSv2 should show LOWER LM alignment than VideoMAE-K400

Comparison:
  Model             | Data      | Objective       | Temporal content |
  ------------------|-----------|-----------------|-----------------|
  MAE               | Images    | Reconstruction  | None            |
  VideoMAE-K400     | Video     | Reconstruction  | Low (actions)   |
  VideoMAE-SSv2     | Video     | Reconstruction  | HIGH (dynamics) |  <- NEW
  V-JEPA 2          | Video     | Temporal pred   | High            |

If VideoMAE-SSv2 aligns lower with LMs than VideoMAE-K400:
  → Temporal *content* within video matters (not just video vs image)
  → Brings story back toward "temporal dynamics" as the true driver

If VideoMAE-SSv2 ≈ VideoMAE-K400:
  → Video data generically (regardless of temporal richness) drives the gap

Model: MCG-NJU/videomae-base-ssv2 (ViT-Base, SSv2 pre-trained)
  Note: base model (768-dim) vs large (1024-dim) for K400 variant.
  Dim difference is fine — VQCodebook projects to shared embed_dim anyway.

Output: lm_output/phrase_level/videomae_ssv2_hiddens_phrase.npy  [N, 768]

Usage:
  python extract_videomae_ssv2.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import VideoMAEImageProcessor, VideoMAEModel

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID   = "MCG-NJU/videomae-base-ssv2"   # ViT-Base, 768-dim, SSv2 pre-trained
OUTPUT_DIR = "lm_output/phrase_level"
OUT_FILE   = os.path.join(OUTPUT_DIR, "videomae_ssv2_hiddens_phrase.npy")
IMAGE_DIR  = "lm_output/phrase_images"
EVENT_IDX  = os.path.join(OUTPUT_DIR, "event_index.json")
N_FRAMES   = 16
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

def load_videomae_ssv2():
    print(f"Loading {MODEL_ID}...")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model     = VideoMAEModel.from_pretrained(MODEL_ID).to(DEVICE)
    model.eval()
    # Cast to float16 to avoid dtype mismatch on CUDA
    model     = model.half()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    embed_dim = model.config.hidden_size
    print(f"  Loaded on {DEVICE}  ({n_params:.0f}M params, embed_dim={embed_dim})")
    return processor, model, embed_dim


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(image, processor, model):
    """
    Same zero-velocity condition: static image repeated N_FRAMES times.
    Returns mean-pooled hidden state as float32 numpy array.
    """
    frames = [image] * N_FRAMES
    inputs = processor(frames, return_tensors="pt")
    # Cast pixel_values to match model dtype (float16)
    pixel_values = inputs["pixel_values"].half().to(DEVICE)
    outputs      = model(pixel_values=pixel_values)
    hidden       = outputs.last_hidden_state[0]   # [seq_len, embed_dim]
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

    processor, model, embed_dim = load_videomae_ssv2()

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
    print("Key question: does VideoMAE-SSv2 align LOWER with LMs than VideoMAE-K400?")
    print("  Yes → temporal content within video matters (temporal dynamics story)")
    print("  No  → video data generically drives the gap (regardless of content)")


if __name__ == "__main__":
    main()
