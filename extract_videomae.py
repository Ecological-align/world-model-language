"""
extract_videomae.py
===================

Extracts VideoMAE embeddings for all phrase-level events.

WHY THIS MODEL MATTERS — the critical control
==============================================
Current probe has:
  MAE       = static image reconstruction (no temporal training)
  V-JEPA 2  = video temporal prediction (temporal training)

Missing variable: video data WITHOUT temporal prediction.

VideoMAE fills this gap:
  VideoMAE  = video reconstruction (video data, tube masking, NO temporal prediction)

This creates a clean 3-way isolation:

  Model       | Trained on | Objective        | Temporal? |
  ------------|------------|------------------|-----------|
  MAE         | Images     | Reconstruction   | No        |
  VideoMAE    | Video      | Reconstruction   | No        |  <- NEW
  V-JEPA 2    | Video      | Temporal predict | Yes       |

Hypothesis (if temporal prediction is the driver):
  {LM, MAE, VideoMAE} cluster together  →  temporal prediction, not video data, drives the gap

Hypothesis (if video data is the driver):
  {LM, MAE} cluster, {VideoMAE, V-JEPA 2} cluster  →  video training itself drives the gap

Model used: MCG-NJU/videomae-large (ViT-L, pre-trained on Kinetics-400)
  - Same ViT-L architecture as MAE and V-JEPA 2
  - 1024-dim CLS token hidden state
  - Natively supported in HuggingFace transformers

Input: same static image repeated 16 frames (zero-velocity condition)
  This is intentionally the same condition as V-JEPA 2 for fair comparison.
  If VideoMAE clusters with MAE under this condition, the static proxy argument
  is less of a concern — because VideoMAE also sees no motion, yet clusters differently.

Output: lm_output/phrase_level/videomae_hiddens_phrase.npy  [N, 1024]

Usage:
  python extract_videomae.py

After running, add 'videomae' to the quadmodal probe:
  python alt_lm_probe.py      (already handles arbitrary visual models)
  python process_concept_probe.py  (also handles arbitrary visual models)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from transformers import VideoMAEImageProcessor, VideoMAEModel

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID   = "MCG-NJU/videomae-large"   # ViT-L, 1024-dim, ~307M params
OUTPUT_DIR = "lm_output/phrase_level"
OUT_FILE   = os.path.join(OUTPUT_DIR, "videomae_hiddens_phrase.npy")
IMAGE_DIR  = "lm_output/phrase_images"
EVENT_IDX  = os.path.join(OUTPUT_DIR, "event_index.json")
N_FRAMES   = 16    # VideoMAE default; same as V-JEPA 2 setup
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load event index ───────────────────────────────────────────────────────────

def load_events():
    with open(EVENT_IDX) as f:
        data = json.load(f)
    # Handle both list format and dict-with-events format
    if isinstance(data, list):
        return data
    return data["events"]

# ── Load image for event ───────────────────────────────────────────────────────

def load_image(event):
    """
    Load cached phrase image, falling back to a blank image if not found.
    Matches the approach used in extract_phrase_level.py.
    """
    event_id = event.get("event_id", f"{event['concept']}_{event['phrase'][:20]}")
    # Try common filename patterns used by the downloader
    candidates = [
        os.path.join(IMAGE_DIR, f"{event_id}.jpg"),
        os.path.join(IMAGE_DIR, f"{event_id}.png"),
        os.path.join(IMAGE_DIR, f"{event['concept']}_{event['phrase'][:30].replace(' ', '_')}.jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return Image.open(path).convert("RGB")
            except Exception:
                pass
    # Also search by concept subdirectory if used
    concept_dir = os.path.join(IMAGE_DIR, event.get("concept", ""))
    if os.path.isdir(concept_dir):
        for fname in os.listdir(concept_dir):
            if fname.endswith((".jpg", ".png")):
                try:
                    return Image.open(os.path.join(concept_dir, fname)).convert("RGB")
                except Exception:
                    pass
    # Fallback: blank grey image
    return Image.new("RGB", (224, 224), color=(128, 128, 128))


# ── Model loading ──────────────────────────────────────────────────────────────

def load_videomae():
    print(f"Loading {MODEL_ID}...")
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model     = VideoMAEModel.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded on {DEVICE}  ({n_params:.0f}M params)")
    return processor, model


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_videomae(image, processor, model):
    """
    Extract CLS token hidden state from VideoMAE.

    Input: single PIL image, repeated N_FRAMES times (zero-velocity condition).
    This matches the V-JEPA 2 extraction approach for fair comparison.

    Returns: 1024-dim float32 numpy array
    """
    # Repeat same frame N_FRAMES times — same "zero-velocity" condition as V-JEPA 2
    frames = [image] * N_FRAMES

    # VideoMAEImageProcessor expects a list of PIL images (one per frame)
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device=DEVICE, dtype=torch.float16) for k, v in inputs.items()}

    # pixel_values shape: [1, N_FRAMES, C, H, W]
    # VideoMAEModel forward pass (no masking needed for feature extraction)
    outputs = model(pixel_values=inputs["pixel_values"])

    # CLS token from last hidden state: [1, seq_len, 1024]
    # Use mean over all patch tokens + CLS (same as MAE extraction)
    hidden = outputs.last_hidden_state[0]  # [seq_len, 1024]
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
    print(f"Zero-velocity condition: {N_FRAMES} repeated frames (same as V-JEPA 2)")
    print()

    processor, model = load_videomae()

    hiddens = np.zeros((N, 1024), dtype=np.float32)
    failed  = 0

    for i, event in enumerate(events):
        image = load_image(event)
        try:
            hiddens[i] = extract_videomae(image, processor, model)
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
    print("Next steps:")
    print("  1. Run alt_lm_probe.py   — add 'videomae' to VISUAL_MODELS dict")
    print("  2. Run process_concept_probe.py — test VIS/MAE/VideoMAE on process concepts")
    print()
    print("Key comparison to watch:")
    print("  If VideoMAE clusters with MAE → temporal PREDICTION drives the gap")
    print("  If VideoMAE clusters with V-JEPA 2 → video DATA drives the gap")


if __name__ == "__main__":
    main()
