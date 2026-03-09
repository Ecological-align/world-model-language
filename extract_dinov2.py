"""
extract_dinov2.py
=================

Extracts DINOv2-Large embeddings for all phrase-level events.

WHY THIS MODEL MATTERS — controls for reconstruction objective in image models
==============================================================================
Current evidence:
  MAE (image, reconstruction)  → HIGH LM alignment (~18-32%)
  VideoMAE (video, reconstruction) → LOW LM alignment (~12-15%)

Question: is MAE's high alignment due to (a) image training, or (b) reconstruction?

DINOv2 is image-trained but uses self-distillation (not reconstruction):
  - Contrastive/distillation training on Images
  - No masking or pixel reconstruction
  - State-of-the-art image features

Prediction:
  If image training drives high LM alignment:
    DINOv2 ≈ MAE (both high)  →  IMAGE DATA is the key variable
  
  If reconstruction objective drives high LM alignment:
    DINOv2 ≈ VideoMAE/V-JEPA2 (low)  →  RECONSTRUCTION is the key variable

Full 2×2 design this creates:
  Model           | Data   | Objective          |
  ----------------|--------|--------------------|
  MAE             | Image  | Reconstruction     |
  DINOv2          | Image  | Distillation (NEW) |
  VideoMAE-K400   | Video  | Reconstruction     |
  VideoMAE-SSv2   | Video  | Reconstruction     |
  V-JEPA2         | Video  | Temporal predict   |

Model: facebook/dinov2-large (ViT-L, 1024-dim)
  Matches MAE and VideoMAE-K400 architecture size for fair comparison.

Output: lm_output/phrase_level/dinov2_hiddens_phrase.npy  [N, 1024]

Usage:
  python extract_dinov2.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

# ── Config ─────────────────────────────────────────────────────────────────────

MODEL_ID   = "facebook/dinov2-large"   # ViT-L, 1024-dim, image distillation
OUTPUT_DIR = "lm_output/phrase_level"
OUT_FILE   = os.path.join(OUTPUT_DIR, "dinov2_hiddens_phrase.npy")
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

def load_dinov2():
    print(f"Loading {MODEL_ID}...")
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model     = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to(DEVICE)
    model.eval()
    n_params  = sum(p.numel() for p in model.parameters()) / 1e6
    embed_dim = model.config.hidden_size
    print(f"  Loaded on {DEVICE}  ({n_params:.0f}M params, embed_dim={embed_dim})")
    return processor, model, embed_dim


# ── Extraction ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(image, processor, model):
    """
    Extract CLS token from DINOv2.
    DINOv2 is an image model — takes a single image.
    Uses last_hidden_state mean (same approach as MAE extraction).
    Returns float32 numpy array of shape [embed_dim].
    """
    inputs = processor(images=image, return_tensors="pt")
    # Cast to float16 to match model dtype
    pixel_values = inputs["pixel_values"].half().to(DEVICE)
    outputs      = model(pixel_values=pixel_values)
    # last_hidden_state: [1, seq_len, embed_dim]; seq_len includes CLS token
    hidden = outputs.last_hidden_state[0]   # [seq_len, embed_dim]
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

    processor, model, embed_dim = load_dinov2()

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
    print("Key question: does DINOv2 (image, distillation) align HIGH like MAE?")
    print("  Yes → IMAGE DATA is the key variable (not reconstruction objective)")
    print("  No  → RECONSTRUCTION OBJECTIVE drives MAE's high LM alignment")


if __name__ == "__main__":
    main()
