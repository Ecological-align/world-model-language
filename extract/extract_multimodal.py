"""
Re-extract embeddings using multi-image (CLIP/MAE) and real video (V-JEPA 2).

This replaces the single Wikipedia thumbnail with:
  - CLIP: mean of N image embeddings (from download_multi_images.py)
  - MAE:  mean of N image embeddings
  - V-JEPA 2: mean of M video clip embeddings (from download_concept_videos.py)
  - Mistral / ST: unchanged (text only)

Run AFTER:
  python download_multi_images.py
  python download_concept_videos.py

Output:
  lm_output/clip_hiddens_multiimg.npy     [49, 768]
  lm_output/mae_hiddens_multiimg.npy      [49, 1024]
  lm_output/vjepa2_hiddens_video.npy      [49, 1024]
  lm_output/concept_index_v2.json         (same order as expanded)

These can be dropped in as replacements for the original *.npy files
in any downstream experiment.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
MULTI_IMG_DIR  = "lm_output/multi_images"
VIDEO_DIR      = "lm_output/concept_videos"
MAX_IMAGES     = 50   # max images to average per concept for CLIP/MAE
MAX_CLIPS      = 5    # max video clips per concept for V-JEPA

with open("lm_output/concept_index.json", "r", encoding="utf-8") as f:
    idx = json.load(f)
ALL_CONCEPTS = idx["all_concepts"]
N = len(ALL_CONCEPTS)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}, Concepts: {N}")

# ── Load models ───────────────────────────────────────────────────────────────

print("\nLoading CLIP...")
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

print("Loading MAE...")
from transformers import ViTMAEModel, ViTMAEConfig, AutoImageProcessor
mae_config = ViTMAEConfig.from_pretrained("facebook/vit-mae-large")
mae_config.mask_ratio = 0.0
mae_proc  = AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
mae_model = ViTMAEModel.from_pretrained(
    "facebook/vit-mae-large", config=mae_config, ignore_mismatched_sizes=True
).to(device)
mae_model.eval()

print("Loading V-JEPA 2...")
from transformers import AutoVideoProcessor, AutoModel
vjepa_proc  = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
vjepa_model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256", dtype=torch.float16
).to(device)
vjepa_model.eval()

print("All models loaded.\n")

# ── Extraction functions ──────────────────────────────────────────────────────

def extract_clip_from_image(img):
    inputs = clip_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(**inputs)
        projected  = clip_model.visual_projection(vision_out.pooler_output)
    return projected[0].float().cpu().numpy()


def extract_mae_from_image(img):
    inputs = mae_proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mae_model(**inputs)
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


def extract_vjepa_from_frames(frames):
    """frames: list of PIL images (video frames)."""
    inputs = vjepa_proc(videos=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = vjepa_model(**inputs, skip_predictor=True)
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


def load_video_frames(video_path, n_frames=8):
    """Load n_frames uniformly sampled from a video file."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        if len(frames) == n_frames:
            return frames
    except ImportError:
        pass

    # Fallback: use decord
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = len(vr)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)
        frames = [Image.fromarray(vr[int(i)].asnumpy()) for i in indices]
        return frames
    except Exception:
        pass

    return None


# ── Multi-image CLIP / MAE extraction ────────────────────────────────────────

print("=" * 65)
print("EXTRACTING CLIP + MAE (multi-image averages)")
print("=" * 65)

clip_vecs = []
mae_vecs  = []

for concept in ALL_CONCEPTS:
    concept_dir = os.path.join(MULTI_IMG_DIR, concept)

    if not os.path.exists(concept_dir):
        print(f"  {concept}: no multi-image dir, falling back to original")
        # Fall back to original single embedding
        orig = np.load("lm_output/clip_hiddens_expanded.npy")
        ci = ALL_CONCEPTS.index(concept)
        clip_vecs.append(orig[ci])
        orig_mae = np.load("lm_output/mae_hiddens_expanded.npy")
        mae_vecs.append(orig_mae[ci])
        continue

    img_files = sorted([f for f in os.listdir(concept_dir)
                        if f.endswith((".jpg", ".png"))])[:MAX_IMAGES]

    if not img_files:
        print(f"  {concept}: no images found, falling back to original")
        orig = np.load("lm_output/clip_hiddens_expanded.npy")
        ci = ALL_CONCEPTS.index(concept)
        clip_vecs.append(orig[ci])
        orig_mae = np.load("lm_output/mae_hiddens_expanded.npy")
        mae_vecs.append(orig_mae[ci])
        continue

    clip_per_img = []
    mae_per_img  = []
    for fname in img_files:
        try:
            img = Image.open(os.path.join(concept_dir, fname)).convert("RGB")
            clip_per_img.append(extract_clip_from_image(img))
            mae_per_img.append(extract_mae_from_image(img))
        except Exception as e:
            print(f"    Error on {fname}: {e}")

    if clip_per_img:
        clip_vecs.append(np.mean(clip_per_img, axis=0))
        mae_vecs.append(np.mean(mae_per_img, axis=0))
        print(f"  {concept}: averaged {len(clip_per_img)} images")
    else:
        print(f"  {concept}: all images failed, falling back")
        orig = np.load("lm_output/clip_hiddens_expanded.npy")
        ci = ALL_CONCEPTS.index(concept)
        clip_vecs.append(orig[ci])
        orig_mae = np.load("lm_output/mae_hiddens_expanded.npy")
        mae_vecs.append(orig_mae[ci])

clip_arr = np.array(clip_vecs)
mae_arr  = np.array(mae_vecs)
np.save("lm_output/clip_hiddens_multiimg.npy", clip_arr)
np.save("lm_output/mae_hiddens_multiimg.npy",  mae_arr)
print(f"\nSaved: clip_hiddens_multiimg.npy {clip_arr.shape}")
print(f"Saved: mae_hiddens_multiimg.npy  {mae_arr.shape}")

# ── Video V-JEPA extraction ───────────────────────────────────────────────────

print("\n" + "=" * 65)
print("EXTRACTING V-JEPA 2 (real video clips)")
print("=" * 65)

# Check if cv2 or decord is available
has_video = False
try:
    import cv2
    has_video = True
    print("Using cv2 for video loading")
except ImportError:
    try:
        from decord import VideoReader
        has_video = True
        print("Using decord for video loading")
    except ImportError:
        print("WARNING: Neither cv2 nor decord available.")
        print("Install with: pip install opencv-python  OR  pip install decord")

vjepa_vecs = []

for concept in ALL_CONCEPTS:
    concept_dir = os.path.join(VIDEO_DIR, concept)
    orig_vjepa  = np.load("lm_output/vjepa2_hiddens_expanded.npy")
    ci = ALL_CONCEPTS.index(concept)

    if not has_video or not os.path.exists(concept_dir):
        print(f"  {concept}: no video dir, using static fallback")
        vjepa_vecs.append(orig_vjepa[ci])
        continue

    video_files = sorted([f for f in os.listdir(concept_dir)
                          if f.endswith((".mp4", ".webm", ".avi"))])[:MAX_CLIPS]

    if not video_files:
        print(f"  {concept}: no video files, using static fallback")
        vjepa_vecs.append(orig_vjepa[ci])
        continue

    clip_embeddings = []
    for fname in video_files:
        video_path = os.path.join(concept_dir, fname)
        frames = load_video_frames(video_path, n_frames=8)
        if frames and len(frames) == 8:
            try:
                vec = extract_vjepa_from_frames(frames)
                clip_embeddings.append(vec)
            except Exception as e:
                print(f"    V-JEPA error on {fname}: {e}")
        else:
            print(f"    Could not load frames from {fname}")

    if clip_embeddings:
        vjepa_vecs.append(np.mean(clip_embeddings, axis=0))
        print(f"  {concept}: averaged {len(clip_embeddings)} video clips")
    else:
        print(f"  {concept}: all videos failed, using static fallback")
        vjepa_vecs.append(orig_vjepa[ci])

vjepa_arr = np.array(vjepa_vecs)
np.save("lm_output/vjepa2_hiddens_video.npy", vjepa_arr)
print(f"\nSaved: vjepa2_hiddens_video.npy {vjepa_arr.shape}")

# ── Quick RSA comparison ──────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("QUICK RSA: multi-image vs single-image")
print("=" * 65)

from scipy.stats import spearmanr

def build_rdm(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    v = vecs / norms
    return 1.0 - v @ v.T

def rsa(a, b):
    n = a.shape[0]
    i = np.triu_indices(n, k=1)
    r, p = spearmanr(a[i], b[i])
    return float(r), float(p)

clip_orig = np.load("lm_output/clip_hiddens_expanded.npy")
vj_orig   = np.load("lm_output/vjepa2_hiddens_expanded.npy")

rdm_clip_orig  = build_rdm(clip_orig)
rdm_clip_multi = build_rdm(clip_arr)
rdm_vj_orig    = build_rdm(vj_orig)
rdm_vj_video   = build_rdm(vjepa_arr)

r_cc, _ = rsa(rdm_clip_orig, rdm_clip_multi)
r_vv, _ = rsa(rdm_vj_orig,   rdm_vj_video)

print(f"CLIP single-image vs multi-image RSA:   r = {r_cc:+.3f}")
print(f"V-JEPA single-frame vs video-clip RSA:  r = {r_vv:+.3f}")
print()
print("High r → geometry unchanged by new extraction method")
print("Low r  → geometry significantly affected (thumbnail bias was real)")

# RSA against each other with new embeddings
r_vj_clip_new, p_vj_clip_new = rsa(rdm_vj_video, rdm_clip_multi)
r_mae_clip_new, p_mae_clip_new = rsa(build_rdm(mae_arr), rdm_clip_multi)
print(f"\nNew V-JEPA (video) vs new CLIP (multi-image): r = {r_vj_clip_new:+.3f} (p={p_vj_clip_new:.4f})")
print(f"New MAE (multi-image) vs new CLIP (multi):    r = {r_mae_clip_new:+.3f} (p={p_mae_clip_new:.4f})")
print(f"Gap: {r_vj_clip_new - r_mae_clip_new:+.3f}")

# Save summary
summary = {
    "clip_single_vs_multi_rsa": r_cc,
    "vjepa_static_vs_video_rsa": r_vv,
    "new_vjepa_vs_new_clip_rsa": {"r": r_vj_clip_new, "p": p_vj_clip_new},
    "new_mae_vs_new_clip_rsa":   {"r": r_mae_clip_new, "p": p_mae_clip_new},
    "new_gap_vjepa_minus_mae":   r_vj_clip_new - r_mae_clip_new,
}
with open("lm_output/multimodal_extraction_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nSaved to lm_output/multimodal_extraction_summary.json")
