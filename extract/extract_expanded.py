"""
Extract representations for new concepts and append to existing .npy files.

Run AFTER preregister_expanded.py, BEFORE any experiments.

This script:
  1. Loads existing embeddings (lm_output/*.npy)
  2. Extracts embeddings for new concepts only
  3. Saves combined arrays back to lm_output/
  4. Saves a concept index file so downstream scripts know which row = which concept

Existing concept order is preserved. New concepts are appended in the order
defined in preregister_expanded.py.

Requirements: same environment as extract_lm_standalone.py + extract_wm_visual.py
  pip install transformers sentence-transformers torch requests pillow
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import time
import numpy as np
import torch
import requests
from PIL import Image
from io import BytesIO

# ── Concept lists ─────────────────────────────────────────────────────────────

ORIGINAL_CONCEPTS = [
    "apple", "chair", "water", "fire", "stone",
    "rope", "door", "container", "shadow", "mirror",
    "knife", "wheel", "hand", "wall", "hole",
    "bridge", "ladder"
]

NEW_CONCEPTS = [
    # High polysemy / high sensorimotor
    "spring", "bark", "wave", "charge", "field",
    "light", "strike", "press", "shoot", "run",
    # Low polysemy / high sensorimotor
    "hammer", "scissors", "bowl", "bucket", "bench",
    "fence", "needle", "drum", "clock", "telescope",
    # Low polysemy / low sensorimotor
    "cloud", "sand", "ice", "feather", "leaf",
    "thread", "glass", "coin", "shelf", "pipe",
    "net", "chain",
]

ALL_CONCEPTS = ORIGINAL_CONCEPTS + NEW_CONCEPTS

print(f"Original concepts: {len(ORIGINAL_CONCEPTS)}")
print(f"New concepts:      {len(NEW_CONCEPTS)}")
print(f"Total:             {len(ALL_CONCEPTS)}")

# ── Image fetching (same approach as original extraction scripts) ─────────────

IMAGE_CACHE_DIR = "lm_output/image_cache"
os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)

def get_wikipedia_image(concept):
    """Fetch Wikipedia lead image for a concept, with local caching."""
    cache_path = os.path.join(IMAGE_CACHE_DIR, f"{concept}.jpg")
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    # Try Wikipedia API
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={concept}&prop=pageimages"
        f"&format=json&pithumbsize=512"
    )
    headers = {"User-Agent": "Mozilla/5.0 (research; contact: research@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        pages = data["query"]["pages"]
        page  = next(iter(pages.values()))
        img_url = page["thumbnail"]["source"]
        time.sleep(2)  # be polite
        img_resp = requests.get(img_url, headers=headers, timeout=15)
        img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        img.save(cache_path)
        return img
    except Exception as e:
        print(f"  Warning: could not fetch image for '{concept}': {e}")
        # Return a grey placeholder
        return Image.new("RGB", (224, 224), color=(128, 128, 128))


def get_wikipedia_text(concept):
    """Fetch Wikipedia introductory paragraph for a concept."""
    url = (
        f"https://en.wikipedia.org/w/api.php"
        f"?action=query&titles={concept}&prop=extracts"
        f"&exintro=true&explaintext=true&format=json"
    )
    headers = {"User-Agent": "Mozilla/5.0 (research; contact: research@example.com)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        pages = data["query"]["pages"]
        page  = next(iter(pages.values()))
        text  = page.get("extract", "")
        # Take first paragraph only
        first_para = text.strip().split("\n")[0]
        if len(first_para) < 20:
            first_para = text.strip()[:500]
        time.sleep(1)
        return first_para[:1000]  # cap length
    except Exception as e:
        print(f"  Warning: could not fetch text for '{concept}': {e}")
        return concept  # fallback to just the word


# ── Load existing embeddings ──────────────────────────────────────────────────

print("\nLoading existing embeddings...")
lm_existing  = np.load("lm_output/lm_hiddens.npy")   # [71, 4096] (includes abstract+physical)
st_existing  = np.load("lm_output/st_hiddens.npy")
clip_existing = np.load("lm_output/clip_hiddens.npy")
vj_existing  = np.load("lm_output/vjepa2_hiddens.npy")
mae_existing = np.load("lm_output/mae_hiddens.npy")

print(f"  LM:   {lm_existing.shape}")
print(f"  ST:   {st_existing.shape}")
print(f"  CLIP: {clip_existing.shape}")
print(f"  VJ:   {vj_existing.shape}")
print(f"  MAE:  {mae_existing.shape}")

# The first 17 rows of vj/mae/clip are the physical concepts
# We'll extract new concept embeddings and append to THOSE arrays
# For LM/ST we append to the full arrays
# Save a separate "physical_only" index file for clarity

# ── Load models ───────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# Sentence Transformers
print("Loading Sentence-Transformers...")
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("all-mpnet-base-v2")

# CLIP
print("Loading CLIP...")
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

# Mistral
print("Loading Mistral 7B (this takes a moment)...")
from transformers import AutoTokenizer, AutoModelForCausalLM
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    dtype=torch.float16,
    device_map="auto",
    output_hidden_states=True,
)
mistral_model.eval()

# V-JEPA 2
print("Loading V-JEPA 2...")
from transformers import AutoVideoProcessor, AutoModel
vjepa_processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
vjepa_model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
).to(device)
vjepa_model.eval()

# MAE
print("Loading MAE...")
from transformers import ViTMAEModel, ViTMAEConfig, AutoImageProcessor
mae_config = ViTMAEConfig.from_pretrained("facebook/vit-mae-large")
mae_config.mask_ratio = 0.0
mae_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
mae_model = ViTMAEModel.from_pretrained(
    "facebook/vit-mae-large",
    config=mae_config,
    ignore_mismatched_sizes=True,
).to(device)
mae_model.eval()

print("All models loaded.\n")

# ── Extraction functions ──────────────────────────────────────────────────────

def extract_mistral(text, layer=16):
    inputs = mistral_tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mistral_model(**inputs, output_hidden_states=True)
    hidden = out.hidden_states[layer]  # [1, seq_len, 4096]
    return hidden[0, -1, :].float().cpu().numpy()  # last token


def extract_st(text):
    return st_model.encode(text, normalize_embeddings=True)


def extract_clip(image):
    inputs = clip_proc(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(**inputs)
        pooled = vision_out.pooler_output
        projected = clip_model.visual_projection(pooled)
    return projected[0].float().cpu().numpy()


def extract_vjepa(image):
    # Repeat image 8x to form pseudo-video
    frames = [image] * 8
    inputs = vjepa_processor(videos=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = vjepa_model(**inputs, skip_predictor=True)
    # Mean of context tokens
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


def extract_mae(image):
    inputs = mae_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        out = mae_model(**inputs)
    # Mean of all patch tokens
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


# ── Extract new concepts ──────────────────────────────────────────────────────

new_lm   = []
new_st   = []
new_clip = []
new_vj   = []
new_mae  = []

log = []

for concept in NEW_CONCEPTS:
    print(f"Processing: {concept}")

    # Fetch content
    text  = get_wikipedia_text(concept)
    image = get_wikipedia_image(concept)
    print(f"  Text:  {text[:80]}...")
    print(f"  Image: {image.size}")

    # Extract
    lm_vec   = extract_mistral(text)
    st_vec   = extract_st(text)
    clip_vec = extract_clip(image)
    vj_vec   = extract_vjepa(image)
    mae_vec  = extract_mae(image)

    new_lm.append(lm_vec)
    new_st.append(st_vec)
    new_clip.append(clip_vec)
    new_vj.append(vj_vec)
    new_mae.append(mae_vec)

    log.append({
        "concept": concept,
        "text_preview": text[:200],
        "image_size": list(image.size),
        "lm_norm":   float(np.linalg.norm(lm_vec)),
        "st_norm":   float(np.linalg.norm(st_vec)),
        "clip_norm": float(np.linalg.norm(clip_vec)),
        "vj_norm":   float(np.linalg.norm(vj_vec)),
        "mae_norm":  float(np.linalg.norm(mae_vec)),
    })
    print(f"  ✓ extracted all 5 representations")
    print()

new_lm   = np.array(new_lm)
new_st   = np.array(new_st)
new_clip = np.array(new_clip)
new_vj   = np.array(new_vj)
new_mae  = np.array(new_mae)

print(f"New concept arrays:")
print(f"  LM:   {new_lm.shape}")
print(f"  ST:   {new_st.shape}")
print(f"  CLIP: {new_clip.shape}")
print(f"  VJ:   {new_vj.shape}")
print(f"  MAE:  {new_mae.shape}")

# ── Save expanded physical-only arrays ───────────────────────────────────────
# For experiments we want physical concepts only (original 17 + new 32)
# Rows 0:17 of vj/clip/mae/st are the original physical concepts

st_phys_orig  = st_existing[:17]
clip_phys_orig = clip_existing[:17]
vj_phys_orig  = vj_existing[:17]
mae_phys_orig = mae_existing[:17]
lm_phys_orig  = lm_existing[:17]

# Append new concepts
st_expanded   = np.vstack([st_phys_orig,   new_st])
clip_expanded = np.vstack([clip_phys_orig, new_clip])
vj_expanded   = np.vstack([vj_phys_orig,   new_vj])
mae_expanded  = np.vstack([mae_phys_orig,  new_mae])
lm_expanded   = np.vstack([lm_phys_orig,   new_lm])

print(f"\nExpanded physical arrays:")
print(f"  LM:   {lm_expanded.shape}")
print(f"  ST:   {st_expanded.shape}")
print(f"  CLIP: {clip_expanded.shape}")
print(f"  VJ:   {vj_expanded.shape}")
print(f"  MAE:  {mae_expanded.shape}")

# Save
np.save("lm_output/lm_hiddens_expanded.npy",   lm_expanded)
np.save("lm_output/st_hiddens_expanded.npy",   st_expanded)
np.save("lm_output/clip_hiddens_expanded.npy", clip_expanded)
np.save("lm_output/vjepa2_hiddens_expanded.npy", vj_expanded)
np.save("lm_output/mae_hiddens_expanded.npy",  mae_expanded)

# Save concept index so downstream scripts know row order
concept_index = {
    "all_concepts":      ALL_CONCEPTS,
    "original_concepts": ORIGINAL_CONCEPTS,
    "new_concepts":      NEW_CONCEPTS,
    "n_original":        len(ORIGINAL_CONCEPTS),
    "n_new":             len(NEW_CONCEPTS),
    "n_total":           len(ALL_CONCEPTS),
    "array_shapes": {
        "lm":   list(lm_expanded.shape),
        "st":   list(st_expanded.shape),
        "clip": list(clip_expanded.shape),
        "vj":   list(vj_expanded.shape),
        "mae":  list(mae_expanded.shape),
    },
    "note": "Row i corresponds to ALL_CONCEPTS[i]. Original 17 concepts are rows 0-16."
}

with open("lm_output/concept_index.json", "w", encoding="utf-8") as f:
    json.dump(concept_index, f, indent=2)

with open("lm_output/extraction_log_expanded.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2)

print("\nSaved:")
print("  lm_output/lm_hiddens_expanded.npy")
print("  lm_output/st_hiddens_expanded.npy")
print("  lm_output/clip_hiddens_expanded.npy")
print("  lm_output/vjepa2_hiddens_expanded.npy")
print("  lm_output/mae_hiddens_expanded.npy")
print("  lm_output/concept_index.json")
print("  lm_output/extraction_log_expanded.json")
print("\nDone. Run train_codebook_generalization.py next, pointing to *_expanded.npy files.")
