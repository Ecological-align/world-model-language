"""
extract_phrase_level.py
=======================

Extract phrase-level embeddings for all (phrase, image) pairs in phrase_bank.py.

For each event pair (phrase, image_query):
  - LLM (Mistral):    last-token hidden state of the PHRASE (not the bare word)
  - ST:               mean-pooled embedding of the PHRASE
  - CLIP (visual):    image embedding of the best-matched image for the event
  - V-JEPA 2:         same image repeated 8× (same workaround as before)
  - MAE:              same image

Key difference from concept-level extraction:
  - Input to LLM/ST is a full grounded phrase: "water flowing downhill over rocks"
    instead of just "water"
  - Image is fetched for the SPECIFIC PHYSICAL SENSE of the event, not a
    generic Wikipedia thumbnail for the concept word

Outputs (in lm_output/phrase_level/):
  lm_hiddens_phrase.npy       [N_events, 4096]   Mistral phrase embeddings
  st_hiddens_phrase.npy       [N_events, 768]    ST phrase embeddings
  clip_hiddens_phrase.npy     [N_events, 768]    CLIP image embeddings
  vjepa2_hiddens_phrase.npy   [N_events, 1024]   V-JEPA 2 image embeddings
  mae_hiddens_phrase.npy      [N_events, 1024]   MAE image embeddings
  event_index.json            Maps row index → {concept, phrase, image_query, ...}

Run:
  python extract_phrase_level.py

Estimated time: ~90 min on RTX 5090 (image fetches + 5 model passes per event)
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
from pathlib import Path

from phrase_bank import PHRASE_BANK

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "lm_output/phrase_level"
IMAGE_DIR    = "lm_output/phrase_images"
MISTRAL_LAYER = 16          # same as concept-level extraction
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

# ── Build event list ──────────────────────────────────────────────────────────
events = []  # list of dicts: {concept, phrase, image_query, event_id}
for concept, pairs in PHRASE_BANK.items():
    for i, (phrase, image_query) in enumerate(pairs):
        events.append({
            "concept":     concept,
            "phrase":      phrase,
            "image_query": image_query,
            "event_id":    f"{concept}__{i}",
        })

N = len(events)
print(f"Total events: {N}")
print(f"Concepts:     {len(PHRASE_BANK)}")

# ── Image fetching ────────────────────────────────────────────────────────────

def fetch_wikimedia_image(query, cache_path):
    """
    Search Wikimedia Commons for query, download the first relevant image.
    Falls back to a Wikipedia article image if Commons fails.
    """
    if os.path.exists(cache_path):
        return Image.open(cache_path).convert("RGB")

    headers = {"User-Agent": "Mozilla/5.0 (phrase-level codebook research)"}

    # Try Wikimedia Commons search API
    try:
        url = (
            "https://commons.wikimedia.org/w/api.php"
            f"?action=query&list=search&srsearch={requests.utils.quote(query)}"
            "&srnamespace=6&srlimit=5&format=json"
        )
        resp = requests.get(url, headers=headers, timeout=10)
        results = resp.json().get("query", {}).get("search", [])
        for result in results:
            title = result["title"]  # e.g. "File:Water flowing.jpg"
            # Get image URL via imageinfo API
            info_url = (
                "https://commons.wikimedia.org/w/api.php"
                f"?action=query&titles={requests.utils.quote(title)}"
                "&prop=imageinfo&iiprop=url&iiurlwidth=512&format=json"
            )
            info_resp = requests.get(info_url, headers=headers, timeout=10)
            pages = info_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                img_url = page.get("imageinfo", [{}])[0].get("thumburl", "")
                if img_url and any(img_url.lower().endswith(ext)
                                   for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                    time.sleep(1.5)
                    img_resp = requests.get(img_url, headers=headers, timeout=15)
                    img = Image.open(BytesIO(img_resp.content)).convert("RGB")
                    img.save(cache_path)
                    return img
    except Exception as e:
        print(f"    Wikimedia failed for '{query}': {e}")

    # Fallback: Wikipedia article image for first word of query
    concept_word = query.split()[0]
    try:
        url = (
            f"https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={concept_word}&prop=pageimages"
            f"&format=json&pithumbsize=512"
        )
        resp = requests.get(url, headers=headers, timeout=10)
        pages = resp.json()["query"]["pages"]
        page = next(iter(pages.values()))
        img_url = page["thumbnail"]["source"]
        time.sleep(1.5)
        img_resp = requests.get(img_url, headers=headers, timeout=15)
        img = Image.open(BytesIO(img_resp.content)).convert("RGB")
        img.save(cache_path)
        print(f"    Used Wikipedia fallback for '{query}'")
        return img
    except Exception as e:
        print(f"    Wikipedia fallback also failed for '{query}': {e}")
        return Image.new("RGB", (224, 224), color=(100, 100, 100))


def get_image(event):
    safe_id = event["event_id"].replace("/", "_").replace(" ", "_")
    cache_path = os.path.join(IMAGE_DIR, f"{safe_id}.jpg")
    return fetch_wikimedia_image(event["image_query"], cache_path)


# ── Load models ───────────────────────────────────────────────────────────────

print("\nLoading models...")

# Mistral
from transformers import AutoTokenizer, AutoModelForCausalLM
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
mistral_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16,
    device_map="auto",
)
mistral_model.eval()
print("  Mistral loaded")

# Sentence Transformer
from sentence_transformers import SentenceTransformer
st_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
print("  ST loaded")

# CLIP
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE)
clip_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()
print("  CLIP loaded")

# V-JEPA 2
from transformers import AutoModel, AutoVideoProcessor
vjepa_model = AutoModel.from_pretrained(
    "facebook/vjepa2-vitl-fpc64-256",
    dtype=torch.float16,
).to(DEVICE)
vjepa_proc = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256")
vjepa_model.eval()
print("  V-JEPA 2 loaded")

# MAE
from transformers import ViTMAEModel, AutoImageProcessor
mae_model = ViTMAEModel.from_pretrained("facebook/vit-mae-large").to(DEVICE)
mae_proc  = AutoImageProcessor.from_pretrained("facebook/vit-mae-large")
mae_model.config.mask_ratio = 0.0
mae_model.eval()
print("  MAE loaded")


# ── Extraction functions ───────────────────────────────────────────────────────

def extract_mistral_phrase(phrase):
    """
    Extract Mistral hidden state for a grounded phrase.
    Uses a short instructional wrapper to ensure the model processes the
    physical meaning rather than treating it as a bare prompt.
    """
    # Wrap in instruction format to activate physical reasoning mode
    text = f"[INST] Describe the physical dynamics of the following event: {phrase} [/INST]"
    inputs = mistral_tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = mistral_model(**inputs, output_hidden_states=True)
    # Take hidden state at layer 16, last token of the prompt
    # (This is the final token of [/INST], where the model is about to generate)
    hidden = out.hidden_states[MISTRAL_LAYER]  # [1, seq_len, 4096]
    return hidden[0, -1, :].float().cpu().numpy()


def extract_st_phrase(phrase):
    """ST encodes the phrase directly — it's designed for sentence-level input."""
    return st_model.encode(phrase, normalize_embeddings=True)


def extract_clip_image(image):
    inputs = clip_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_out = clip_model.vision_model(**inputs)
        pooled = vision_out.pooler_output
        projected = clip_model.visual_projection(pooled)
    return projected[0].float().cpu().numpy()


def extract_vjepa(image):
    frames = [image] * 8
    inputs = vjepa_proc(videos=frames, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        out = vjepa_model(**inputs, skip_predictor=True)
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


def extract_mae(image):
    inputs = mae_proc(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = mae_model(**inputs)
    return out.last_hidden_state[0].mean(0).float().cpu().numpy()


# ── Main extraction loop ───────────────────────────────────────────────────────

lm_hiddens    = np.zeros((N, 4096),  dtype=np.float32)
st_hiddens    = np.zeros((N, 768),   dtype=np.float32)
clip_hiddens  = np.zeros((N, 768),   dtype=np.float32)
vjepa_hiddens = np.zeros((N, 1024),  dtype=np.float32)
mae_hiddens   = np.zeros((N, 1024),  dtype=np.float32)
event_log     = []

print(f"\nExtracting {N} events...\n")

for i, event in enumerate(events):
    phrase = event["phrase"]
    concept = event["concept"]
    t0 = time.time()

    # Fetch image first (cached after first run)
    image = get_image(event)

    # Extract all 5 representations
    lm_h    = extract_mistral_phrase(phrase)
    st_h    = extract_st_phrase(phrase)
    clip_h  = extract_clip_image(image)
    vjepa_h = extract_vjepa(image)
    mae_h   = extract_mae(image)

    lm_hiddens[i]    = lm_h
    st_hiddens[i]    = st_h
    clip_hiddens[i]  = clip_h
    vjepa_hiddens[i] = vjepa_h
    mae_hiddens[i]   = mae_h

    event_log.append({**event, "row": i})
    elapsed = time.time() - t0

    if (i + 1) % 10 == 0 or i < 5:
        print(f"  [{i+1:3d}/{N}] {concept:12s} | {phrase[:55]:<55s} ({elapsed:.1f}s)")

# ── Save ──────────────────────────────────────────────────────────────────────

np.save(os.path.join(OUTPUT_DIR, "lm_hiddens_phrase.npy"),    lm_hiddens)
np.save(os.path.join(OUTPUT_DIR, "st_hiddens_phrase.npy"),    st_hiddens)
np.save(os.path.join(OUTPUT_DIR, "clip_hiddens_phrase.npy"),  clip_hiddens)
np.save(os.path.join(OUTPUT_DIR, "vjepa2_hiddens_phrase.npy"),vjepa_hiddens)
np.save(os.path.join(OUTPUT_DIR, "mae_hiddens_phrase.npy"),   mae_hiddens)

with open(os.path.join(OUTPUT_DIR, "event_index.json"), "w", encoding="utf-8") as f:
    json.dump({
        "events":        event_log,
        "n_events":      N,
        "n_concepts":    len(PHRASE_BANK),
        "concepts":      list(PHRASE_BANK.keys()),
        "events_per_concept": {c: len(v) for c, v in PHRASE_BANK.items()},
    }, f, indent=2)

print(f"\nSaved to {OUTPUT_DIR}/")
print(f"  lm_hiddens_phrase.npy:     {lm_hiddens.shape}")
print(f"  st_hiddens_phrase.npy:     {st_hiddens.shape}")
print(f"  clip_hiddens_phrase.npy:   {clip_hiddens.shape}")
print(f"  vjepa2_hiddens_phrase.npy: {vjepa_hiddens.shape}")
print(f"  mae_hiddens_phrase.npy:    {mae_hiddens.shape}")
print(f"  event_index.json:          {N} events")
