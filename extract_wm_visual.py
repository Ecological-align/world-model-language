"""
World Model Visual Extraction: V-JEPA 2 + MAE
----------------------------------------------
Extract latent representations from two visual-predictive models:

1. V-JEPA 2 (facebook/vjepa2-vitl-fpc64-256)
   - Temporal world model: predicts future visual states from video
   - For static images: duplicate across 8 frames as a static video clip
   - Extract encoder last_hidden_state, mean-pool across patch tokens

2. MAE (facebook/vit-mae-large)
   - Spatial prediction: reconstructs 75% masked image patches
   - Extract encoder last_hidden_state, mean-pool across patch tokens

Both use the same Wikipedia images as CLIP (from clip_image_log.json).
Images are downloaded locally first, then processed by both models.

Output:
    lm_output/vjepa2_hiddens.npy  -- [71, 1024] (zero vectors for non-physical)
    lm_output/mae_hiddens.npy     -- [71, 1024] (zero vectors for non-physical)
    lm_output/wm_extraction_log.json -- URLs and status per concept
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import numpy as np
import torch
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from extract_lm_standalone import ALL_CONCEPTS, CONCEPTS, CONCEPT_CATEGORIES

output_dir = Path("lm_output")
output_dir.mkdir(exist_ok=True)
image_cache_dir = output_dir / "image_cache"
image_cache_dir.mkdir(exist_ok=True)

N_CONCEPTS = len(ALL_CONCEPTS)
PHYSICAL_CONCEPTS = CONCEPTS["physical"]


# ── Phase 1: Download all images locally ─────────────────────────────────────

def download_all_images():
    """Download concept images from Wikipedia API, cache locally."""
    print("=" * 65)
    print("PHASE 1: DOWNLOADING IMAGES (Wikipedia API)")
    print("=" * 65)

    with open(output_dir / "clip_image_log.json", "r", encoding="utf-8") as f:
        image_log = json.load(f)

    image_map = {}  # concept -> list of local file paths

    for concept in ALL_CONCEPTS:
        if concept not in PHYSICAL_CONCEPTS:
            continue

        entry = image_log.get(concept, {})
        urls = entry.get("urls", [])
        if not urls:
            print(f"  {concept:12s} -- no URLs in clip_image_log.json, skipping")
            image_map[concept] = []
            continue

        local_paths = []
        for j, url in enumerate(urls):
            # Check cache first
            safe_name = f"{concept}_{j}.jpg"
            local_path = image_cache_dir / safe_name

            if local_path.exists() and local_path.stat().st_size > 100:
                local_paths.append(str(local_path))
                continue

            # Download with proper rate limiting
            try:
                r = requests.get(url, timeout=20, headers={
                    "User-Agent": "RSA-Experiment/1.0 (academic research; codebook project)"
                })
                if r.status_code == 403:
                    # Try fetching fresh URL via Wikipedia API
                    r = _fetch_via_api(concept, j, url)
                    if r is None:
                        print(f"  {concept:12s} img {j}: 403 + API fallback failed")
                        continue
                elif r.status_code == 429:
                    print(f"  {concept:12s} img {j}: 429, waiting 5s...")
                    time.sleep(5)
                    r = requests.get(url, timeout=20, headers={
                        "User-Agent": "RSA-Experiment/1.0 (academic research)"
                    })
                r.raise_for_status()

                # Verify it's a valid image
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img.save(local_path, "JPEG")
                local_paths.append(str(local_path))

            except Exception as e:
                print(f"  {concept:12s} img {j}: error - {e}")

            time.sleep(2.0)  # be very polite to Wikipedia

        n = len(local_paths)
        status = "OK" if n > 0 else "FAILED"
        print(f"  {concept:12s} -- {n}/{len(urls)} downloaded [{status}]")
        image_map[concept] = local_paths

    return image_map


def _fetch_via_api(concept, img_idx, original_url):
    """Fallback: fetch a fresh thumbnail URL via the Wikipedia API."""
    # Map concepts to Wikipedia article titles
    title_map = {
        "apple": "Apple", "chair": "Chair", "water": "Water", "fire": "Fire",
        "stone": "Rock_(geology)", "rope": "Rope", "door": "Door",
        "container": "Container", "shadow": "Shadow", "mirror": "Mirror",
        "knife": "Knife", "wheel": "Wheel", "hand": "Hand", "wall": "Wall",
        "hole": "Hole", "bridge": "Bridge", "ladder": "Ladder", "key": "Key_(lock)",
    }
    title = title_map.get(concept, concept.capitalize())

    try:
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "images" if img_idx > 0 else "pageimages",
            "pithumbsize": 500,
            "format": "json",
        }
        if img_idx == 0:
            # Get main page thumbnail
            r = requests.get(api_url, params=params, timeout=10, headers={
                "User-Agent": "RSA-Experiment/1.0 (academic research)"
            })
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                thumb = page.get("thumbnail", {}).get("source")
                if thumb:
                    time.sleep(1.5)
                    return requests.get(thumb, timeout=15, headers={
                        "User-Agent": "RSA-Experiment/1.0 (academic research)"
                    })
    except Exception:
        pass
    return None


# ── Phase 2: V-JEPA 2 extraction ────────────────────────────────────────────

def extract_vjepa2(image_map):
    print("\n" + "=" * 65)
    print("PHASE 2: V-JEPA 2 EXTRACTION (facebook/vjepa2-vitl-fpc64-256)")
    print("=" * 65)

    from transformers import AutoVideoProcessor, AutoModel

    model_name = "facebook/vjepa2-vitl-fpc64-256"
    print(f"Loading {model_name}...")

    processor = AutoVideoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model on {device}, dtype={next(model.parameters()).dtype}")

    hiddens = np.zeros((N_CONCEPTS, 1024), dtype=np.float32)
    extraction_log = {}

    for i, concept in enumerate(ALL_CONCEPTS):
        if concept not in PHYSICAL_CONCEPTS:
            extraction_log[concept] = {"status": "skipped", "reason": "non-physical"}
            continue

        paths = image_map.get(concept, [])
        if not paths:
            print(f"  [{i+1:2d}/71] {concept:12s} -- no images, zero vector")
            extraction_log[concept] = {"status": "no_images"}
            continue

        print(f"  [{i+1:2d}/71] {concept:12s} -- {len(paths)} images... ", end="", flush=True)

        embeddings = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                img_array = np.array(img)
                # Create 8-frame static video: (8, H, W, 3)
                video_frames = np.stack([img_array] * 8, axis=0)

                inputs = processor(video_frames, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs, skip_predictor=True)
                    lhs = outputs.last_hidden_state  # (1, seq_len, 1024)
                    pooled = lhs.mean(dim=1)  # (1, 1024)
                    embedding = pooled.squeeze(0).float().cpu().numpy()

                embeddings.append(embedding)
            except Exception as e:
                print(f"error({e}), ", end="")

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-8:
                avg = avg / norm
            hiddens[i] = avg
            print(f"OK ({len(embeddings)} encoded)")
        else:
            print(f"FAILED")

        extraction_log[concept] = {
            "status": "ok" if embeddings else "failed",
            "n_encoded": len(embeddings),
            "image_paths": paths,
        }

    np.save(output_dir / "vjepa2_hiddens.npy", hiddens)
    print(f"\nSaved vjepa2_hiddens.npy: shape={hiddens.shape}")

    del model
    torch.cuda.empty_cache()

    return hiddens, extraction_log


# ── Phase 3: MAE extraction ─────────────────────────────────────────────────

def extract_mae(image_map):
    print("\n" + "=" * 65)
    print("PHASE 3: MAE EXTRACTION (facebook/vit-mae-large)")
    print("=" * 65)

    from transformers import ViTMAEModel, ViTMAEConfig, ViTImageProcessor

    model_name = "facebook/vit-mae-large"
    print(f"Loading {model_name}...")

    processor = ViTImageProcessor.from_pretrained(model_name)
    config = ViTMAEConfig.from_pretrained(model_name)
    config.mask_ratio = 0.0
    model = ViTMAEModel.from_pretrained(model_name, config=config).to("cuda").half()
    model.eval()
    print(f"Masking disabled (mask_ratio=0), all {(config.image_size // config.patch_size)**2}+1 tokens kept")

    hiddens = np.zeros((N_CONCEPTS, 1024), dtype=np.float32)
    extraction_log = {}

    for i, concept in enumerate(ALL_CONCEPTS):
        if concept not in PHYSICAL_CONCEPTS:
            extraction_log[concept] = {"status": "skipped", "reason": "non-physical"}
            continue

        paths = image_map.get(concept, [])
        if not paths:
            print(f"  [{i+1:2d}/71] {concept:12s} -- no images, zero vector")
            extraction_log[concept] = {"status": "no_images"}
            continue

        print(f"  [{i+1:2d}/71] {concept:12s} -- {len(paths)} images... ", end="", flush=True)

        embeddings = []
        for path in paths:
            try:
                img = Image.open(path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                inputs = {k: v.to("cuda").half() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    lhs = outputs.last_hidden_state  # (1, 197, 1024)
                    pooled = lhs.mean(dim=1)  # (1, 1024)
                    embedding = pooled.squeeze(0).float().cpu().numpy()

                embeddings.append(embedding)
            except Exception as e:
                print(f"error({e}), ", end="")

        if embeddings:
            avg = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(avg)
            if norm > 1e-8:
                avg = avg / norm
            hiddens[i] = avg
            print(f"OK ({len(embeddings)} encoded)")
        else:
            print(f"FAILED")

        extraction_log[concept] = {
            "status": "ok" if embeddings else "failed",
            "n_encoded": len(embeddings),
            "image_paths": paths,
        }

    np.save(output_dir / "mae_hiddens.npy", hiddens)
    print(f"\nSaved mae_hiddens.npy: shape={hiddens.shape}")

    del model
    torch.cuda.empty_cache()

    return hiddens, extraction_log


# ── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(name, hiddens):
    norms = np.linalg.norm(hiddens, axis=-1)
    nonzero = np.sum(norms > 1e-8)
    print(f"\n  {name}: {nonzero}/{N_CONCEPTS} non-zero vectors, dim={hiddens.shape[1]}")

    phys_idx = [ALL_CONCEPTS.index(c) for c in PHYSICAL_CONCEPTS]
    phys_norms = norms[phys_idx]
    phys_nonzero = np.sum(phys_norms > 1e-8)
    print(f"  Physical: {phys_nonzero}/{len(PHYSICAL_CONCEPTS)} non-zero")

    if phys_nonzero < 3:
        print(f"  WARNING: Too few physical concepts for RSA")
        return

    valid_phys = [c for c in PHYSICAL_CONCEPTS if norms[ALL_CONCEPTS.index(c)] > 1e-8]
    if len(valid_phys) >= 3:
        idx = [ALL_CONCEPTS.index(c) for c in valid_phys]
        vecs = hiddens[idx]
        safe_norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        safe_norms = np.clip(safe_norms, 1e-8, None)
        normalized = vecs / safe_norms
        rsm = normalized @ normalized.T

        n = len(valid_phys)
        triu = rsm[np.triu_indices(n, k=1)]
        print(f"  Physical mean pairwise sim: {triu.mean():.4f} (std={triu.std():.4f})")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Phase 1: Download images locally (rate-limited, cached)
    image_map = download_all_images()

    full_log = {}

    # Phase 2: V-JEPA 2
    vjepa2_hiddens, vjepa2_log = extract_vjepa2(image_map)
    full_log["vjepa2"] = vjepa2_log
    sanity_check("V-JEPA 2", vjepa2_hiddens)

    # Phase 3: MAE
    mae_hiddens, mae_log = extract_mae(image_map)
    full_log["mae"] = mae_log
    sanity_check("MAE", mae_hiddens)

    # Save combined log
    with open(output_dir / "wm_extraction_log.json", "w", encoding="utf-8") as f:
        json.dump(full_log, f, indent=2)

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"  vjepa2_hiddens.npy: {vjepa2_hiddens.shape}")
    print(f"  mae_hiddens.npy:    {mae_hiddens.shape}")
    print(f"  wm_extraction_log.json saved")
