"""
LLM Extraction — Standalone Script
------------------------------------
Run this on your machine. It will:
    1. Download Mistral 7B (4-bit, ~4GB download, ~4GB VRAM)
    2. Extract hidden states for 71 concepts
    3. Save to lm_hiddens.npy
    4. Run a quick sanity check and print what it found

Then send me lm_hiddens.npy and lm_sanity.txt and we'll analyze them.

Requirements:
    pip install transformers bitsandbytes accelerate numpy scipy

Time: ~20 min first run (download), ~3 min after that
VRAM: ~4GB (4-bit quantized)
"""

import numpy as np
import json
import time
from pathlib import Path


# ── Concept list (inline so this file is self-contained) ──────────────────

CONCEPTS = {
    "physical": [
        "apple", "chair", "water", "fire", "stone", "rope",
        "door", "container", "shadow", "mirror", "knife", "wheel",
        "hand", "wall", "hole", "bridge", "ladder", "key",
    ],
    "actions": [
        "falling", "pushing", "grasping", "breaking", "pouring",
        "cutting", "balancing", "rolling", "bouncing", "sliding",
        "lifting", "spinning", "colliding", "melting", "flowing",
    ],
    "spatial": [
        "inside", "above", "beside", "behind", "between",
        "touching", "distance", "boundary", "path", "center",
        "surrounding", "through", "against", "along",
    ],
    "abstract": [
        "danger", "support", "intention", "causation", "similarity",
        "change", "direction", "force", "pattern", "constraint",
        "possibility", "sequence", "category", "quantity",
    ],
    "social": [
        "helping", "blocking", "following", "pointing", "giving",
        "taking", "waiting", "approaching", "avoiding", "leading",
    ],
}

ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}
CATEGORY_BOUNDARIES = {}

idx = 0
for category, items in CONCEPTS.items():
    CATEGORY_BOUNDARIES[category] = (idx, idx + len(items))
    for item in items:
        ALL_CONCEPTS.append(item)
        CONCEPT_CATEGORIES[item] = category
        idx += 1

# ── Prompt variants — we'll run all 3 and check consistency ───────────────

PROMPTS = {
    "concept":   "The concept of {concept} refers to:",
    "physical":  "The physical properties of {concept} include:",
    "imagine":   "When I imagine {concept}, I picture:",
}


# ── Extraction ─────────────────────────────────────────────────────────────

def extract():
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    except ImportError:
        print("Missing dependencies. Run:")
        print("  pip install transformers bitsandbytes accelerate torch")
        return

    model_id = "mistralai/Mistral-7B-v0.1"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("WARNING: No GPU detected. Running on CPU will be very slow (~2hrs).")
        print("If you have a GPU, make sure CUDA drivers are installed.")
        response = input("Continue on CPU? (y/n): ")
        if response.lower() != "y":
            return

    print(f"Device: {device}")
    print(f"Loading {model_id} with 4-bit quantization...")
    print("(First run downloads ~4GB — one time only)\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()
    print("Model loaded.\n")

    # Extract for each prompt variant
    all_results = {}

    for prompt_name, template in PROMPTS.items():
        print(f"Extracting with prompt: '{template}'")
        hiddens = []
        t0 = time.time()

        with torch.no_grad():
            for i, concept in enumerate(ALL_CONCEPTS):
                prompt = template.format(concept=concept)
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=64,
                ).to(device)

                outputs = model(**inputs, output_hidden_states=True)

                # Middle layer (16/32), last token — better semantic structure
                # than last layer + mean pool (which collapses all concepts together)
                mid_hidden = outputs.hidden_states[16]  # [1, seq, hidden]
                pooled = mid_hidden[0, -1, :].float().cpu().numpy()
                hiddens.append(pooled)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (len(ALL_CONCEPTS) - i - 1)
                    print(f"  {i+1:3d}/{len(ALL_CONCEPTS)}  "
                          f"elapsed: {elapsed:.0f}s  eta: {eta:.0f}s")

        all_results[prompt_name] = np.stack(hiddens)
        print(f"  Done. Shape: {all_results[prompt_name].shape}\n")

    return all_results


# ── Sentence-transformer extraction ─────────────────────────────────────────

def extract_st():
    from sentence_transformers import SentenceTransformer

    model_id = "sentence-transformers/all-mpnet-base-v2"
    print(f"Loading {model_id}...")
    model = SentenceTransformer(model_id)
    print("Model loaded.\n")

    all_results = {}
    for prompt_name, template in PROMPTS.items():
        print(f"Extracting with prompt: '{template}'")
        sentences = [template.format(concept=c) for c in ALL_CONCEPTS]
        embeddings = model.encode(sentences, show_progress_bar=True,
                                  convert_to_numpy=True)
        all_results[prompt_name] = embeddings
        print(f"  Done. Shape: {embeddings.shape}\n")

    return all_results


# ── CLIP visual extraction via Wikipedia images ──────────────────────────────

def extract_clip(n_images=3):
    import torch
    import requests
    import time
    import json
    from PIL import Image
    from io import BytesIO
    from transformers import CLIPProcessor, CLIPModel

    model_id = "openai/clip-vit-large-patch14"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device}...")
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()
    print("Model loaded.\n")

    session = requests.Session()
    session.headers.update({"User-Agent": "ConceptRSAExperiment/1.0 (research)"})

    # Only skip truly non-photographic file types / UI elements
    SKIP_KW = ("flag", "icon", "logo", "symbol", "ambox", "stub",
               "edit-", "question_mark", "red_question", "wikidata-logo",
               "portal-puzzle", "wikimedia-logo")

    def get_with_retry(url, params=None, max_retries=5, timeout=12):
        """GET with exponential backoff on 429."""
        delay = 2.0
        for attempt in range(max_retries):
            try:
                r = session.get(url, params=params, timeout=timeout)
                if r.status_code == 429:
                    wait = delay * (2 ** attempt)
                    print(f"    [429] backing off {wait:.0f}s", end="\r")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(delay * (2 ** attempt))
        return None

    def query_article_images(title):
        """Query Wikipedia for thumbnail + image list for a given article title."""
        r = get_with_retry(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": title,
                "prop": "pageimages|images",
                "format": "json",
                "pithumbsize": 336,
                "imlimit": 15,
                "redirects": 1,
            },
        )
        if r is None:
            return None
        page = next(iter(r.json()["query"]["pages"].values()))
        if page.get("pageid") is None:
            return None
        return page

    def get_extra_thumb_urls(file_titles, existing_urls):
        """Resolve file titles to thumbnail URLs via imageinfo."""
        urls = []
        r2 = get_with_retry(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "titles": "|".join(file_titles[:8]),
                "prop": "imageinfo",
                "iiprop": "url|mime",
                "iiurlwidth": 336,
                "format": "json",
            },
        )
        if r2 is None:
            return urls
        for pg in r2.json()["query"]["pages"].values():
            if "imageinfo" in pg:
                info = pg["imageinfo"][0]
                mime = info.get("mime", "")
                if mime.startswith("image/") and "svg" not in mime:
                    u = info.get("thumburl") or info.get("url", "")
                    if u and u not in existing_urls:
                        urls.append(u)
        return urls

    def get_wiki_thumb_urls(concept):
        """Fetch thumbnail URLs for a concept, with opensearch fallback."""
        urls = []
        try:
            page = query_article_images(concept)

            # Fallback: opensearch to find the canonical article title
            if page is None:
                r_srch = get_with_retry(
                    "https://en.wikipedia.org/w/api.php",
                    params={"action": "opensearch", "search": concept,
                            "limit": 3, "format": "json"},
                )
                if r_srch and r_srch.json() and len(r_srch.json()) > 1:
                    for alt in r_srch.json()[1]:
                        page = query_article_images(alt)
                        if page is not None:
                            break

            if page is None:
                return urls

            if "thumbnail" in page:
                urls.append(page["thumbnail"]["source"])

            if "images" in page and len(urls) < n_images:
                file_titles = [
                    img["title"] for img in page["images"]
                    if not img["title"].lower().endswith((".svg", ".gif"))
                    and not any(kw in img["title"].lower() for kw in SKIP_KW)
                ]
                if file_titles:
                    extra = get_extra_thumb_urls(file_titles, urls)
                    urls.extend(extra)

        except Exception as e:
            print(f"    [wiki] {e}")
        return urls[:n_images]

    def encode_image_url(url):
        """Download a thumbnail and return CLIP visual embedding."""
        r = get_with_retry(url, timeout=15)
        if r is None:
            raise ValueError(f"Max retries exceeded for {url}")
        img = Image.open(BytesIO(r.content)).convert("RGB")
        inp = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            # Use vision model + projection explicitly (robust across transformers versions)
            vision_out = model.vision_model(pixel_values=inp["pixel_values"])
            pooled = vision_out.pooler_output          # [1, hidden]
            feat = model.visual_projection(pooled)     # [1, proj_dim]
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().cpu().numpy()

    hiddens = []
    image_log = {}
    emb_dim = None
    n_failed = 0

    for i, concept in enumerate(ALL_CONCEPTS):
        urls = get_wiki_thumb_urls(concept)
        vecs = []
        for url in urls:
            try:
                vecs.append(encode_image_url(url))
            except Exception as e:
                print(f"    [img] {url[:60]}... {e}")

        image_log[concept] = {"urls": urls, "encoded": len(vecs)}

        if vecs:
            avg = np.mean(vecs, axis=0)
            if emb_dim is None:
                emb_dim = avg.shape[0]
            hiddens.append(avg)
            status = f"✓ {len(vecs)}/{len(urls)} imgs"
        else:
            if emb_dim is None:
                emb_dim = 768  # ViT-L/14 projection dim
            hiddens.append(np.zeros(emb_dim))
            n_failed += 1
            status = "✗ FAILED"

        print(f"  [{i+1:2d}/{len(ALL_CONCEPTS)}] {concept:15s}  {status}")
        time.sleep(1.5)  # 1.5s between concepts to stay within Wikipedia rate limits

    hiddens = np.stack(hiddens)
    print(f"\nShape: {hiddens.shape}  Failed: {n_failed}/{len(ALL_CONCEPTS)}")
    return {"concept": hiddens}, image_log


# ── Sanity checks ───────────────────────────────────────────────────────────

def sanity_check(all_results: dict) -> str:
    from scipy import stats
    from scipy.spatial.distance import cdist

    lines = []
    lines.append("=" * 60)
    lines.append("LLM EXTRACTION SANITY CHECK")
    lines.append("=" * 60)

    # Use the main prompt for primary analysis
    hiddens = all_results["concept"]
    N, D = hiddens.shape
    lines.append(f"\nShape: {N} concepts x {D} dimensions")

    # 1. Norm distribution
    norms = np.linalg.norm(hiddens, axis=-1)
    zero_mask = norms < 1e-8
    n_zero = zero_mask.sum()
    lines.append(f"\nVector norms:")
    lines.append(f"  mean={norms.mean():.2f}  std={norms.std():.2f}  "
                 f"min={norms.min():.2f}  max={norms.max():.2f}")
    if n_zero > 0:
        failed = [ALL_CONCEPTS[i] for i in np.where(zero_mask)[0]]
        lines.append(f"  WARNING: {n_zero} concepts have zero-norm vectors (no images found):")
        lines.append(f"  {failed}")
    elif norms.mean() > 0 and norms.std() / norms.mean() > 0.5:
        lines.append("  WARNING: High norm variance — some concepts may have")
        lines.append("  degenerate representations. Check the concept list.")

    # 2. Cosine similarity matrix — mask out zero vectors to avoid NaN
    safe_norms = np.where(zero_mask[:, None], 1.0, norms[:, None])
    normalized = hiddens / safe_norms
    normalized[zero_mask] = 0.0   # zero vectors → zero similarity with everything
    rsm = normalized @ normalized.T
    np.fill_diagonal(rsm, 0)  # exclude self-similarity

    lines.append(f"\nPairwise cosine similarities (off-diagonal):")
    lines.append(f"  mean={rsm.mean():.4f}  std={rsm.std():.4f}  "
                 f"min={rsm.min():.4f}  max={rsm.max():.4f}")

    # 3. Category clustering — the key check
    # Within-category similarity should be higher than across-category
    lines.append(f"\nWithin-category vs across-category similarity:")
    lines.append(f"  {'Category':12s} | {'within':>8s} | {'across':>8s} | "
                 f"{'ratio':>6s} | {'clustered?':>10s}")
    lines.append(f"  {'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")

    np.fill_diagonal(rsm, 0)
    for cat, (start, end) in CATEGORY_BOUNDARIES.items():
        # Within-category similarity (upper triangle of block)
        block = rsm[start:end, start:end]
        n = end - start
        triu = np.triu_indices(n, k=1)
        within = block[triu].mean() if len(triu[0]) > 0 else 0

        # Across-category similarity
        mask = np.zeros(N, dtype=bool)
        mask[start:end] = True
        cross_block = rsm[start:end, :][:, ~mask]
        across = cross_block.mean()

        ratio = within / (across + 1e-8)
        clustered = "✓ YES" if ratio > 1.2 else "✗ NO"
        lines.append(f"  {cat:12s} | {within:>8.4f} | {across:>8.4f} | "
                     f"{ratio:>6.2f} | {clustered:>10s}")

    # 4. Most similar pairs — intuition check
    np.fill_diagonal(rsm, -2)
    lines.append(f"\nTop 10 most similar concept pairs:")
    flat = rsm.copy()
    for _ in range(10):
        idx = np.unravel_index(flat.argmax(), flat.shape)
        i, j = idx
        sim = rsm[i, j]
        lines.append(f"  {ALL_CONCEPTS[i]:15s} <-> {ALL_CONCEPTS[j]:15s}  "
                     f"sim={sim:.4f}  "
                     f"({'same cat' if CONCEPT_CATEGORIES[ALL_CONCEPTS[i]] == CONCEPT_CATEGORIES[ALL_CONCEPTS[j]] else 'diff cat'})")
        flat[i, j] = -2
        flat[j, i] = -2

    # 5. Prompt consistency check
    if len(all_results) > 1:
        lines.append(f"\nPrompt consistency (do different prompts give similar RSMs?):")
        prompt_names = list(all_results.keys())
        for i in range(len(prompt_names)):
            for j in range(i+1, len(prompt_names)):
                na, nb = prompt_names[i], prompt_names[j]
                ha = all_results[na]
                hb = all_results[nb]
                # Build RSMs for each
                na_norm = ha / np.linalg.norm(ha, axis=-1, keepdims=True)
                nb_norm = hb / np.linalg.norm(hb, axis=-1, keepdims=True)
                rsm_a = na_norm @ na_norm.T
                rsm_b = nb_norm @ nb_norm.T
                N = rsm_a.shape[0]
                triu = np.triu_indices(N, k=1)
                r, p = stats.spearmanr(rsm_a[triu], rsm_b[triu])
                lines.append(f"  '{na}' vs '{nb}':  r={r:.4f}  p={p:.2e}")
                if r < 0.7:
                    lines.append(f"    WARNING: Low consistency. "
                                 f"Prompt choice significantly affects structure.")

    # 6. Overall verdict
    lines.append(f"\n{'='*60}")
    lines.append("VERDICT")
    lines.append(f"{'='*60}")

    n_clustered = sum(
        1 for cat, (start, end) in CATEGORY_BOUNDARIES.items()
        if _within_across_ratio(rsm, start, end, N) > 1.2
    )

    if n_clustered >= 3:
        lines.append("✓ LLM representations look healthy.")
        lines.append("  Categories show meaningful clustering.")
        lines.append("  Ready to compare against world model representations.")
    elif n_clustered >= 1:
        lines.append("~ Mixed results. Some categories cluster, others don't.")
        lines.append("  This may reflect genuine structure (abstract concepts")
        lines.append("  are less clustered in LLMs than physical ones).")
        lines.append("  Probably still usable -- wait for WM comparison.")
    else:
        lines.append("✗ No category clustering detected.")
        lines.append("  Either the representations are degenerate or the")
        lines.append("  concept list needs revision. Check the top pairs above.")

    return "\n".join(lines)


def _within_across_ratio(rsm, start, end, N):
    n = end - start
    block = rsm[start:end, start:end]
    triu = np.triu_indices(n, k=1)
    within = block[triu].mean() if len(triu[0]) > 0 else 0
    mask = np.zeros(N, dtype=bool)
    mask[start:end] = True
    np.fill_diagonal(rsm, 0)
    cross = rsm[start:end, :][:, ~mask].mean()
    return within / (cross + 1e-8)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("LLM Extraction — Step 1 of RSA Experiment")
    print("=" * 60)

    output_dir = Path("lm_output")
    output_dir.mkdir(exist_ok=True)

    # ── Mistral extraction ───────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Mistral 7B (layer 16, last token)")
    print("=" * 60)
    mistral_results = extract()
    if mistral_results is None:
        exit(1)

    np.save(output_dir / "lm_hiddens.npy", mistral_results["concept"])
    for name, arr in mistral_results.items():
        np.save(output_dir / f"lm_hiddens_{name}.npy", arr)
    print(f"Mistral hiddens saved to {output_dir}/")

    mistral_report = sanity_check(mistral_results)
    print(mistral_report)
    with open(output_dir / "lm_sanity.txt", "w", encoding="utf-8") as f:
        f.write(mistral_report)
    print(f"Mistral sanity report saved to {output_dir}/lm_sanity.txt")

    # ── Sentence-transformer extraction ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Sentence-transformers (all-mpnet-base-v2)")
    print("=" * 60)
    st_results = extract_st()

    np.save(output_dir / "st_hiddens.npy", st_results["concept"])
    for name, arr in st_results.items():
        np.save(output_dir / f"st_hiddens_{name}.npy", arr)
    print(f"ST hiddens saved to {output_dir}/")

    st_report = sanity_check(st_results)
    print(st_report)
    with open(output_dir / "st_sanity.txt", "w", encoding="utf-8") as f:
        f.write(st_report)
    print(f"ST sanity report saved to {output_dir}/st_sanity.txt")

    # ── CLIP extraction ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: CLIP ViT-L/14 (Wikipedia images, 3 per concept)")
    print("=" * 60)
    import json
    clip_results, image_log = extract_clip(n_images=3)

    np.save(output_dir / "clip_hiddens.npy", clip_results["concept"])
    print(f"CLIP hiddens saved to {output_dir}/clip_hiddens.npy")
    with open(output_dir / "clip_image_log.json", "w", encoding="utf-8") as f:
        json.dump(image_log, f, indent=2, ensure_ascii=False)
    print(f"Image log saved to {output_dir}/clip_image_log.json")

    clip_report = sanity_check(clip_results)
    print(clip_report)
    with open(output_dir / "clip_sanity.txt", "w", encoding="utf-8") as f:
        f.write(clip_report)
    print(f"CLIP sanity report saved to {output_dir}/clip_sanity.txt")

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("Four-way RSA comparison ready:")
    print("  lm_output/lm_hiddens.npy    — Mistral 7B hidden states (layer 16)")
    print("  lm_output/st_hiddens.npy    — all-mpnet-base-v2 embeddings")
    print("  lm_output/clip_hiddens.npy  — CLIP ViT-L/14 visual (Wikipedia)")
    print("  (world model representations to be compared against all three)")
