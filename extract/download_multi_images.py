"""
Download multiple images per concept using icrawler.

icrawler handles rate limiting, retries, and multiple search backends.
Uses Bing image search (no API key needed).

Requirements: pip install icrawler
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import shutil
from pathlib import Path

OUT_DIR = "lm_output/multi_images"
N_TARGET = 30
N_MIN = 10
os.makedirs(OUT_DIR, exist_ok=True)

ALL_CONCEPTS = [
    "apple","chair","water","fire","stone","rope","door","container",
    "shadow","mirror","knife","wheel","hand","wall","hole","bridge","ladder",
    "spring","bark","wave","charge","field","light","strike","press","shoot",
    "run","hammer","scissors","bowl","bucket","bench","fence","needle","drum",
    "clock","telescope","cloud","sand","ice","feather","leaf","thread","glass",
    "coin","shelf","pipe","net","chain",
]

# More specific search queries to avoid polysemy confusion
# (e.g. "bark" → tree bark, not dog bark)
SEARCH_QUERIES = {
    "apple":     "apple fruit red",
    "chair":     "wooden chair furniture",
    "water":     "water liquid flowing",
    "fire":      "fire flame burning",
    "stone":     "stone rock natural",
    "rope":      "rope fiber twisted",
    "door":      "wooden door entrance",
    "container": "container box object",
    "shadow":    "shadow cast ground",
    "mirror":    "mirror reflection glass",
    "knife":     "kitchen knife blade",
    "wheel":     "wheel round spokes",
    "hand":      "human hand fingers",
    "wall":      "stone brick wall",
    "hole":      "hole opening ground",
    "bridge":    "bridge over water",
    "ladder":    "wooden ladder steps",
    "spring":    "metal coil spring",
    "bark":      "tree bark texture close up",
    "wave":      "ocean wave water",
    "charge":    "electric spark discharge",
    "field":     "green field grass meadow",
    "light":     "beam of light sunlight",
    "strike":    "lightning strike bolt",
    "press":     "hydraulic press machine",
    "shoot":     "plant shoot sprout growing",
    "run":       "person running sport",
    "hammer":    "hammer tool metal",
    "scissors":  "scissors cutting tool",
    "bowl":      "ceramic bowl empty",
    "bucket":    "metal bucket pail",
    "bench":     "wooden bench park",
    "fence":     "wooden fence posts",
    "needle":    "sewing needle thread",
    "drum":      "drum musical instrument",
    "clock":     "clock face time",
    "telescope": "telescope optical instrument",
    "cloud":     "cloud sky white",
    "sand":      "sand grains beach",
    "ice":       "ice frozen water",
    "feather":   "bird feather close up",
    "leaf":      "green leaf plant",
    "thread":    "sewing thread spool",
    "glass":     "glass transparent object",
    "coin":      "coin metal currency",
    "shelf":     "wooden shelf furniture",
    "pipe":      "metal pipe tube",
    "net":       "fishing net mesh",
    "chain":     "metal chain links",
}

def download_via_icrawler(concept, concept_dir, existing):
    """Try Google, then Bing via icrawler."""
    import logging
    logging.disable(logging.CRITICAL)  # suppress icrawler spam

    needed = N_TARGET - existing
    query = SEARCH_QUERIES.get(concept, f"{concept} object photo")
    tmp_dir = concept_dir + "_tmp"

    for CrawlerClass in _get_crawlers():
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            crawler = CrawlerClass(
                storage={"root_path": tmp_dir},
                downloader_threads=4,
                parser_threads=2,
            )
            crawler.crawl(
                keyword=query,
                max_num=needed + 10,
                min_size=(100, 100),
                file_idx_offset=existing,
            )
        except Exception:
            pass

        # Move downloaded files
        idx = existing
        if os.path.exists(tmp_dir):
            for fname in sorted(os.listdir(tmp_dir)):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    src = os.path.join(tmp_dir, fname)
                    dst = os.path.join(concept_dir, f"{idx:03d}.jpg")
                    try:
                        shutil.move(src, dst)
                        idx += 1
                    except Exception:
                        pass
            shutil.rmtree(tmp_dir, ignore_errors=True)

        current = len([f for f in os.listdir(concept_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        if current >= N_MIN:
            return current
    return len([f for f in os.listdir(concept_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def _get_crawlers():
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
    return [GoogleImageCrawler, BingImageCrawler]


def download_via_wikimedia(concept, concept_dir, existing):
    """Fallback: fetch from Wikimedia Commons API."""
    import requests
    import time

    query = SEARCH_QUERIES.get(concept, concept)
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query", "format": "json",
        "generator": "search", "gsrnamespace": "6",
        "gsrsearch": query, "gsrlimit": 50,
        "prop": "imageinfo", "iiprop": "url|size|mime",
        "iiurlwidth": 500,
    }
    headers = {"User-Agent": "CodebookResearch/1.0 (research project; mailto:rmjocic@gmail.com)"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=30)
        data = resp.json()
    except Exception:
        return existing

    pages = data.get("query", {}).get("pages", {})
    idx = existing
    for pid, page in sorted(pages.items(), key=lambda x: x[0]):
        if idx >= N_TARGET:
            break
        for info in page.get("imageinfo", []):
            mime = info.get("mime", "")
            if "image" not in mime:
                continue
            img_url = info.get("thumburl") or info.get("url")
            if not img_url:
                continue
            dst = os.path.join(concept_dir, f"{idx:03d}.jpg")
            if os.path.exists(dst):
                idx += 1
                continue
            try:
                r = requests.get(img_url, headers=headers, timeout=15)
                if r.status_code == 200 and len(r.content) > 5000:
                    with open(dst, "wb") as f:
                        f.write(r.content)
                    idx += 1
                time.sleep(0.5)
            except Exception:
                pass
    return len([f for f in os.listdir(concept_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def download_concept(concept, concept_dir):
    existing = len([f for f in os.listdir(concept_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    if existing >= N_TARGET:
        return existing, "cached"

    # Try icrawler first
    n = download_via_icrawler(concept, concept_dir, existing)
    if n >= N_MIN:
        return n, "icrawler"

    # Fallback to Wikimedia Commons
    n = download_via_wikimedia(concept, concept_dir, n)
    return n, "wikimedia" if n > existing else "icrawler"


def main():
    print("=" * 60)
    print("MULTI-IMAGE DOWNLOAD  (Google/Bing via icrawler + Wikimedia)")
    print(f"Target: {N_TARGET} images × {len(ALL_CONCEPTS)} concepts")
    print("=" * 60 + "\n")

    index = {}
    total = 0

    for concept in ALL_CONCEPTS:
        cdir = os.path.join(OUT_DIR, concept)
        os.makedirs(cdir, exist_ok=True)
        print(f"  {concept:<14}", end=" ", flush=True)
        n, src = download_concept(concept, cdir)
        flag = "✓" if n >= N_MIN else "⚠"
        print(f"{flag} {n}/{N_TARGET}")
        if n < N_MIN:
            print(f"    ⚠ WARNING: only {n} images")
        index[concept] = {"n": n, "source": src}
        total += n

    print(f"\n{'='*60}")
    print(f"Total: {total} images across {len(ALL_CONCEPTS)} concepts")
    below = [c for c, v in index.items() if v["n"] < N_MIN]
    if below:
        print(f"Below minimum ({N_MIN}): {below}")
    else:
        print("All concepts above minimum.")

    with open("lm_output/multi_image_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print("Saved: lm_output/multi_image_index.json")


if __name__ == "__main__":
    main()
