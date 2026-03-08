"""
Find and download video clips per concept from Something-Something v2
and Kinetics-700 (both free for research).

Something-Something v2 (SS2):
  - 220k clips of humans performing actions with everyday objects
  - Download requires registration at: https://developer.qualcomm.com/software/ai-datasets/something-something
  - Once downloaded, this script indexes clips by the objects they contain

Kinetics-700:
  - YouTube clips, free download via yt-dlp
  - This script downloads 5-10 clips per concept using yt-dlp + Kinetics label matching

UCF-101:
  - Direct download, no registration
  - 101 action classes, some overlap with our concepts

This script has TWO modes:
  MODE 1: Index existing SS2 download (if you have it)
  MODE 2: Download from Kinetics-700 + UCF-101 (no registration needed)

Requirements:
  pip install yt-dlp requests tqdm
  (yt-dlp must also be on PATH or installed as package)

Output:
  lm_output/concept_videos/{concept}/0.mp4 ... N.mp4
  lm_output/video_index.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import time
import requests
import subprocess
import random
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
N_CLIPS_PER_CONCEPT  = 5    # target clips per concept (more = slower)
CLIP_DURATION_SEC    = 3    # how many seconds to download
OUTPUT_DIR           = "lm_output/concept_videos"
SS2_DIR              = None  # Set to your SS2 path if downloaded, e.g. "D:/ss2/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_CONCEPTS = [
    "apple", "chair", "water", "fire", "stone", "rope", "door",
    "container", "shadow", "mirror", "knife", "wheel", "hand",
    "wall", "hole", "bridge", "ladder",
    "spring", "bark", "wave", "charge", "field", "light", "strike",
    "press", "shoot", "run", "hammer", "scissors", "bowl", "bucket",
    "bench", "fence", "needle", "drum", "clock", "telescope",
    "cloud", "sand", "ice", "feather", "leaf", "thread", "glass",
    "coin", "shelf", "pipe", "net", "chain",
]

# ── Something-Something v2 label patterns ────────────────────────────────────
# SS2 labels are action templates like "Pushing [something] from left to right"
# Map concepts to SS2 label substrings

SS2_CONCEPT_PATTERNS = {
    "apple":     ["apple"],
    "chair":     ["chair"],
    "water":     ["water", "liquid"],
    "fire":      ["fire", "candle", "lighter"],
    "stone":     ["stone", "rock"],
    "rope":      ["rope", "string", "cord"],
    "door":      ["door"],
    "knife":     ["knife", "cutting"],
    "wheel":     ["wheel", "rolling"],
    "hand":      ["hand"],
    "hammer":    ["hammer"],
    "scissors":  ["scissors", "cutting"],
    "bowl":      ["bowl"],
    "bucket":    ["bucket"],
    "bench":     ["bench"],
    "drum":      ["drum"],
    "clock":     ["clock"],
    "leaf":      ["leaf"],
    "thread":    ["thread"],
    "glass":     ["glass"],
    "coin":      ["coin"],
    "net":       ["net"],
    "chain":     ["chain"],
    "sand":      ["sand"],
    "ice":       ["ice"],
    "feather":   ["feather"],
    "needle":    ["needle"],
    "pipe":      ["pipe", "tube"],
}

# ── Kinetics-700 class mappings ───────────────────────────────────────────────
# Kinetics class names → our concepts
KINETICS_CLASS_MAP = {
    "water":     ["swimming", "diving", "waterfall climbing", "surfing water"],
    "fire":      ["lighting fire", "tending fire", "blowing out candles"],
    "stone":     ["rock climbing", "skipping stone"],
    "rope":      ["jumping rope", "tying knot"],
    "knife":     ["sharpening knives", "slicing vegetables"],
    "wave":      ["surfing water", "swimming"],
    "run":       ["running", "jogging"],
    "shoot":     ["archery", "shooting goal", "shooting basketball"],
    "cloud":     ["paragliding", "skydiving"],
    "ice":       ["ice skating", "ice climbing"],
    "leaf":      ["raking leaves"],
    "sand":      ["sandcastle building"],
    "hammer":    ["using a hammer"],
    "scissors":  ["cutting hair"],
    "drum":      ["drumming fingers", "playing drums"],
    "clock":     ["winding up"],
    "bench":     ["bench pressing"],
    "hand":      ["clapping", "waving hand"],
    "chair":     ["sitting down", "standing up"],
    "spring":    ["bungee jumping"],
    "strike":    ["bowling", "hitting baseball"],
}

# ── UCF-101 class mappings ────────────────────────────────────────────────────
UCF_CLASS_MAP = {
    "fire":    ["Blowing Candles"],
    "rope":    ["RopeClimbing"],
    "knife":   ["Knifing"],
    "run":     ["Running"],
    "water":   ["Diving", "Swimming"],
    "shoot":   ["Archery", "ShootingBallsports"],
    "hammer":  ["Hammering"],
    "drum":    ["Drumming"],
    "ice":     ["IceDancing"],
    "hand":    ["HandstandWalking"],
}

UCF_BASE_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"


# ══════════════════════════════════════════════════════════════════════════════
# MODE 1: Index existing Something-Something v2 download
# ══════════════════════════════════════════════════════════════════════════════

def index_ss2(ss2_dir):
    """
    If you've downloaded SS2, this indexes clips by concept.
    SS2 labels are in: {ss2_dir}/labels.json or something-something-v2-labels.json
    Videos are in:     {ss2_dir}/videos/*.webm
    """
    print("=" * 65)
    print("MODE 1: Indexing Something-Something v2")
    print("=" * 65)

    # Find label file
    label_candidates = [
        os.path.join(ss2_dir, "something-something-v2-labels.json"),
        os.path.join(ss2_dir, "labels.json"),
        os.path.join(ss2_dir, "annotations", "labels.json"),
    ]
    label_file = next((f for f in label_candidates if os.path.exists(f)), None)
    if not label_file:
        print(f"ERROR: Cannot find SS2 label file in {ss2_dir}")
        return {}

    # Find annotations (train/val split files)
    train_file = os.path.join(ss2_dir, "something-something-v2-train.json")
    val_file   = os.path.join(ss2_dir, "something-something-v2-validation.json")

    all_annotations = []
    for ann_file in [train_file, val_file]:
        if os.path.exists(ann_file):
            with open(ann_file, encoding="utf-8") as f:
                all_annotations.extend(json.load(f))

    print(f"Loaded {len(all_annotations)} SS2 annotations")

    # Index by concept
    concept_clips = {c: [] for c in ALL_CONCEPTS}
    for item in all_annotations:
        label = item.get("label", "").lower()
        placeholders = item.get("placeholders", [])
        video_id = item.get("id", "")
        for concept in ALL_CONCEPTS:
            patterns = SS2_CONCEPT_PATTERNS.get(concept, [concept])
            if any(p in label for p in patterns) or \
               any(p in ph.lower() for p in patterns for ph in placeholders):
                video_path = os.path.join(ss2_dir, "videos", f"{video_id}.webm")
                if os.path.exists(video_path):
                    concept_clips[concept].append(video_path)

    # Copy N clips per concept to output dir
    index = {}
    for concept, clips in concept_clips.items():
        if not clips:
            print(f"  {concept}: no SS2 clips found")
            continue
        concept_dir = os.path.join(OUTPUT_DIR, concept)
        os.makedirs(concept_dir, exist_ok=True)
        selected = random.sample(clips, min(N_CLIPS_PER_CONCEPT, len(clips)))
        for i, clip_path in enumerate(selected):
            dst = os.path.join(concept_dir, f"{i:03d}.webm")
            if not os.path.exists(dst):
                import shutil
                shutil.copy2(clip_path, dst)
        print(f"  {concept}: {len(clips)} clips available, copied {len(selected)}")
        index[concept] = {"n": len(selected), "source": "ss2",
                          "path": concept_dir, "total_available": len(clips)}

    return index


# ══════════════════════════════════════════════════════════════════════════════
# MODE 2: Download from YouTube (Kinetics queries) via yt-dlp
# ══════════════════════════════════════════════════════════════════════════════

def check_ytdlp():
    try:
        result = subprocess.run(["yt-dlp", "--version"],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Try as Python module
    try:
        result = subprocess.run([sys.executable, "-m", "yt_dlp", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


def ytdlp_cmd():
    """Return the yt-dlp command prefix."""
    try:
        r = subprocess.run(["yt-dlp", "--version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return ["yt-dlp"]
    except Exception:
        pass
    return [sys.executable, "-m", "yt_dlp"]


def download_youtube_clip(query, save_path, duration_sec=3):
    """
    Search YouTube for query, download first result as short clip.
    Uses yt-dlp with ytsearch.
    """
    cmd = ytdlp_cmd() + [
        f"ytsearch1:{query}",
        "--output", save_path,
        "--format", "mp4[height<=480]/mp4/best[height<=480]",
        "--download-sections", f"*0-{duration_sec}",
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        "--extractor-args", "youtube:skip=dash",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0 and os.path.exists(save_path)
    except (subprocess.TimeoutExpired, Exception):
        return False


def download_kinetics_style(concept, concept_dir):
    """Download clips for a concept using Kinetics-style YouTube search queries."""
    queries = []

    # Kinetics class matches
    kinetics_classes = KINETICS_CLASS_MAP.get(concept, [])
    for cls in kinetics_classes:
        queries.append(f"{cls} video clip short")

    # Generic physical queries for the concept
    queries += [
        f"{concept} physical demonstration slow motion",
        f"{concept} natural motion close up",
        f"how does {concept} move",
        f"{concept} in action real world",
        f"physical properties of {concept}",
    ]

    downloaded = 0
    for i, query in enumerate(queries):
        if downloaded >= N_CLIPS_PER_CONCEPT:
            break
        save_path = os.path.join(concept_dir, f"{downloaded:03d}.mp4")
        if os.path.exists(save_path):
            downloaded += 1
            continue
        print(f"    Query: '{query[:50]}...'")
        if download_youtube_clip(query, save_path, CLIP_DURATION_SEC):
            downloaded += 1
            print(f"    ✓ Downloaded clip {downloaded}")
        else:
            print(f"    ✗ Failed")
        time.sleep(1.0)  # rate limit

    return downloaded


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("CONCEPT VIDEO DOWNLOADER")
    print("=" * 65)

    index = {}

    # Mode 1: SS2 indexing
    if SS2_DIR and os.path.exists(SS2_DIR):
        print(f"\nFound SS2 directory: {SS2_DIR}")
        ss2_index = index_ss2(SS2_DIR)
        index.update(ss2_index)
        concepts_remaining = [c for c in ALL_CONCEPTS
                              if c not in index or index[c]["n"] < N_CLIPS_PER_CONCEPT]
    else:
        print("\nSS2_DIR not set or not found. Using YouTube download only.")
        print("To use SS2: download from https://developer.qualcomm.com/software/ai-datasets/something-something")
        print("Then set SS2_DIR at the top of this script.\n")
        concepts_remaining = ALL_CONCEPTS

    # Mode 2: YouTube download
    if not check_ytdlp():
        print("yt-dlp not found. Install with: pip install yt-dlp")
        print("Then re-run this script.")
        # Save partial index
        with open("lm_output/video_index.json", "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        return

    print(f"Downloading YouTube clips for {len(concepts_remaining)} concepts...")
    print(f"Target: {N_CLIPS_PER_CONCEPT} clips × {CLIP_DURATION_SEC}s each\n")

    for concept in concepts_remaining:
        concept_dir = os.path.join(OUTPUT_DIR, concept)
        os.makedirs(concept_dir, exist_ok=True)

        existing = [f for f in os.listdir(concept_dir)
                    if f.endswith((".mp4", ".webm"))]
        if len(existing) >= N_CLIPS_PER_CONCEPT:
            print(f"  {concept}: already have {len(existing)} clips")
            index[concept] = {"n": len(existing), "source": "cached",
                              "path": concept_dir}
            continue

        print(f"\n  {concept}:")
        n = download_kinetics_style(concept, concept_dir)
        print(f"  → {n}/{N_CLIPS_PER_CONCEPT} clips downloaded")
        index[concept] = {"n": n, "source": "youtube", "path": concept_dir}

    # Report
    print("\n" + "=" * 65)
    total = sum(v["n"] for v in index.values())
    print(f"Total clips: {total} across {len(index)} concepts")
    missing = [c for c in ALL_CONCEPTS
               if c not in index or index[c]["n"] == 0]
    if missing:
        print(f"No clips found for: {missing}")

    with open("lm_output/video_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    print("Saved to lm_output/video_index.json")


if __name__ == "__main__":
    main()
