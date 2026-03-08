"""
check_phrase_images.py
======================

Quick visual check: opens all cached images for a given concept
in a grid so you can verify they're event-specific, not generic.

Usage:
  python check_phrase_images.py run
  python check_phrase_images.py water strike light
  python check_phrase_images.py --all-polysemous
  python check_phrase_images.py --report         (text-only, no display)

Requires: pillow, matplotlib
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import argparse
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from phrase_bank import PHRASE_BANK

IMAGE_DIR = "lm_output/phrase_images"

POLYSEMOUS = ["run", "strike", "press", "light", "charge", "bark",
              "spring", "wave", "field", "shoot", "leaf", "glass"]


def get_image_path(concept, event_idx):
    event_id = f"{concept}__{event_idx}"
    return os.path.join(IMAGE_DIR, f"{event_id}.jpg")


def check_concept(concept, show=True):
    pairs = PHRASE_BANK.get(concept)
    if not pairs:
        print(f"Unknown concept: {concept}")
        return

    n = len(pairs)
    found, missing, fallback = [], [], []

    for i, (phrase, query) in enumerate(pairs):
        path = get_image_path(concept, i)
        if os.path.exists(path):
            found.append((i, phrase, query, path))
        else:
            missing.append((i, phrase, query))

    print(f"\n{'='*60}")
    print(f"CONCEPT: {concept}  ({len(found)}/{n} images found)")
    print(f"{'='*60}")
    for i, phrase, query, path in found:
        size = os.path.getsize(path) // 1024
        # Flag likely grey placeholders (very small files)
        flag = " ⚠ PLACEHOLDER?" if size < 5 else ""
        print(f"  [{i}] {phrase[:55]}")
        print(f"       query: {query}")
        print(f"       file:  {os.path.basename(path)} ({size}KB){flag}")
    for i, phrase, query in missing:
        print(f"  [{i}] NOT YET DOWNLOADED: {phrase[:55]}")

    if not found or not show:
        return

    # Display grid
    cols = min(3, len(found))
    rows = (len(found) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle(f'"{concept}" — event-specific images\n(check: each image should show a different physical situation)',
                 fontsize=13, fontweight='bold')

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < len(found):
                i, phrase, query, path = found[idx]
                try:
                    img = Image.open(path).convert("RGB")
                    ax.imshow(img)
                    # Wrap phrase text
                    words = phrase.split()
                    lines, line = [], []
                    for w in words:
                        line.append(w)
                        if len(' '.join(line)) > 32:
                            lines.append(' '.join(line[:-1]))
                            line = [w]
                    if line:
                        lines.append(' '.join(line))
                    ax.set_title('\n'.join(lines), fontsize=9, pad=4)
                    ax.axis('off')

                    # Red border if placeholder-sized
                    size = os.path.getsize(path) // 1024
                    if size < 5:
                        for spine in ax.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error:\n{e}', ha='center', va='center',
                            transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
                idx += 1
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.show()


def text_report():
    """Text-only report: coverage + placeholder detection for all concepts."""
    print("\nPHRASE IMAGE COVERAGE REPORT")
    print("="*65)
    print(f"{'Concept':<14} {'Found':>5} {'Total':>5} {'Coverage':>9}  {'Issues'}")
    print("-"*65)

    total_found = 0
    total_events = 0
    issues = []

    for concept, pairs in PHRASE_BANK.items():
        n = len(pairs)
        found = 0
        placeholders = 0
        for i in range(n):
            path = get_image_path(concept, i)
            if os.path.exists(path):
                found += 1
                if os.path.getsize(path) // 1024 < 5:
                    placeholders += 1

        total_found  += found
        total_events += n
        pct = found / n * 100
        issue_str = ""
        if placeholders:
            issue_str += f"⚠ {placeholders} placeholder(s)"
        if found < n:
            issue_str += f"  ✗ {n-found} missing"

        marker = "←" if concept in POLYSEMOUS else ""
        print(f"  {concept:<12} {found:>5} {n:>5}   {pct:>6.0f}%   {issue_str} {marker}")

        if issue_str:
            issues.append((concept, issue_str))

    print("-"*65)
    print(f"  {'TOTAL':<12} {total_found:>5} {total_events:>5}   {total_found/total_events*100:>6.0f}%")
    print()

    if issues:
        print(f"Concepts with issues ({len(issues)}):")
        for c, s in issues:
            print(f"  {c}: {s}")
    else:
        print("No issues found.")

    print()
    print("← = polysemous concept (most important to verify visually)")
    print("⚠  = likely grey placeholder (Wikimedia failed, no Wikipedia fallback)")
    print()
    print("To visually inspect a concept:")
    print("  python check_phrase_images.py run")
    print("  python check_phrase_images.py water strike light")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("concepts", nargs="*", help="Concept(s) to inspect visually")
    parser.add_argument("--all-polysemous", action="store_true",
                        help="Show all polysemous concepts")
    parser.add_argument("--report", action="store_true",
                        help="Text-only coverage report, no display")
    args = parser.parse_args()

    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory not found: {IMAGE_DIR}")
        print("Run extract_phrase_level.py first.")
        return

    if args.report:
        text_report()
        return

    if args.all_polysemous:
        for concept in POLYSEMOUS:
            check_concept(concept, show=True)
        return

    if args.concepts:
        for concept in args.concepts:
            check_concept(concept, show=True)
        return

    # Default: text report
    text_report()


if __name__ == "__main__":
    main()
