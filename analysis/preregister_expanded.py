"""
Pre-registration for expanded concept set (50 concepts).

Run this BEFORE extract_expanded.py and any experiments on the new concepts.
Saves predictions to lm_output/preregistration_expanded.json with timestamp.

Ranking logic:
  - WordNet noun senses: proxy for polysemy (more senses = harder)
  - Sensorimotor rating: 1-5, based on Lancaster Sensorimotor Norms where available,
    manually estimated where not (marked with *)
  - Combined score = wordnet_senses + (sm_rating * 4)
  - Higher score = predicted harder to align

Prediction: concepts in the top tercile (score >= threshold) will require
more contrastive pressure to align and will show lower agreement at lambda=0.1.
At lambda=0.5 we predict the top 5 will be the most likely to fail in any seed.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from datetime import datetime

# ── Original 17 concepts (already run, included for completeness) ─────────────

ORIGINAL = [
    # (concept, wordnet_noun_senses, sensorimotor_rating, notes)
    ("apple",     5,  4, "original"),
    ("chair",     4,  4, "original"),
    ("water",     8,  5, "original"),
    ("fire",      9,  5, "original"),
    ("stone",    13,  4, "original"),
    ("rope",      5,  4, "original"),
    ("door",      5,  3, "original"),
    ("container", 3,  3, "original"),
    ("shadow",    6,  3, "original"),
    ("mirror",    4,  3, "original"),
    ("knife",     4,  5, "original"),
    ("wheel",     5,  4, "original"),
    ("hand",     14,  5, "original - pre-registered hard"),
    ("wall",      7,  3, "original"),
    ("hole",      8,  3, "original - pre-registered hard"),
    ("bridge",    7,  3, "original"),
    ("ladder",    4,  4, "original"),
]

# ── New concepts: HIGH polysemy / HIGH sensorimotor (predicted hardest) ───────
# These should pattern like hand/hole — need most contrastive pressure

HIGH_HIGH = [
    # (concept, wordnet_noun_senses, sensorimotor_rating, notes)
    ("spring",   11,  5, "new-hard: coil/season/water source/to jump — high polysemy + haptic"),
    ("bark",      7,  5, "new-hard: tree bark/dog bark/boat — tactile + auditory"),
    ("wave",      9,  5, "new-hard: ocean/electromagnetic/gesture — physical + abstract"),
    ("charge",   10,  4, "new-hard: electrical/attack/fee/criminal — highly polysemous"),
    ("field",    12,  3, "new-hard: grass field/magnetic/academic — very high polysemy"),
    ("light",    13,  4, "new-hard: electromagnetic/weight/ignite — highest polysemy category"),
    ("strike",    9,  5, "new-hard: hit/labor/bowling/lightning — high action grounding"),
    ("press",     9,  4, "new-hard: printing press/to push/media — action + tool + institution"),
    ("shoot",     8,  5, "new-hard: projectile/plant shoot/photo — high action grounding"),
    ("run",      12,  5, "new-hard: locomotion/river run/stocking run — very high polysemy"),
]

# ── New concepts: LOW polysemy / HIGH sensorimotor (predicted medium) ─────────
# Clear physical affordances, few alternative meanings

LOW_SM_HIGH = [
    ("hammer",     4,  5, "new-medium: tool with clear haptic affordance, few meanings"),
    ("scissors",   3,  5, "new-medium: tool, clear haptic affordance"),
    ("bowl",       5,  4, "new-medium: container with clear shape, limited polysemy"),
    ("bucket",     4,  4, "new-medium: container, clear physical form"),
    ("bench",      5,  3, "new-medium: seating/workbench — moderate polysemy"),
    ("fence",      5,  3, "new-medium: barrier/to fence — some polysemy"),
    ("needle",     6,  5, "new-medium: sewing/compass/to needle — moderate polysemy, high haptic"),
    ("drum",       5,  5, "new-medium: instrument/container/to drum — moderate polysemy, high motor"),
    ("clock",      4,  3, "new-medium: timepiece, low sensorimotor, few meanings"),
    ("telescope",  2,  3, "new-easy: scientific instrument, very low polysemy"),
]

# ── New concepts: LOW polysemy / LOW sensorimotor (predicted easiest) ─────────
# Abstract-ish physical concepts with few alternate meanings

LOW_LOW = [
    ("cloud",      5,  3, "new-easy: weather/cloud computing — moderate but visual"),
    ("sand",       4,  4, "new-easy: material, few meanings, strong tactile"),
    ("ice",        5,  4, "new-easy: frozen water/ice rink/to ice — moderate"),
    ("feather",    4,  4, "new-easy: bird feather, low polysemy, tactile"),
    ("leaf",       6,  3, "new-medium: plant leaf/to leaf through — some polysemy"),
    ("thread",     6,  4, "new-medium: sewing/conversation thread — moderate polysemy"),
    ("glass",      7,  4, "new-medium: material/drinking glass/lens — moderate polysemy"),
    ("coin",       4,  2, "new-easy: money, low sensorimotor, few meanings"),
    ("shelf",      3,  3, "new-easy: furniture, very low polysemy"),
    ("pipe",       7,  4, "new-medium: tube/smoking pipe/to pipe — moderate polysemy"),
    ("net",        8,  4, "new-medium: fishing/internet/sports — moderate polysemy"),
    ("chain",      7,  4, "new-medium: links/sequence/to chain — moderate polysemy"),
    ("mirror",     4,  3, "original — included in original set"),  # already in original, skip
]

# Remove mirror (duplicate) from LOW_LOW
LOW_LOW = [c for c in LOW_LOW if c[0] != "mirror"]

# ── Combine all new concepts ──────────────────────────────────────────────────

NEW_CONCEPTS = HIGH_HIGH + LOW_SM_HIGH + LOW_LOW

# ── Deduplicate against originals ────────────────────────────────────────────

original_names = {c[0] for c in ORIGINAL}
NEW_CONCEPTS   = [c for c in NEW_CONCEPTS if c[0] not in original_names]

# ── Score and rank ALL concepts ──────────────────────────────────────────────

def combined_score(wordnet_senses, sm_rating):
    return wordnet_senses + (sm_rating * 4)

ALL_CONCEPTS = ORIGINAL + NEW_CONCEPTS

ranked = sorted(
    ALL_CONCEPTS,
    key=lambda c: combined_score(c[1], c[2]),
    reverse=True
)

# ── Predictions ───────────────────────────────────────────────────────────────

# Top 8 predicted to require most alignment pressure / most likely to fail at low lambda
TOP_N = 8
predicted_hard  = [c[0] for c in ranked[:TOP_N]]
predicted_easy  = [c[0] for c in ranked[-8:]]

# Specific directional predictions for lambda=0.1 vs lambda=0.5:
# At lambda=0.1: top 8 will show meaningfully lower agreement than bottom 8
# At lambda=0.5: at least 3 of top 8 will fail in at least 1/5 seeds
# Generalization: top-tercile concepts will show lower held-out agreement than bottom-tercile

print("=" * 65)
print("EXPANDED PRE-REGISTRATION: 50-CONCEPT SET")
print(f"Registered: {datetime.now().isoformat()}")
print("=" * 65)
print(f"\nTotal concepts: {len(ALL_CONCEPTS)}")
print(f"  Original: {len(ORIGINAL)}")
print(f"  New:      {len(NEW_CONCEPTS)}")

print(f"\nTop {TOP_N} predicted hardest (most pressure needed):")
for i, c in enumerate(ranked[:TOP_N]):
    score = combined_score(c[1], c[2])
    print(f"  {i+1}. {c[0]:15s}  WN={c[1]:2d}  SM={c[2]}  score={score:2d}  [{c[3]}]")

print(f"\nBottom 8 predicted easiest:")
for i, c in enumerate(ranked[-8:]):
    score = combined_score(c[1], c[2])
    print(f"  {i+1}. {c[0]:15s}  WN={c[1]:2d}  SM={c[2]}  score={score:2d}  [{c[3]}]")

print("\nFull ranking:")
for i, c in enumerate(ranked):
    score = combined_score(c[1], c[2])
    tag = " ← PREDICTED HARD" if c[0] in predicted_hard else ""
    tag = " ← PREDICTED EASY" if c[0] in predicted_easy else tag
    print(f"  {i+1:2d}. {c[0]:15s}  WN={c[1]:2d}  SM={c[2]}  score={score:2d}{tag}")

# ── Save ──────────────────────────────────────────────────────────────────────

output = {
    "registered_at": datetime.now().isoformat(),
    "description": "Pre-registration for expanded 50-concept alignment experiment",
    "scoring": {
        "formula": "wordnet_noun_senses + (sensorimotor_rating * 4)",
        "wordnet_source": "WordNet 3.1 noun synsets (manual count)",
        "sensorimotor_source": "Lancaster Sensorimotor Norms where available, manual estimate (*) otherwise",
    },
    "predictions": {
        "primary": f"Top {TOP_N} concepts by combined score will require more contrastive pressure",
        "lambda_0.1": f"Top {TOP_N} will show lower cross-modal agreement than bottom 8 at lambda=0.1",
        "lambda_0.5": "At least 3 of top 8 will fail (disagree) in at least 1 of 5 seeds at lambda=0.5",
        "generalization": "Top-tercile concepts will show lower held-out agreement than bottom-tercile in generalization test",
    },
    "predicted_hard": predicted_hard,
    "predicted_easy": predicted_easy,
    "all_concepts_ranked": [
        {
            "rank": i + 1,
            "concept": c[0],
            "wordnet_senses": c[1],
            "sensorimotor_rating": c[2],
            "combined_score": combined_score(c[1], c[2]),
            "notes": c[3],
            "predicted": "hard" if c[0] in predicted_hard else ("easy" if c[0] in predicted_easy else "medium"),
        }
        for i, c in enumerate(ranked)
    ],
    "new_concept_list": [c[0] for c in NEW_CONCEPTS],
    "original_concept_list": [c[0] for c in ORIGINAL],
    "full_concept_list": [c[0] for c in ALL_CONCEPTS],
}

import os
os.makedirs("lm_output", exist_ok=True)
outpath = "lm_output/preregistration_expanded.json"
with open(outpath, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nPre-registration saved to {outpath}")
print("Run extract_expanded.py NEXT — do not run experiments before that file is saved.")
