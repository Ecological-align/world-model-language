"""
Pre-registration: Which physical concepts will resist cross-modal alignment?
----------------------------------------------------------------------------
Prediction: concepts with high polysemy OR high sensorimotor grounding
will resist alignment between V-JEPA 2 (video world model) and language models.

- High polysemy: many WordNet senses -> language model spreads meaning across
  contexts that don't map to a single visual prototype.
- High sensorimotor grounding: meaning depends on bodily/physical experience
  that language captures poorly -> world model has richer representation than
  language can express.

This file must be run BEFORE examining codebook results.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
from pathlib import Path

# Lazy import to keep script self-contained
import nltk
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import wordnet as wn

output_dir = Path("lm_output")

CONCEPTS = [
    "apple", "chair", "water", "fire", "stone", "rope", "door", "container",
    "shadow", "mirror", "knife", "wheel", "hand", "wall", "hole", "bridge",
    "ladder"
]

# ── 1. WordNet sense counts ──────────────────────────────────────────────────

wordnet_senses = {}
for c in CONCEPTS:
    synsets = wn.synsets(c, pos=wn.NOUN)
    wordnet_senses[c] = len(synsets)

# ── 2. Sensorimotor proxy ratings (1-5) ─────────────────────────────────────
#
# Rating criteria: How much does the concept's core meaning depend on
# physical/bodily/sensory experience vs. being definable abstractly?
#
# 5 = meaning is almost entirely sensorimotor (you can't understand it
#     without physical experience)
# 4 = strongly embodied, some abstract extension
# 3 = mixed: clear physical referent but significant abstract/functional use
# 2 = primarily functional/relational, physical form is secondary
# 1 = can be fully understood from abstract definition

sensorimotor = {
    "apple": (4, "Strongly multisensory: color, texture, taste, weight, smell. "
                 "Minor abstract use (Apple Inc, Adam's apple)."),
    "chair": (3, "Clear visual form but meaning is primarily functional (something to sit on). "
                 "Metaphorical uses: chair of department, electric chair."),
    "water": (5, "Fundamentally sensory: temperature, flow, wetness, taste, sound. "
                 "Understanding water requires physical experience with liquid states."),
    "fire": (5, "Intensely multisensory: heat, light, movement, crackling sound, smell. "
                "Also highly polysemous (firing someone, gunfire, passion)."),
    "stone": (4, "Strong tactile/weight grounding: hardness, coldness, heaviness. "
                 "Many metaphorical extensions (stoned, stepping stone, cornerstone)."),
    "rope": (4, "Defined by tactile/functional properties: flexibility, tension, texture. "
                "Limited abstract extension (know the ropes, end of one's rope)."),
    "door": (3, "Physical referent is clear but meaning is heavily functional/spatial: "
                "boundary, passage, access. Metaphorical: door of opportunity."),
    "container": (2, "Primarily a functional/relational category: 'thing that contains'. "
                     "Physical form varies enormously (box, jar, bag, room)."),
    "shadow": (3, "Visual phenomenon requiring light/body interaction. "
                  "But extensive abstract use: shadow of doubt, shadow government, shadowy figure."),
    "mirror": (3, "Strong visual/reflective grounding but also deeply metaphorical: "
                  "mirror neurons, mirror of society. Functional definition dominates."),
    "knife": (4, "Strong sensorimotor: sharpness, cutting action, grip, weight. "
                 "Meaning is inseparable from the cutting affordance."),
    "wheel": (3, "Visual form is distinctive but meaning extends heavily: steering wheel, "
                 "wheel of fortune, reinvent the wheel, potter's wheel."),
    "hand": (5, "The paradigm case of embodied meaning. Manipulation, gesture, touch, grip. "
                "Massively polysemous precisely because bodily experience is so rich: "
                "hand of cards, helping hand, secondhand, handiwork."),
    "wall": (3, "Clear physical referent (vertical barrier) but meaning is largely functional: "
                "separation, obstacle. Firewall, wall of sound, hit a wall."),
    "hole": (3, "Defined by absence of material — a spatial/relational concept. "
                "Visual but its meaning is about what's NOT there. Polysemous: plot hole, hole up."),
    "bridge": (3, "Clear physical structure but heavily metaphorical: bridge between ideas, "
                  "bridge in music, dental bridge, bridge of nose. Functional meaning dominates."),
    "ladder": (3, "Visual form is distinctive, climbing action is embodied. "
                  "Metaphorical: corporate ladder, social ladder. Moderate extension."),
}

# ── 3. Combined scoring and ranking ─────────────────────────────────────────
#
# Normalization: rank each dimension separately, then sum ranks.
# Higher combined rank = more likely to resist alignment.

# Rank by WordNet senses (more senses = higher rank = harder to align)
sorted_by_senses = sorted(CONCEPTS, key=lambda c: wordnet_senses[c])
sense_rank = {c: i + 1 for i, c in enumerate(sorted_by_senses)}

# Rank by sensorimotor rating (higher = harder to align, because
# the world model captures more than language can)
sorted_by_sm = sorted(CONCEPTS, key=lambda c: sensorimotor[c][0])
sm_rank = {c: i + 1 for i, c in enumerate(sorted_by_sm)}

# Combined score = sum of ranks (higher = more likely to resist)
combined = {c: sense_rank[c] + sm_rank[c] for c in CONCEPTS}
sorted_combined = sorted(CONCEPTS, key=lambda c: combined[c], reverse=True)

# Top 3 predicted to fail
predicted_to_fail = set(sorted_combined[:3])

# ── 4. Print table ──────────────────────────────────────────────────────────

print("=" * 95)
print("PRE-REGISTRATION: PREDICTED CROSS-MODAL ALIGNMENT RESISTANCE")
print("=" * 95)
print()
print("Hypothesis: concepts with high polysemy OR high sensorimotor grounding")
print("will resist alignment between V-JEPA 2 and language models.")
print()
print(f"{'Rank':>4s}  {'Concept':12s} | {'WN senses':>9s} | {'SM rating':>9s} | "
      f"{'sense_rk':>8s} | {'sm_rk':>5s} | {'combined':>8s} | fail?")
print(f"{'':4s}  {'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*8}-+-{'-'*5}-+-{'-'*8}-+------")

for rank_i, c in enumerate(sorted_combined, 1):
    wn_s = wordnet_senses[c]
    sm_r = sensorimotor[c][0]
    s_rk = sense_rank[c]
    sm_rk_v = sm_rank[c]
    comb = combined[c]
    fail = "<<< PREDICTED" if c in predicted_to_fail else ""
    print(f"  {rank_i:2d}  {c:12s} | {wn_s:9d} | {sm_r:9d} | {s_rk:8d} | {sm_rk_v:5d} | {comb:8d} | {fail}")

print()
print("Sensorimotor rating justifications:")
print("-" * 70)
for c in sorted_combined:
    sm_r, reason = sensorimotor[c]
    print(f"  {c:12s} ({sm_r}/5): {reason}")

# ── 5. Save to JSON ─────────────────────────────────────────────────────────

results = {
    "hypothesis": (
        "Concepts with high polysemy (many WordNet noun senses) or high "
        "sensorimotor grounding (meaning depends on physical experience) "
        "will resist cross-modal alignment between V-JEPA 2 and language models."
    ),
    "methodology": {
        "polysemy": "WordNet noun synset count via nltk.corpus.wordnet",
        "sensorimotor": "Manual 1-5 proxy rating based on how much core meaning "
                        "depends on physical/bodily experience vs abstract definition",
        "ranking": "Sum of within-dimension ranks (sense_rank + sm_rank), "
                   "higher = more likely to resist alignment",
        "prediction": "Top 3 by combined score predicted to fail cross-modal alignment"
    },
    "concepts": {}
}

for rank_i, c in enumerate(sorted_combined, 1):
    results["concepts"][c] = {
        "wordnet_senses": wordnet_senses[c],
        "sensorimotor_proxy": sensorimotor[c][0],
        "sensorimotor_justification": sensorimotor[c][1],
        "sense_rank": sense_rank[c],
        "sm_rank": sm_rank[c],
        "combined_score": combined[c],
        "combined_rank": rank_i,
        "predicted_to_fail": c in predicted_to_fail
    }

results["predicted_failures"] = sorted(list(predicted_to_fail))
results["predicted_easy"] = [sorted_combined[-1], sorted_combined[-2], sorted_combined[-3]]

with open(output_dir / "polysemy_preregistration.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to lm_output/polysemy_preregistration.json")
print(f"\nPREDICTED TO FAIL: {', '.join(sorted(predicted_to_fail))}")
print(f"PREDICTED EASY:    {', '.join(results['predicted_easy'])}")
