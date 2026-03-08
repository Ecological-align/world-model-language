"""
Concept list for RSA alignment experiment.

Organized into categories that stress-test different aspects of
world model vs LLM representational alignment:

- PHYSICAL: things a robot encounters and manipulates
  (world model should have strong signal here)
- ACTIONS: dynamic events with causal structure
  (world model's home turf — temporal/predictive)
- SPATIAL: relational geometry
  (should be strong in WM, possibly weak in LLM)
- ABSTRACT: concepts with no direct sensory grounding
  (LLM's home turf — may diverge from WM)
- SOCIAL: agent-relative concepts
  (interesting middle ground)

The divergence pattern across categories is itself a result.
If WM/LLM agree on PHYSICAL but diverge on ABSTRACT, that's
informative about what embodiment adds to representation.
"""

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

# Flat list preserving category order for matrix indexing
ALL_CONCEPTS = []
CONCEPT_CATEGORIES = {}  # concept -> category label
CATEGORY_BOUNDARIES = {}  # category -> (start_idx, end_idx)

idx = 0
for category, items in CONCEPTS.items():
    CATEGORY_BOUNDARIES[category] = (idx, idx + len(items))
    for item in items:
        ALL_CONCEPTS.append(item)
        CONCEPT_CATEGORIES[item] = category
        idx += 1

N_CONCEPTS = len(ALL_CONCEPTS)

if __name__ == "__main__":
    print(f"Total concepts: {N_CONCEPTS}")
    for cat, items in CONCEPTS.items():
        print(f"  {cat:12s}: {len(items)} concepts")
    print(f"\nCategory boundaries (for matrix block visualization):")
    for cat, (s, e) in CATEGORY_BOUNDARIES.items():
        print(f"  {cat:12s}: [{s}:{e}]")
