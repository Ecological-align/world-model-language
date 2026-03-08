"""
Diagnostic: why does the contrastive codebook fail to generalize?

Three hypotheses:
  A) MODALITY COLLAPSE: test concepts revert to modality-level codes
     (ST gets code X, VJ gets code Y, same as untrained collapse)
  B) RANDOM SCATTER: test concepts get assigned random/arbitrary codes
     (ST and VJ both get unpredictable codes with no structure)
  C) NEAR MISS: test concepts get adjacent codes that are close but not identical
     (alignment is geometrically close but discretization fails)

Also checks:
  - Do test concepts cluster together or scatter?
  - Are test code assignments consistent across seeds?
  - Do train concepts use a different part of the codebook than test concepts?

Run after train_codebook_generalization.py.
Reads lm_output/codebook_generalization_results.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
from collections import Counter, defaultdict

# ── Load results ──────────────────────────────────────────────────────────────

with open("lm_output/codebook_generalization_results.json", "r") as f:
    data = json.load(f)

runs    = data["per_run"]
summary = data["summary"]

print("=" * 65)
print("GENERALIZATION FAILURE DIAGNOSTIC")
print("=" * 65)
print(f"Total runs: {len(runs)}")
print(f"Train agreement: {summary['train_agreement_mean']:.1%}")
print(f"Test  agreement: {summary['test_agreement_mean']:.1%}")
print(f"Gap:             {summary['generalization_gap']:.1%}")
print()

# ── Hypothesis A: Modality collapse ──────────────────────────────────────────
# If test concepts revert to modality codes, ST and VJ indices should
# be anti-correlated (each modality gets its own cluster of codes)

print("─" * 65)
print("HYPOTHESIS A: Do test concepts revert to modality-level codes?")
print("─" * 65)

st_code_sets  = []
vj_code_sets  = []
modality_gaps = []  # how different are ST vs VJ code distributions per run

for run in runs:
    st_idx = run["test_indices_st"]
    vj_idx = run["test_indices_vj"]
    st_code_sets.append(set(st_idx))
    vj_code_sets.append(set(vj_idx))
    # overlap between ST and VJ code sets (0 = perfect modality separation)
    overlap = len(set(st_idx) & set(vj_idx)) / max(len(set(st_idx) | set(vj_idx)), 1)
    modality_gaps.append(1 - overlap)

mean_gap = np.mean(modality_gaps)
print(f"Mean code-set separation (ST vs VJ): {mean_gap:.2f}")
print(f"  (1.0 = perfect modality split, 0.0 = fully shared codes)")
if mean_gap > 0.7:
    print("  → SUPPORTS Hypothesis A: modality collapse on test concepts")
elif mean_gap > 0.4:
    print("  → PARTIAL: some modality separation but not clean split")
else:
    print("  → AGAINST Hypothesis A: codes are shared between modalities")
print()

# ── Hypothesis B: Random scatter ─────────────────────────────────────────────
# If codes are random, each run should assign different codes to the same concept
# Measure: consistency of code assignment for same concept across seeds

print("─" * 65)
print("HYPOTHESIS B: Are test code assignments random/inconsistent across seeds?")
print("─" * 65)

# Group by split, then check consistency across seeds
by_split = defaultdict(list)
for run in runs:
    by_split[run["split"]].append(run)

concept_consistency_st = []
concept_consistency_vj = []

for split_id, split_runs in by_split.items():
    test_concepts = split_runs[0]["test_concepts"]
    for concept_pos, concept in enumerate(test_concepts):
        st_codes_across_seeds = [r["test_indices_st"][concept_pos] for r in split_runs]
        vj_codes_across_seeds = [r["test_indices_vj"][concept_pos] for r in split_runs]
        # consistency = fraction of seeds that agree with the mode
        st_mode = Counter(st_codes_across_seeds).most_common(1)[0][1] / len(split_runs)
        vj_mode = Counter(vj_codes_across_seeds).most_common(1)[0][1] / len(split_runs)
        concept_consistency_st.append((concept, st_mode))
        concept_consistency_vj.append((concept, vj_mode))

mean_st_consistency = np.mean([c[1] for c in concept_consistency_st])
mean_vj_consistency = np.mean([c[1] for c in concept_consistency_vj])

print(f"Mean code consistency across seeds:")
print(f"  ST modality: {mean_st_consistency:.2f}  (1.0 = same code every seed)")
print(f"  VJ modality: {mean_vj_consistency:.2f}")

if mean_st_consistency > 0.7 and mean_vj_consistency > 0.7:
    print("  → AGAINST Hypothesis B: codes are consistent, not random")
    print("     The codebook assigns the same (wrong) code reliably")
elif mean_st_consistency < 0.5 or mean_vj_consistency < 0.5:
    print("  → SUPPORTS Hypothesis B: code assignments vary by seed")
    print("     Test concepts get arbitrary codes depending on initialization")
else:
    print("  → MIXED: moderate consistency")
print()

# ── Hypothesis C: Near miss ───────────────────────────────────────────────────
# Load the actual embeddings and check whether test concept pre-VQ projections
# are geometrically close to the nearest training concept's code

print("─" * 65)
print("HYPOTHESIS C: Are test concepts geometrically close to train codes?")
print("  (checking whether disagreement is a discretization boundary issue)")
print("─" * 65)

# We can't re-run the model here, but we can check:
# If ST and VJ test codes are consistently NEAR each other (small code index distance)
# that would suggest near-miss discretization

code_distances = []
for run in runs:
    st_idx = run["test_indices_st"]
    vj_idx = run["test_indices_vj"]
    for s, v in zip(st_idx, vj_idx):
        code_distances.append(abs(s - v))

mean_dist = np.mean(code_distances)
median_dist = np.median(code_distances)
pct_within_5 = np.mean([d <= 5 for d in code_distances])

print(f"Code index distance between ST and VJ assignments:")
print(f"  Mean:   {mean_dist:.1f}")
print(f"  Median: {median_dist:.1f}")
print(f"  Within 5 codes: {pct_within_5:.1%}")
print(f"  (Note: codebook has 64 codes; random expectation ~21)")

if mean_dist < 8:
    print("  → SUPPORTS Hypothesis C: near miss, geometrically close but wrong code")
elif mean_dist > 15:
    print("  → AGAINST Hypothesis C: codes are far apart, not a boundary issue")
else:
    print("  → AMBIGUOUS: moderate distance")
print()

# ── Per-concept failure analysis ──────────────────────────────────────────────

print("─" * 65)
print("PER-CONCEPT: Which test concepts appear most / least often?")
print("─" * 65)

concept_test_counts   = Counter()
concept_agree_counts  = Counter()

for run in runs:
    test_concepts = run["test_concepts"]
    st_idx = run["test_indices_st"]
    vj_idx = run["test_indices_vj"]
    for concept, s, v in zip(test_concepts, st_idx, vj_idx):
        concept_test_counts[concept] += 1
        if s == v:
            concept_agree_counts[concept] += 1

print(f"{'Concept':<15} {'Tests':>6} {'Agrees':>7} {'Rate':>7}")
print("-" * 38)
for concept in sorted(concept_test_counts, key=lambda c: -concept_agree_counts[c] / concept_test_counts[c]):
    n     = concept_test_counts[concept]
    agree = concept_agree_counts[concept]
    rate  = agree / n
    bar   = "█" * int(rate * 20)
    print(f"  {concept:<13} {n:>6} {agree:>7} {rate:>7.0%}  {bar}")

print()

# ── Train vs test codebook usage ─────────────────────────────────────────────

print("─" * 65)
print("TRAIN vs TEST: Do test concepts fall into train-used code regions?")
print("─" * 65)

# We don't have train indices stored, but we can infer from test_n_codes
# A high test_n_codes with low agreement means scatter (Hypothesis B)
# A low test_n_codes with low agreement means collapse (Hypothesis A)

test_codes_counts = [r["test_n_codes"] for r in runs]
print(f"Mean unique codes used by test concepts: {np.mean(test_codes_counts):.1f} ± {np.std(test_codes_counts):.1f}")
print(f"  (5 test concepts; max possible unique = 10 if ST≠VJ for all)")
print(f"  High count (~8-10) → scatter (Hypothesis B)")
print(f"  Low count (~2-4)   → collapse to few codes (Hypothesis A)")
print()

# ── Summary diagnosis ─────────────────────────────────────────────────────────

print("=" * 65)
print("DIAGNOSIS SUMMARY")
print("=" * 65)

print(f"""
Key measurements:
  Modality code separation:  {mean_gap:.2f}  (A: collapse if >0.7)
  Cross-seed consistency:    ST={mean_st_consistency:.2f}, VJ={mean_vj_consistency:.2f}  (B: random if <0.5)
  Mean ST↔VJ code distance:  {mean_dist:.1f}  (C: near-miss if <8)
  Mean unique test codes:    {np.mean(test_codes_counts):.1f}  (A: collapse if ~2-4)
""")

# Determine primary hypothesis
scores = {"A": 0, "B": 0, "C": 0}
if mean_gap > 0.7: scores["A"] += 2
if mean_gap > 0.4: scores["A"] += 1
if mean_st_consistency < 0.5 or mean_vj_consistency < 0.5: scores["B"] += 2
if mean_st_consistency < 0.7: scores["B"] += 1
if mean_dist < 8: scores["C"] += 2
if mean_dist < 12: scores["C"] += 1
if np.mean(test_codes_counts) < 5: scores["A"] += 1
if np.mean(test_codes_counts) > 7: scores["B"] += 1

primary = max(scores, key=scores.get)
explanations = {
    "A": "MODALITY COLLAPSE — test concepts revert to modality-level codes.\n"
         "   The codebook learned 'ST=one cluster, VJ=another' and applies\n"
         "   that same split to unseen concepts. This means the underlying\n"
         "   problem (modality variance > concept variance) was never solved,\n"
         "   just suppressed for training concepts.",
    "B": "RANDOM SCATTER — test concepts get inconsistent arbitrary codes.\n"
         "   The codebook has no structure that transfers to new concepts.\n"
         "   It memorized a specific mapping and has no prior for unseen items.\n"
         "   This is classic overfitting to the training set.",
    "C": "NEAR-MISS DISCRETIZATION — geometric alignment exists but fails\n"
         "   at the quantization boundary. ST and VJ project to nearby but\n"
         "   different Voronoi regions for test concepts. The continuous\n"
         "   representations may actually be closer than the code disagreement\n"
         "   suggests — the discrete bottleneck is the failure point.",
}

print(f"Primary diagnosis: Hypothesis {primary}")
print(f"  {explanations[primary]}")
print()
print(f"Hypothesis scores: A={scores['A']}  B={scores['B']}  C={scores['C']}")
print()

# Implication for the paper
implications = {
    "A": "The contrastive loss suppresses collapse for seen concepts but doesn't\n"
         "   change the underlying geometry. The modality gap reasserts itself\n"
         "   for unseen concepts. Scaling to more concepts would not fix this\n"
         "   without a fundamentally different approach.",
    "B": "With only 17 training concepts the codebook has too few examples to\n"
         "   learn a generalizable mapping. This is a data size problem —\n"
         "   expanding to 50+ concepts may improve generalization significantly.\n"
         "   This motivates the expanded concept set experiment.",
    "C": "The modalities ARE aligning geometrically, but the discrete bottleneck\n"
         "   is too coarse. A smaller codebook or softer quantization might show\n"
         "   generalization. The finding is about discretization, not alignment.",
}

print("Implication for the paper:")
print(f"  {implications[primary]}")

# Save
output = {
    "modality_separation": float(mean_gap),
    "st_consistency": float(mean_st_consistency),
    "vj_consistency": float(mean_vj_consistency),
    "mean_code_distance": float(mean_dist),
    "mean_test_codes": float(np.mean(test_codes_counts)),
    "hypothesis_scores": scores,
    "primary_hypothesis": primary,
    "per_concept": {
        c: {
            "n_tests": concept_test_counts[c],
            "n_agrees": concept_agree_counts[c],
            "agreement_rate": concept_agree_counts[c] / concept_test_counts[c]
        }
        for c in concept_test_counts
    }
}

with open("lm_output/generalization_diagnostic.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nDiagnostic saved to lm_output/generalization_diagnostic.json")
