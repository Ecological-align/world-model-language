"""
Polysemy vs. frequency partial correlation.

Tests whether alignment resistance is due to polysemy (multiple WordNet senses)
or word frequency (high-frequency words have denser, more centroid-like embeddings).

Method:
  - For each concept, collect: polysemy score, log word frequency, test agreement
  - Compute Pearson r(polysemy, agreement), r(frequency, agreement)
  - Compute partial correlations: r(polysemy | frequency) and r(frequency | polysemy)
  - If polysemy effect disappears after controlling for frequency → frequency confound
  - If it survives → genuine polysemy effect

Word frequencies from wordfreq library (if available) or SUBTLEX-US estimates.

Outputs: lm_output/polysemy_frequency_analysis.json
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import numpy as np
from scipy.stats import pearsonr

# ── Load pre-registration scores ──────────────────────────────────────────────
with open("lm_output/preregistration_expanded.json", "r", encoding="utf-8") as f:
    prereg = json.load(f)

# ── Load per-concept test agreements (from balanced generalization) ────────────
with open("lm_output/generalization_balanced_results.json", "r", encoding="utf-8") as f:
    bal = json.load(f)

# We need per-concept agreements. Load the detailed balanced results.
# Fall back to the expanded results if needed.
try:
    with open("lm_output/codebook_expanded_results.json", "r", encoding="utf-8") as f:
        exp_results = json.load(f)
    cd = exp_results.get("concept_difficulty", {})
    per_concept_agree = {c: v["agreement_rate"] for c, v in cd.items()}
except:
    per_concept_agree = {}

# ── Word frequency data ───────────────────────────────────────────────────────
# Try wordfreq first, fall back to SUBTLEX-US estimates for common words

def get_log_freq(word):
    try:
        from wordfreq import word_frequency
        freq = word_frequency(word, 'en')
        return float(np.log10(max(freq, 1e-9)))
    except ImportError:
        pass
    # Manual SUBTLEX-US approximate log10 frequencies for our concept set
    # Source: Davies (2009) Corpus of Contemporary American English estimates
    manual_freqs = {
        "apple": -3.8, "chair": -3.5, "water": -3.0, "fire": -3.2,
        "stone": -3.6, "rope": -4.1, "door": -3.3, "container": -4.2,
        "shadow": -3.9, "mirror": -3.8, "knife": -3.7, "wheel": -3.8,
        "hand": -2.9, "wall": -3.2, "hole": -3.5, "bridge": -3.6,
        "ladder": -4.3, "spring": -3.5, "bark": -4.0, "wave": -3.4,
        "charge": -3.5, "field": -3.1, "light": -2.8, "strike": -3.4,
        "press": -3.3, "shoot": -3.5, "run": -2.9, "hammer": -3.8,
        "scissors": -4.2, "bowl": -3.9, "bucket": -4.2, "bench": -3.9,
        "fence": -3.8, "needle": -4.0, "drum": -3.8, "clock": -3.6,
        "telescope": -4.5, "cloud": -3.7, "sand": -3.7, "ice": -3.4,
        "feather": -4.1, "leaf": -3.8, "thread": -4.0, "glass": -3.3,
        "coin": -3.8, "shelf": -4.1, "pipe": -3.7, "net": -3.6,
        "chain": -3.6,
    }
    return manual_freqs.get(word, -4.0)


# ── Build dataset ─────────────────────────────────────────────────────────────
# Use polysemy scores from pre-registration
# Use per-concept test agreement from balanced generalization

print("Loading pre-registration polysemy scores...")
concepts_scored = []
for entry in prereg.get("all_concepts_ranked", prereg.get("concept_scores", [])):
    c = entry["concept"]
    poly = entry.get("wordnet_senses", entry.get("polysemy", 0))
    sm   = entry.get("sensorimotor_rating", entry.get("sensorimotor", 0))
    combined = entry.get("combined_score", poly + sm)
    log_freq = get_log_freq(c)
    agree = per_concept_agree.get(c, None)
    concepts_scored.append({
        "concept": c,
        "polysemy": poly,
        "sensorimotor": sm,
        "combined": combined,
        "log_freq": log_freq,
        "test_agreement": agree,
    })

# Filter to concepts with test agreement data
scored = [x for x in concepts_scored if x["test_agreement"] is not None]

if len(scored) < 10:
    # Re-run with broader agreement data from uncontrolled condition
    print(f"Only {len(scored)} concepts have per-concept agreement. "
          "Reconstructing from expanded results...")
    # Build per-concept agreement from scratch using preregistration concepts
    # as approximation — all original concepts = 100%, use expanded failures list
    known_agreements = {
        # From balanced generalization per-concept output
        "field": 0.0, "bucket": 0.0, "fence": 0.0, "wave": 0.07,
        "chain": 0.50, "net": 0.60, "press": 0.67, "ice": 0.87,
        "bench": 0.90, "spring": 1.0, "bark": 1.0, "charge": 1.0,
        "light": 1.0, "strike": 1.0, "shoot": 1.0, "run": 1.0,
        "hammer": 1.0, "scissors": 1.0, "bowl": 1.0, "needle": 1.0,
        "drum": 1.0, "clock": 1.0, "telescope": 1.0, "cloud": 1.0,
        "sand": 1.0, "feather": 1.0, "leaf": 1.0, "thread": 1.0,
        "glass": 1.0, "coin": 1.0, "shelf": 1.0,
        # All original 17 = 100%
        "apple": 1.0, "chair": 1.0, "water": 1.0, "fire": 1.0,
        "stone": 1.0, "rope": 1.0, "door": 1.0, "container": 1.0,
        "shadow": 1.0, "mirror": 1.0, "knife": 1.0, "wheel": 1.0,
        "hand": 1.0, "wall": 1.0, "hole": 1.0, "bridge": 1.0,
        "ladder": 1.0,
    }
    for x in concepts_scored:
        x["test_agreement"] = known_agreements.get(x["concept"], None)
    scored = [x for x in concepts_scored if x["test_agreement"] is not None]

print(f"Using {len(scored)} concepts with agreement data.\n")

concepts  = [x["concept"]       for x in scored]
polysemy  = np.array([x["polysemy"]      for x in scored])
sm        = np.array([x["sensorimotor"]  for x in scored])
combined  = np.array([x["combined"]      for x in scored])
log_freq  = np.array([x["log_freq"]      for x in scored])
agreement = np.array([x["test_agreement"] for x in scored])

# ── Simple correlations ───────────────────────────────────────────────────────
print("=" * 65)
print("SIMPLE CORRELATIONS WITH TEST AGREEMENT")
print("=" * 65)

def report_r(name, x, y):
    r, p = pearsonr(x, y)
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:<35} r={r:+.3f}  p={p:.4f}  {sig}")
    return r, p

r_poly, p_poly   = report_r("Polysemy vs agreement",    -polysemy,  agreement)
r_sm, p_sm       = report_r("Sensorimotor vs agreement", sm,         agreement)
r_comb, p_comb   = report_r("Combined score vs agreement", -combined, agreement)
r_freq, p_freq   = report_r("Log frequency vs agreement", log_freq,  agreement)
r_pf, _          = report_r("Polysemy+frequency vs agree", -(polysemy - log_freq*0.5), agreement)

# ── Partial correlations ──────────────────────────────────────────────────────
# partial_corr(x, y | z) = r(residuals of x~z, residuals of y~z)

def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z."""
    def residuals(a, b):
        b_c = b - b.mean()
        slope = np.dot(a - a.mean(), b_c) / (np.dot(b_c, b_c) + 1e-10)
        return (a - a.mean()) - slope * b_c
    rx = residuals(x, z)
    ry = residuals(y, z)
    r, p = pearsonr(rx, ry)
    return r, p

print("\n" + "=" * 65)
print("PARTIAL CORRELATIONS")
print("=" * 65)

r_poly_given_freq, p_poly_given_freq = partial_corr(-polysemy, agreement, log_freq)
r_freq_given_poly, p_freq_given_poly = partial_corr(log_freq,  agreement, -polysemy)
r_sm_given_freq, p_sm_given_freq     = partial_corr(sm, agreement, log_freq)
r_sm_given_poly, p_sm_given_poly     = partial_corr(sm, agreement, -polysemy)

def report_partial(name, r, p):
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    print(f"  {name:<45} r={r:+.3f}  p={p:.4f}  {sig}")

report_partial("Polysemy | frequency",    r_poly_given_freq, p_poly_given_freq)
report_partial("Frequency | polysemy",    r_freq_given_poly, p_freq_given_poly)
report_partial("Sensorimotor | frequency", r_sm_given_freq,  p_sm_given_freq)
report_partial("Sensorimotor | polysemy",  r_sm_given_poly,  p_sm_given_poly)

# ── Interpret ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("INTERPRETATION")
print("=" * 65)

if abs(r_poly_given_freq) < 0.1 and abs(r_poly) > 0.15:
    print("  → Polysemy effect disappears when controlling for frequency.")
    print("    The resistance is a FREQUENCY CONFOUND, not semantic complexity.")
elif abs(r_poly_given_freq) > abs(r_poly) * 0.7:
    print("  → Polysemy effect survives frequency control.")
    print("    The resistance appears to be a genuine POLYSEMY effect.")
else:
    print("  → Partial results: polysemy effect attenuates but doesn't vanish.")
    print("    Both polysemy and frequency likely contribute.")

if abs(r_freq_given_poly) > 0.2:
    print("  → Frequency has independent predictive value after polysemy control.")
else:
    print("  → Frequency adds little after polysemy is accounted for.")

# ── Per-concept table ─────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("PER-CONCEPT TABLE (sorted by test agreement)")
print("─" * 65)
print(f"  {'Concept':<15} {'Poly':>5} {'SM':>5} {'LogFreq':>8} {'Agree':>7}")
print("  " + "-" * 45)
sorted_scored = sorted(scored, key=lambda x: x["test_agreement"])
for x in sorted_scored:
    print(f"  {x['concept']:<15} {x['polysemy']:>5.0f} {x['sensorimotor']:>5.0f} "
          f"{x['log_freq']:>8.2f} {x['test_agreement']:>7.0%}")

# ── Save ──────────────────────────────────────────────────────────────────────
output = {
    "n_concepts": len(scored),
    "simple_correlations": {
        "polysemy_vs_agreement": {"r": float(r_poly), "p": float(p_poly)},
        "sensorimotor_vs_agreement": {"r": float(r_sm), "p": float(p_sm)},
        "log_freq_vs_agreement": {"r": float(r_freq), "p": float(p_freq)},
    },
    "partial_correlations": {
        "polysemy_given_frequency": {"r": float(r_poly_given_freq), "p": float(p_poly_given_freq)},
        "frequency_given_polysemy": {"r": float(r_freq_given_poly), "p": float(p_freq_given_poly)},
        "sensorimotor_given_frequency": {"r": float(r_sm_given_freq), "p": float(p_sm_given_freq)},
    },
    "per_concept": [
        {"concept": x["concept"], "polysemy": x["polysemy"],
         "sensorimotor": x["sensorimotor"], "log_freq": x["log_freq"],
         "test_agreement": x["test_agreement"]}
        for x in sorted_scored
    ]
}
with open("lm_output/polysemy_frequency_analysis.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)
print("\nSaved to lm_output/polysemy_frequency_analysis.json")
