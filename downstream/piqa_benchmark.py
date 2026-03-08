"""
PIQA Physical Reasoning Benchmark Pipeline
===========================================

Tests whether injecting V-JEPA 2 / CLIP world-model embeddings as
nearest-neighbour context improves Mistral 7B accuracy on PIQA questions
that involve our 49 physical concepts.

Two conditions per matched question:
  A) Mistral alone       — standard zero-shot prompt
  B) Mistral + WM context — same prompt prefixed with V-JEPA 2 nearest-
                            neighbour concept descriptions

Outputs:
  lm_output/piqa_results.json   — per-question results + summary stats
  lm_output/piqa_summary.txt    — human-readable report

Requirements:
  pip install datasets   (HuggingFace datasets library)

Run:
  python piqa_benchmark.py

Estimated runtime: 30–60 min on RTX 5090 (depends on # matched questions)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, re, time
import numpy as np
import torch
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CONCEPT_INDEX_PATH = "lm_output/concept_index.json"
VJEPA_HIDDENS_PATH = "lm_output/vjepa2_hiddens_expanded.npy"
CLIP_HIDDENS_PATH  = "lm_output/clip_hiddens_expanded.npy"
RESULTS_PATH       = "lm_output/piqa_results.json"
SUMMARY_PATH       = "lm_output/piqa_summary.txt"

# How many nearest-neighbour concepts to inject as context
N_NEIGHBOURS = 3

# Minimum questions to proceed (warn if below)
MIN_QUESTIONS = 100

# Set to an integer to test on a subset first (e.g. 50), None for full run
DEBUG_LIMIT = None

# ── Physical concept descriptions for world-model context ─────────────────────
# These are the "translations" from V-JEPA 2 nearest-neighbour space to
# human-readable physical descriptions injected into Mistral's prompt.
# Describes what the world model knows about each concept: dynamics, affordances.
CONCEPT_DESCRIPTIONS = {
    "apple":     "a round fruit that rolls, falls under gravity, bruises on impact, floats in water",
    "chair":     "a rigid support structure that bears weight, can tip if unbalanced, slides on smooth floors",
    "water":     "a liquid that flows downhill, fills containers, evaporates when heated, conducts electricity",
    "fire":      "a dynamic process that spreads to combustible material, produces heat and smoke, requires oxygen",
    "stone":     "a dense solid that sinks in water, resists compression, rolls on slopes, conducts heat slowly",
    "rope":      "a flexible tension member that transmits pulling force, can be knotted, frays under abrasion",
    "door":      "a hinged panel that rotates around a fixed edge to open and close an opening",
    "container": "an enclosed volume that holds liquids or solids, can be filled, emptied, sealed or opened",
    "shadow":    "a region of reduced light formed when an opaque object blocks a light source",
    "mirror":    "a reflective surface that reverses left-right and produces virtual images at equal distance",
    "knife":     "a rigid blade that cuts by applying concentrated pressure along a thin edge",
    "wheel":     "a circular rigid body that reduces friction by converting sliding to rolling motion",
    "hand":      "a grasping appendage that can apply force, pinch, squeeze, push, pull, and manipulate objects",
    "wall":      "a vertical planar barrier that blocks passage, supports weight from above, reflects sound",
    "hole":      "an absence in a surface through which objects can pass or fall; depth determines what fits",
    "bridge":    "a spanning structure that transfers load across a gap to supports on either side",
    "ladder":    "a vertical climbing structure with rungs; leans against a surface, can slide if base slips",
    "spring":    "an elastic coil that stores energy under compression or extension and releases it on release",
    "bark":      "the rough outer protective layer of a tree; fibrous, waterproof, burns readily",
    "wave":      "a periodic oscillation of a water surface that transfers energy without net water transport",
    "charge":    "an electrical property that causes attraction or repulsion; builds on insulators, conducts through metals",
    "field":     "a large open area of ground; flat terrain allows objects to roll and slide with low friction",
    "light":     "electromagnetic radiation that travels in straight lines, reflects off surfaces, refracts through transparent media",
    "strike":    "a sharp impulsive contact that transfers momentum; harder objects break softer ones on impact",
    "press":     "application of compressive force over a surface area; flattens deformable materials",
    "shoot":     "a young growing plant stem; bends toward light, fragile under lateral force",
    "run":       "rapid bipedal locomotion requiring balance and coordination; momentum builds with speed",
    "hammer":    "a heavy head on a handle that concentrates impact force; drives nails, breaks brittle materials",
    "scissors":  "two crossing blades that shear material by applying opposing forces near a pivot point",
    "bowl":      "a concave vessel that holds liquids and loose solids; tips when centre of mass exceeds rim",
    "bucket":    "a cylindrical container with a handle; holds liquids, can be carried, tips when unbalanced",
    "bench":     "a long horizontal seating surface supported at two ends; supports distributed weight",
    "fence":     "a vertical barrier of posts and rails; blocks movement, can be climbed, bends under lateral force",
    "needle":    "a thin sharp point that penetrates surfaces by concentrating force on a tiny area",
    "drum":      "a cylindrical membrane that vibrates when struck; resonates at a fundamental frequency",
    "clock":     "a timekeeping mechanism with hands or digits; requires power, sensitive to magnetic fields",
    "telescope": "an optical instrument that collects and focuses light to magnify distant objects",
    "cloud":     "a suspension of water droplets in air; forms when air cools below dew point, produces rain",
    "sand":      "a granular material that flows like a liquid under gravity, compresses under weight, abrades surfaces",
    "ice":       "frozen water that is slippery due to a thin liquid layer under pressure, floats on liquid water",
    "feather":   "a light aerofoil structure with low density; drifts in air currents, provides insulation",
    "leaf":      "a flat thin plant structure; catches light, bends in wind, becomes brittle when dry",
    "thread":    "a long thin fibre under tension that transmits pulling force; breaks under sharp bending",
    "glass":     "a brittle transparent solid that transmits light, shatters on sharp impact, cuts under pressure",
    "coin":      "a small dense disc that rolls on its edge, stacks, sinks in water, conducts heat quickly",
    "shelf":     "a horizontal flat surface fixed to a wall; supports objects placed on it, deflects under excess load",
    "pipe":      "a hollow cylinder that channels fluids under pressure; bursts if pressure exceeds wall strength",
    "net":       "a mesh of intersecting fibres with regular gaps; traps objects larger than the mesh size",
    "chain":     "a series of rigid links that transmits tension but not compression; can wrap around objects",
}

ALL_CONCEPTS = list(CONCEPT_DESCRIPTIONS.keys())

# Concept synonyms / variants for matching PIQA questions
CONCEPT_VARIANTS = {
    "apple":     ["apple", "apples"],
    "chair":     ["chair", "chairs", "seat", "stool"],
    "water":     ["water", "liquid", "fluid"],
    "fire":      ["fire", "flame", "flames", "burning", "burn", "ignite"],
    "stone":     ["stone", "stones", "rock", "rocks", "pebble"],
    "rope":      ["rope", "ropes", "cord", "twine", "string"],
    "door":      ["door", "doors"],
    "container": ["container", "containers", "box", "boxes", "jar", "jars", "bin"],
    "shadow":    ["shadow", "shadows"],
    "mirror":    ["mirror", "mirrors"],
    "knife":     ["knife", "knives", "blade", "blades"],
    "wheel":     ["wheel", "wheels", "tire", "tires"],
    "hand":      ["hand", "hands", "finger", "fingers", "grip"],
    "wall":      ["wall", "walls"],
    "hole":      ["hole", "holes", "gap", "opening"],
    "bridge":    ["bridge", "bridges"],
    "ladder":    ["ladder", "ladders"],
    "spring":    ["spring", "springs", "coil", "coils"],
    "bark":      ["bark", "tree bark"],
    "wave":      ["wave", "waves", "surf"],
    "charge":    ["charge", "electric charge", "electricity", "static"],
    "field":     ["field", "fields"],
    "light":     ["light", "sunlight", "lamp", "torch", "flashlight"],
    "strike":    ["strike", "hit", "impact", "blow"],
    "press":     ["press", "pressing", "compress", "squeeze"],
    "shoot":     ["shoot", "sprout", "seedling"],
    "run":       ["run", "running", "jog", "sprint"],
    "hammer":    ["hammer", "hammers", "mallet"],
    "scissors":  ["scissors", "shears", "snips"],
    "bowl":      ["bowl", "bowls", "basin"],
    "bucket":    ["bucket", "buckets", "pail"],
    "bench":     ["bench", "benches"],
    "fence":     ["fence", "fences", "railing"],
    "needle":    ["needle", "needles", "pin", "pins"],
    "drum":      ["drum", "drums"],
    "clock":     ["clock", "clocks", "watch", "timer"],
    "telescope": ["telescope", "telescopes", "binoculars"],
    "cloud":     ["cloud", "clouds"],
    "sand":      ["sand", "gravel", "grit"],
    "ice":       ["ice", "frost", "frozen"],
    "feather":   ["feather", "feathers"],
    "leaf":      ["leaf", "leaves"],
    "thread":    ["thread", "threads", "sewing thread"],
    "glass":     ["glass", "glasses", "glassware"],
    "coin":      ["coin", "coins"],
    "shelf":     ["shelf", "shelves"],
    "pipe":      ["pipe", "pipes", "tube", "tubes"],
    "net":       ["net", "nets", "mesh"],
    "chain":     ["chain", "chains"],
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load PIQA dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_piqa():
    from datasets import load_dataset
    print("Loading PIQA validation set...")
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    questions = []
    for item in ds:
        questions.append({
            "goal":    item["goal"],
            "sol1":    item["sol1"],
            "sol2":    item["sol2"],
            "label":   item["label"],   # 0 = sol1 correct, 1 = sol2 correct
        })
    print(f"Loaded {len(questions)} PIQA validation questions")
    return questions


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Match questions to concepts
# ══════════════════════════════════════════════════════════════════════════════

def find_concepts_in_text(text):
    """Return list of concepts mentioned in text (word boundary match)."""
    text_lower = text.lower()
    found = []
    for concept, variants in CONCEPT_VARIANTS.items():
        for v in variants:
            pattern = r'\b' + re.escape(v) + r'\b'
            if re.search(pattern, text_lower):
                if concept not in found:
                    found.append(concept)
                break
    return found


def match_questions(questions):
    """Tag each question with matched concepts."""
    matched = []
    unmatched = []
    for q in questions:
        full_text = f"{q['goal']} {q['sol1']} {q['sol2']}"
        concepts = find_concepts_in_text(full_text)
        if concepts:
            matched.append({**q, "concepts": concepts})
        else:
            unmatched.append(q)
    return matched, unmatched


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Build concept embedding index for nearest-neighbour lookup
# ══════════════════════════════════════════════════════════════════════════════

def load_concept_embeddings():
    with open(CONCEPT_INDEX_PATH) as f:
        idx = json.load(f)
    all_concepts = idx["all_concepts"]

    vjepa = np.load(VJEPA_HIDDENS_PATH)   # [49, 1024]
    clip  = np.load(CLIP_HIDDENS_PATH)    # [49, 768]

    # L2-normalise
    vjepa_n = vjepa / (np.linalg.norm(vjepa, axis=1, keepdims=True) + 1e-8)
    clip_n  = clip  / (np.linalg.norm(clip,  axis=1, keepdims=True) + 1e-8)

    return all_concepts, vjepa_n, clip_n


def get_wm_context(query_concepts, all_concepts, vjepa_n, clip_n):
    """
    For each query concept, find N_NEIGHBOURS nearest neighbours in
    V-JEPA 2 space and return their physical descriptions as context.
    """
    lines = []
    for qc in query_concepts:
        if qc not in all_concepts:
            continue
        qi = all_concepts.index(qc)
        qvec = vjepa_n[qi]

        # Cosine similarity to all concepts
        sims = vjepa_n @ qvec  # [49]
        sims[qi] = -1.0        # exclude self

        top_k = np.argsort(sims)[::-1][:N_NEIGHBOURS]
        neighbours = [all_concepts[j] for j in top_k]

        desc = CONCEPT_DESCRIPTIONS.get(qc, qc)
        nn_descs = [CONCEPT_DESCRIPTIONS.get(n, n) for n in neighbours]

        lines.append(f"- {qc}: {desc}")
        lines.append(f"  [physically similar to: {', '.join(neighbours)}]")
        for n, nd in zip(neighbours, nn_descs):
            lines.append(f"    • {n}: {nd}")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Mistral inference
# ══════════════════════════════════════════════════════════════════════════════

def load_mistral():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Mistral 7B on {device}...")
    tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-Instruct-v0.2",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return tok, model, device


def build_prompt(goal, sol1, sol2, wm_context=None):
    """Build a zero-shot multiple-choice prompt."""
    if wm_context:
        context_block = (
            "Physical world knowledge relevant to this question "
            "(from a visual world model trained on video dynamics):\n"
            f"{wm_context}\n\n"
        )
    else:
        context_block = ""

    prompt = (
        f"{context_block}"
        f"Question: {goal}\n\n"
        f"Option A: {sol1}\n"
        f"Option B: {sol2}\n\n"
        "Which option is the better solution? Answer with just 'A' or 'B'."
    )
    return f"[INST] {prompt} [/INST]"


def get_answer(tok, model, device, prompt, max_new_tokens=8):
    """Run Mistral and extract A or B from output."""
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    # Decode only the new tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = tok.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract A or B
    for char in response.upper():
        if char in ("A", "B"):
            return char
    # Fallback: look for "Option A" or "Option B"
    if "A" in response.upper():
        return "A"
    return "B"


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Run benchmark
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(matched_qs, tok, model, device,
                  all_concepts, vjepa_n, clip_n):

    results = []
    n = len(matched_qs)

    for i, q in enumerate(matched_qs):
        goal  = q["goal"]
        sol1  = q["sol1"]
        sol2  = q["sol2"]
        label = q["label"]    # 0 = A correct, 1 = B correct
        correct_letter = "A" if label == 0 else "B"
        concepts = q["concepts"]

        # ── Condition A: Mistral alone ────────────────────────────────────
        prompt_alone = build_prompt(goal, sol1, sol2, wm_context=None)
        ans_alone    = get_answer(tok, model, device, prompt_alone)
        correct_alone = (ans_alone == correct_letter)

        # ── Condition B: Mistral + WM context ────────────────────────────
        wm_ctx       = get_wm_context(concepts, all_concepts, vjepa_n, clip_n)
        prompt_wm    = build_prompt(goal, sol1, sol2, wm_context=wm_ctx)
        ans_wm       = get_answer(tok, model, device, prompt_wm)
        correct_wm   = (ans_wm == correct_letter)

        results.append({
            "idx":           i,
            "goal":          goal,
            "sol1":          sol1,
            "sol2":          sol2,
            "correct":       correct_letter,
            "concepts":      concepts,
            "ans_alone":     ans_alone,
            "ans_wm":        ans_wm,
            "correct_alone": correct_alone,
            "correct_wm":    correct_wm,
            "flip":          "improved" if (not correct_alone and correct_wm)
                             else "degraded" if (correct_alone and not correct_wm)
                             else "same",
        })

        # Progress
        if (i + 1) % 25 == 0 or i == n - 1:
            done = i + 1
            acc_a = sum(r["correct_alone"] for r in results) / done * 100
            acc_b = sum(r["correct_wm"]    for r in results) / done * 100
            print(f"  [{done:4d}/{n}]  Alone: {acc_a:.1f}%  WM: {acc_b:.1f}%  "
                  f"Δ={acc_b-acc_a:+.1f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Analyse and report
# ══════════════════════════════════════════════════════════════════════════════

def analyse(results):
    n = len(results)
    acc_alone = sum(r["correct_alone"] for r in results) / n
    acc_wm    = sum(r["correct_wm"]    for r in results) / n
    delta     = acc_wm - acc_alone

    improved  = [r for r in results if r["flip"] == "improved"]
    degraded  = [r for r in results if r["flip"] == "degraded"]
    same      = [r for r in results if r["flip"] == "same"]

    # Per-concept breakdown
    concept_stats = {}
    for concept in ALL_CONCEPTS:
        qs = [r for r in results if concept in r["concepts"]]
        if not qs:
            continue
        concept_stats[concept] = {
            "n":         len(qs),
            "acc_alone": sum(r["correct_alone"] for r in qs) / len(qs),
            "acc_wm":    sum(r["correct_wm"]    for r in qs) / len(qs),
            "delta":     sum(r["correct_wm"]    for r in qs) / len(qs) -
                         sum(r["correct_alone"] for r in qs) / len(qs),
        }

    # McNemar's test (paired) — tests if improvements ≠ degradations
    from scipy.stats import binomtest
    # Under H0: P(improve) = P(degrade) = 0.5
    n_change = len(improved) + len(degraded)
    if n_change > 0:
        p_mcnemar = binomtest(len(improved), n_change, 0.5,
                              alternative="greater").pvalue
    else:
        p_mcnemar = 1.0

    return {
        "n_total":       n,
        "acc_alone":     acc_alone,
        "acc_wm":        acc_wm,
        "delta":         delta,
        "n_improved":    len(improved),
        "n_degraded":    len(degraded),
        "n_same":        len(same),
        "p_mcnemar":     p_mcnemar,
        "concept_stats": concept_stats,
        "examples_improved": [
            {"goal": r["goal"], "concepts": r["concepts"]}
            for r in improved[:5]
        ],
        "examples_degraded": [
            {"goal": r["goal"], "concepts": r["concepts"]}
            for r in degraded[:5]
        ],
    }


def write_report(summary, results):
    lines = []
    lines.append("=" * 65)
    lines.append("PIQA BENCHMARK: WORLD MODEL CONTEXT INJECTION")
    lines.append("=" * 65)
    lines.append(f"\nTotal matched questions: {summary['n_total']}")
    lines.append(f"\nOVERALL ACCURACY")
    lines.append(f"  Mistral alone:           {summary['acc_alone']*100:.1f}%")
    lines.append(f"  Mistral + WM context:    {summary['acc_wm']*100:.1f}%")
    lines.append(f"  Delta:                   {summary['delta']*100:+.1f}%")
    lines.append(f"\nQUESTION-LEVEL CHANGES")
    lines.append(f"  Improved (wrong→right):  {summary['n_improved']}")
    lines.append(f"  Degraded (right→wrong):  {summary['n_degraded']}")
    lines.append(f"  Unchanged:               {summary['n_same']}")
    sig = "p < 0.05 ✓" if summary["p_mcnemar"] < 0.05 else f"p = {summary['p_mcnemar']:.3f} n.s."
    lines.append(f"  McNemar test (improvements > degradations): {sig}")

    lines.append(f"\nPER-CONCEPT BREAKDOWN")
    lines.append(f"  {'Concept':<14} {'N':>4}  {'Alone':>6}  {'WM':>6}  {'Δ':>6}")
    lines.append(f"  {'-'*45}")
    cs = summary["concept_stats"]
    for concept in sorted(cs, key=lambda c: -abs(cs[c]["delta"])):
        s = cs[concept]
        lines.append(f"  {concept:<14} {s['n']:>4}  "
                     f"{s['acc_alone']*100:>5.1f}%  "
                     f"{s['acc_wm']*100:>5.1f}%  "
                     f"{s['delta']*100:>+5.1f}%")

    lines.append(f"\nEXAMPLE IMPROVEMENTS (wrong→right with WM context)")
    for ex in summary["examples_improved"]:
        lines.append(f"  [{', '.join(ex['concepts'])}] {ex['goal'][:80]}")

    lines.append(f"\nEXAMPLE DEGRADATIONS (right→wrong with WM context)")
    for ex in summary["examples_degraded"]:
        lines.append(f"  [{', '.join(ex['concepts'])}] {ex['goal'][:80]}")

    lines.append(f"\nINTERPRETATION")
    delta_pct = summary["delta"] * 100
    if delta_pct > 2 and summary["p_mcnemar"] < 0.05:
        lines.append(f"  ✓ World model context significantly improves accuracy (+{delta_pct:.1f}%).")
        lines.append(f"    The V-JEPA 2 physical dynamics descriptions help Mistral reason")
        lines.append(f"    about physical outcomes on PIQA questions involving known concepts.")
    elif delta_pct > 0:
        lines.append(f"  ~ Positive trend (+{delta_pct:.1f}%) but not significant.")
        lines.append(f"    Direction is consistent with the hypothesis; more questions needed.")
    elif delta_pct < -2 and summary["p_mcnemar"] < 0.05:
        lines.append(f"  ✗ World model context significantly hurts accuracy ({delta_pct:.1f}%).")
        lines.append(f"    The context may be introducing noise or misleading the model.")
    else:
        lines.append(f"  ~ No meaningful effect ({delta_pct:+.1f}%). Context neither helps nor hurts.")

    report = "\n".join(lines)
    print("\n" + report)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("PIQA WORLD MODEL CONTEXT INJECTION BENCHMARK")
    print("=" * 65)

    # Load PIQA
    questions = load_piqa()

    # Match to concepts
    print("\nMatching questions to concepts...")
    matched, unmatched = match_questions(questions)
    print(f"Matched:   {len(matched)} questions ({len(matched)/len(questions)*100:.1f}%)")
    print(f"Unmatched: {len(unmatched)} questions")

    # Concept frequency
    from collections import Counter
    concept_counts = Counter(c for q in matched for c in q["concepts"])
    print("\nTop concepts in matched questions:")
    for concept, count in concept_counts.most_common(15):
        print(f"  {concept:<14} {count:>4} questions")

    if len(matched) < MIN_QUESTIONS:
        print(f"\nWARNING: Only {len(matched)} matched questions (threshold: {MIN_QUESTIONS})")
        print("Consider expanding the concept set before running the full benchmark.")
        resp = input("Continue anyway? [y/N] ").strip().lower()
        if resp != "y":
            return

    # Apply debug limit
    if DEBUG_LIMIT:
        matched = matched[:DEBUG_LIMIT]
        print(f"\nDEBUG MODE: running on first {DEBUG_LIMIT} questions")

    # Load concept embeddings
    print("\nLoading concept embeddings...")
    all_concepts, vjepa_n, clip_n = load_concept_embeddings()

    # Load Mistral
    tok, model, device = load_mistral()

    # Run benchmark
    print(f"\nRunning benchmark on {len(matched)} questions...")
    print("(Showing running accuracy every 25 questions)\n")
    t0 = time.time()
    results = run_benchmark(matched, tok, model, device,
                            all_concepts, vjepa_n, clip_n)
    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed/60:.1f} min")

    # Analyse
    summary = analyse(results)

    # Save
    output = {
        "summary":  summary,
        "results":  results,
        "metadata": {
            "n_total_piqa":    len(questions),
            "n_matched":       len(matched),
            "n_neighbours":    N_NEIGHBOURS,
            "debug_limit":     DEBUG_LIMIT,
            "elapsed_min":     round(elapsed / 60, 1),
        }
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Write report
    write_report(summary, results)
    print(f"Report saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
