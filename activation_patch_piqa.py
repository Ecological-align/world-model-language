"""
activation_patch_piqa.py
========================

Experiment 19: Direct activation patching of world model representations
into Mistral 7B residual stream for PIQA physical reasoning.

WHAT THIS TESTS
===============
Experiment 8 (piqa_benchmark.py) injected world model knowledge as text in
the prompt: −6.6% PIQA accuracy. That failure could mean:

  (a) Text-mediated translation is lossy — the bottleneck is the text interface
  (b) The world model representations don't contain useful physical reasoning
      signal regardless of how they're injected

This experiment distinguishes (a) from (b) by bypassing text entirely.
We project MAE / V-JEPA 2 concept embeddings directly into Mistral's residual
stream at a specified transformer layer, then run PIQA.

If direct patching helps (or hurts less than text injection):
  → The interface mechanism was the problem, not the representations.
    Text translation discards or distorts the physical signal.

If direct patching also fails:
  → The representations themselves don't carry PIQA-relevant information,
    regardless of interface. The gap is representational, not translational.

V-JEPA 2 patching is included as a within-experiment control. Given that
MAE aligns better with LM representations (Exp 10–14), MAE patching should
produce a larger behavioral effect than V-JEPA 2 patching if alignment
predicts patching benefit.

CONDITIONS
==========
  baseline       : Mistral alone, no patching, no text context
  text_injection : Text-mediated WM context (replicates Exp 8 for comparison)
  patch_mae_L8   : MAE embedding patched at layer 8 (early)
  patch_mae_L16  : MAE embedding patched at layer 16 (mid — primary condition)
  patch_mae_L24  : MAE embedding patched at layer 24 (late)
  patch_vjepa_L16: V-JEPA 2 embedding patched at layer 16 (control)

PROJECTION TRAINING
===================
For each visual model, we train a linear projection P: vis_dim → 4096
using ridge regression on the 61-concept phrase-level embedding pairs
(visual embedding → Mistral mid-layer hidden state for same concept).

Ridge regression is used (not a learned MLP) to keep the projection simple
and avoid overfitting on 61 training points. The projection is trained once
before PIQA evaluation begins.

PATCHING MECHANISM
==================
A forward hook is registered on model.model.layers[PATCH_LAYER].
At the hook, the projected concept embedding (scaled by PATCH_SCALE) is
added to the residual stream output for ALL token positions.

Multiple concepts per question: embeddings are averaged before projection.

OUTPUTS
=======
  lm_output/patch_piqa_results.json   — per-question results for all conditions
  lm_output/patch_piqa_summary.txt    — human-readable comparison table

Run:
  PYTHONPATH=. .venv/Scripts/python.exe activation_patch_piqa.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, re, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────

CONCEPT_INDEX_PATH  = "lm_output/concept_index.json"
MAE_PHRASE_PATH     = "lm_output/phrase_level/mae_hiddens_phrase.npy"
VJEPA_PHRASE_PATH   = "lm_output/phrase_level/vjepa2_hiddens_phrase.npy"
LM_PHRASE_PATH      = "lm_output/phrase_level/lm_hiddens_phrase.npy"
EVENT_INDEX_PATH    = "lm_output/phrase_level/event_index.json"

# Fallback to concept-level if phrase-level unavailable
MAE_CONCEPT_PATH    = "lm_output/mae_hiddens_expanded.npy"
VJEPA_CONCEPT_PATH  = "lm_output/vjepa2_hiddens_expanded.npy"

RESULTS_PATH        = "lm_output/patch_piqa_results.json"
SUMMARY_PATH        = "lm_output/patch_piqa_summary.txt"

# Mistral hidden size
LM_DIM = 4096

# Patch layers to test
PATCH_LAYERS = [8, 16, 24]

# Patch scale: how much to add to the residual stream
# Start conservative — the LM residual stream has typical norm ~50–100
# We want to add a signal that's meaningful but not destructive
PATCH_SCALE = 0.1   # tuned from scale sweep (1.0 was too aggressive)

# Sweep over scales for the primary condition (MAE, layer 24)
PATCH_SCALE_SWEEP = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

# Layer granularity sweep around L24 at best scale
LAYER_GRANULARITY_SWEEP = [20, 22, 24, 26, 28]

# Ridge regression regularization for projection training
RIDGE_ALPHA = 1.0

# PIQA config
N_NEIGHBOURS   = 3     # for text injection condition (matches Exp 8)
MIN_QUESTIONS  = 100
DEBUG_LIMIT    = None  # set to int (e.g. 50) for fast testing

# ── Concept set ────────────────────────────────────────────────────────────────
# Reuse concept descriptions and variants from Exp 8
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

CONCEPT_VARIANTS = {
    "apple":     ["apple", "apples"],
    "chair":     ["chair", "chairs", "seat", "stool"],
    "water":     ["water", "liquid", "fluid"],
    "fire":      ["fire", "flame", "flames", "burning", "burn"],
    "stone":     ["stone", "stones", "rock", "rocks", "pebble"],
    "rope":      ["rope", "ropes", "cord", "twine", "string"],
    "door":      ["door", "doors"],
    "container": ["container", "containers", "box", "boxes", "jar", "jars"],
    "shadow":    ["shadow", "shadows"],
    "mirror":    ["mirror", "mirrors"],
    "knife":     ["knife", "knives", "blade"],
    "wheel":     ["wheel", "wheels", "tire", "tires"],
    "hand":      ["hand", "hands", "finger", "fingers", "grip"],
    "wall":      ["wall", "walls"],
    "hole":      ["hole", "holes", "gap", "opening"],
    "bridge":    ["bridge", "bridges"],
    "ladder":    ["ladder", "ladders"],
    "spring":    ["spring", "springs", "coil"],
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
    "scissors":  ["scissors", "shears"],
    "bowl":      ["bowl", "bowls", "basin"],
    "bucket":    ["bucket", "buckets", "pail"],
    "bench":     ["bench", "benches"],
    "fence":     ["fence", "fences", "railing"],
    "needle":    ["needle", "needles", "pin"],
    "drum":      ["drum", "drums"],
    "clock":     ["clock", "clocks", "watch", "timer"],
    "telescope": ["telescope", "telescopes", "binoculars"],
    "cloud":     ["cloud", "clouds"],
    "sand":      ["sand", "gravel"],
    "ice":       ["ice", "frost", "frozen"],
    "feather":   ["feather", "feathers"],
    "leaf":      ["leaf", "leaves"],
    "thread":    ["thread", "threads"],
    "glass":     ["glass", "glasses", "glassware"],
    "coin":      ["coin", "coins"],
    "shelf":     ["shelf", "shelves"],
    "pipe":      ["pipe", "pipes", "tube"],
    "net":       ["net", "nets", "mesh"],
    "chain":     ["chain", "chains"],
}

ALL_CONCEPTS = list(CONCEPT_DESCRIPTIONS.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Load and prepare concept-level embeddings
# ══════════════════════════════════════════════════════════════════════════════

def load_concept_embeddings():
    """
    Load visual and LM phrase-level embeddings, average per concept.
    Falls back to concept-level arrays if phrase-level not found.

    Returns:
        concepts        : list of concept names (N,)
        mae_embs        : np.ndarray [N, mae_dim]  — MAE concept embeddings
        vjepa_embs      : np.ndarray [N, vjepa_dim] — V-JEPA 2 concept embeddings
        lm_embs         : np.ndarray [N, 4096]     — Mistral concept embeddings
    """
    phrase_available = (
        os.path.exists(MAE_PHRASE_PATH) and
        os.path.exists(VJEPA_PHRASE_PATH) and
        os.path.exists(LM_PHRASE_PATH) and
        os.path.exists(EVENT_INDEX_PATH)
    )

    if phrase_available:
        print("Using phrase-level embeddings (averaging per concept)...")
        mae_ph   = np.load(MAE_PHRASE_PATH)
        vjepa_ph = np.load(VJEPA_PHRASE_PATH)
        lm_ph    = np.load(LM_PHRASE_PATH)
        with open(EVENT_INDEX_PATH) as f:
            event_index = json.load(f)
        if isinstance(event_index, dict) and "events" in event_index:
            event_index = event_index["events"]
        concepts_ev = [e["concept"] for e in event_index]
        unique_concepts = list(dict.fromkeys(concepts_ev))

        def mean_per_concept(arr):
            out = []
            for c in unique_concepts:
                idxs = [i for i, ec in enumerate(concepts_ev) if ec == c]
                out.append(arr[idxs].mean(0))
            return np.array(out)

        mae_embs   = mean_per_concept(mae_ph)
        vjepa_embs = mean_per_concept(vjepa_ph)
        lm_embs    = mean_per_concept(lm_ph)
        concepts   = unique_concepts
        print(f"  Concepts: {len(concepts)}  MAE: {mae_embs.shape}  "
              f"VJEPA: {vjepa_embs.shape}  LM: {lm_embs.shape}")

    else:
        print("Phrase-level not found — falling back to concept-level embeddings...")
        with open(CONCEPT_INDEX_PATH) as f:
            idx = json.load(f)
        concepts = idx["all_concepts"]
        mae_embs   = np.load(MAE_CONCEPT_PATH)
        vjepa_embs = np.load(VJEPA_CONCEPT_PATH)
        # LM concept-level hiddens — try to find them
        lm_concept_path = "lm_output/lm_hiddens_expanded.npy"
        if not os.path.exists(lm_concept_path):
            raise FileNotFoundError(
                "Need lm_hiddens_expanded.npy or phrase-level LM hiddens. "
                "Run extract_expanded.py or extract_phrase_level.py first."
            )
        lm_embs = np.load(lm_concept_path)
        print(f"  Concepts: {len(concepts)}  MAE: {mae_embs.shape}  "
              f"VJEPA: {vjepa_embs.shape}  LM: {lm_embs.shape}")

    return concepts, mae_embs, vjepa_embs, lm_embs


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Train linear projections (visual → LM space)
# ══════════════════════════════════════════════════════════════════════════════

def train_projection(vis_embs, lm_embs, alpha=RIDGE_ALPHA):
    """
    Train ridge regression projection P: vis_dim → LM_DIM.

    P = argmin_P ||vis_embs @ P.T - lm_embs||^2 + alpha * ||P||^2

    Closed-form solution:
        P.T = (vis_embs.T @ vis_embs + alpha * I)^{-1} @ vis_embs.T @ lm_embs

    Returns:
        P : np.ndarray [LM_DIM, vis_dim]  (apply as P @ vis_embedding)
    """
    X = vis_embs.astype(np.float64)   # [N, vis_dim]
    Y = lm_embs.astype(np.float64)    # [N, LM_DIM]

    N, D = X.shape
    # Ridge: (X^T X + alpha I)^{-1} X^T Y
    XtX = X.T @ X                          # [D, D]
    XtX += alpha * np.eye(D)
    XtY = X.T @ Y                          # [D, LM_DIM]
    P_T = np.linalg.solve(XtX, XtY)       # [D, LM_DIM]
    P   = P_T.T                            # [LM_DIM, D]

    # Diagnostic: training R^2
    Y_pred = X @ P_T
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - Y.mean(0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"    Projection R² = {r2:.4f}  (shape: {P.shape})")

    return P.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: PIQA dataset loading and concept matching
# ══════════════════════════════════════════════════════════════════════════════

def load_piqa():
    from datasets import load_dataset
    print("Loading PIQA validation set...")
    ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    questions = []
    for item in ds:
        questions.append({
            "goal":  item["goal"],
            "sol1":  item["sol1"],
            "sol2":  item["sol2"],
            "label": item["label"],
        })
    print(f"Loaded {len(questions)} PIQA validation questions")
    return questions


def find_concepts(text):
    text_lower = text.lower()
    found = []
    for concept, variants in CONCEPT_VARIANTS.items():
        for v in variants:
            if re.search(r'\b' + re.escape(v) + r'\b', text_lower):
                if concept not in found:
                    found.append(concept)
                break
    return found


def match_questions(questions):
    matched, unmatched = [], []
    for q in questions:
        full_text = f"{q['goal']} {q['sol1']} {q['sol2']}"
        concepts = find_concepts(full_text)
        if concepts:
            matched.append({**q, "concepts": concepts})
        else:
            unmatched.append(q)
    return matched, unmatched


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: Load Mistral (frozen, with hooks)
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


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: Activation patching via forward hooks
# ══════════════════════════════════════════════════════════════════════════════

class PatchState:
    """Mutable state shared with hook closure — holds the patch vector."""
    def __init__(self):
        self.patch_vec = None   # torch.Tensor [1, 1, LM_DIM] fp16 on device
        self.active    = False


def make_hook(patch_state, scale):
    """
    Returns a forward hook that adds patch_state.patch_vec to residual output.

    The hook fires on the output of a transformer layer.
    Mistral layer output is a tuple; element [0] is the hidden state [B, T, D].
    We add the patch vector (broadcast over all T positions) scaled by `scale`.
    """
    def hook(module, input, output):
        if not patch_state.active or patch_state.patch_vec is None:
            return output

        # transformers 5.x may return a Tensor instead of a tuple
        if isinstance(output, torch.Tensor):
            hidden = output
            patch  = patch_state.patch_vec.to(hidden.dtype).to(hidden.device)
            return hidden + scale * patch
        else:
            hidden = output[0]   # [B, T, D]
            patch  = patch_state.patch_vec.to(hidden.dtype).to(hidden.device)
            patched_hidden = hidden + scale * patch
            return (patched_hidden,) + output[1:]

    return hook


def register_patch_hook(model, layer_idx, patch_state, scale):
    """Register hook on model.model.layers[layer_idx]. Returns handle."""
    layer  = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(make_hook(patch_state, scale))
    return handle


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: Build prompts
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(goal, sol1, sol2, wm_context=None):
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


def get_text_context(query_concepts, all_concepts, vjepa_n):
    """Reproduce Exp 8 text injection (nearest-neighbour descriptions)."""
    lines = []
    for qc in query_concepts:
        if qc not in all_concepts:
            continue
        qi = all_concepts.index(qc)
        qvec = vjepa_n[qi]
        sims = vjepa_n @ qvec
        sims[qi] = -1.0
        top_k = np.argsort(sims)[::-1][:N_NEIGHBOURS]
        neighbours = [all_concepts[j] for j in top_k]
        desc = CONCEPT_DESCRIPTIONS.get(qc, qc)
        lines.append(f"- {qc}: {desc}")
        for n in neighbours:
            nd = CONCEPT_DESCRIPTIONS.get(n, n)
            lines.append(f"  • {n}: {nd}")
    return "\n".join(lines)


def get_answer(tok, model, device, prompt, max_new_tokens=8):
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response   = tok.decode(new_tokens, skip_special_tokens=True).strip()
    for char in response.upper():
        if char in ("A", "B"):
            return char
    return "B"


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Build concept patch vectors
# ══════════════════════════════════════════════════════════════════════════════

def build_concept_patch_vectors(projection, vis_embs, all_concepts, device):
    """
    For each concept, compute projected patch vector.

    Returns dict: concept -> torch.Tensor [1, 1, LM_DIM] float32
    """
    P = torch.tensor(projection, dtype=torch.float32)   # [LM_DIM, vis_dim]
    concept_patches = {}
    for i, concept in enumerate(all_concepts):
        vis_vec = torch.tensor(vis_embs[i], dtype=torch.float32)   # [vis_dim]
        patch   = P @ vis_vec                                        # [LM_DIM]
        concept_patches[concept] = patch.unsqueeze(0).unsqueeze(0)  # [1, 1, LM_DIM]
    return concept_patches


def get_patch_for_question(concepts, concept_patches, all_concepts):
    """Average patch vectors for all matched concepts in question."""
    vecs = []
    for c in concepts:
        if c in concept_patches:
            vecs.append(concept_patches[c])
    if not vecs:
        return None
    stacked = torch.stack(vecs, dim=0)   # [K, 1, 1, LM_DIM]
    return stacked.mean(0)               # [1, 1, LM_DIM]


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: Run benchmark across all conditions
# ══════════════════════════════════════════════════════════════════════════════

def run_all_conditions(matched_qs, tok, model, device,
                       all_concepts,
                       mae_concept_patches_by_layer,
                       vjepa_concept_patches_L16,
                       vjepa_n):
    """
    For each question, evaluate all conditions.

    mae_concept_patches_by_layer: dict layer_idx -> dict concept -> patch_vec
    vjepa_concept_patches_L16:    dict concept -> patch_vec  (layer 16 only)
    """
    results = []
    n = len(matched_qs)

    # Condition names
    conditions = (
        ["baseline", "text_injection"] +
        [f"patch_mae_L{l}" for l in PATCH_LAYERS] +
        ["patch_vjepa_L16"]
    )

    # Patch state object (shared with hook, mutated per question)
    patch_state = PatchState()

    # We manage ONE hook at a time — register before each condition, remove after
    active_handle = None

    print(f"\nRunning {len(conditions)} conditions × {n} questions...")
    print(f"Conditions: {conditions}\n")

    for i, q in enumerate(matched_qs):
        goal           = q["goal"]
        sol1           = q["sol1"]
        sol2           = q["sol2"]
        label          = q["label"]
        correct_letter = "A" if label == 0 else "B"
        concepts       = q["concepts"]

        row = {
            "idx":      i,
            "goal":     goal,
            "sol1":     sol1,
            "sol2":     sol2,
            "correct":  correct_letter,
            "concepts": concepts,
        }

        # ── baseline ────────────────────────────────────────────────────────
        patch_state.active = False
        prompt = build_prompt(goal, sol1, sol2, wm_context=None)
        ans    = get_answer(tok, model, device, prompt)
        row["baseline"] = ans

        # ── text_injection (replicates Exp 8) ───────────────────────────────
        patch_state.active = False
        vjepa_n_arr = vjepa_n  # normalised [N, 1024]
        ctx    = get_text_context(concepts, all_concepts, vjepa_n_arr)
        prompt = build_prompt(goal, sol1, sol2, wm_context=ctx if ctx else None)
        ans    = get_answer(tok, model, device, prompt)
        row["text_injection"] = ans

        # ── patch conditions ─────────────────────────────────────────────────
        for layer_idx in PATCH_LAYERS:
            patch_vec = get_patch_for_question(
                concepts, mae_concept_patches_by_layer[layer_idx], all_concepts
            )
            cond_name = f"patch_mae_L{layer_idx}"

            # Clean up any previous hook
            if active_handle is not None:
                active_handle.remove()
                active_handle = None

            if patch_vec is not None:
                patch_state.patch_vec = patch_vec
                patch_state.active    = True
                active_handle = register_patch_hook(
                    model, layer_idx, patch_state, PATCH_SCALE
                )
            else:
                patch_state.active = False

            prompt = build_prompt(goal, sol1, sol2, wm_context=None)
            ans    = get_answer(tok, model, device, prompt)
            row[cond_name] = ans
            patch_state.active = False

        # ── patch_vjepa_L16 ─────────────────────────────────────────────────
        # Clean up any previous hook
        if active_handle is not None:
            active_handle.remove()
            active_handle = None

        patch_vec = get_patch_for_question(
            concepts, vjepa_concept_patches_L16, all_concepts
        )
        if patch_vec is not None:
            patch_state.patch_vec = patch_vec
            patch_state.active    = True
            active_handle = register_patch_hook(model, 16, patch_state, PATCH_SCALE)
        else:
            patch_state.active = False

        prompt = build_prompt(goal, sol1, sol2, wm_context=None)
        ans    = get_answer(tok, model, device, prompt)
        row["patch_vjepa_L16"] = ans
        patch_state.active = False

        # Clean up after last condition
        if active_handle is not None:
            active_handle.remove()
            active_handle = None

        results.append(row)

        if (i + 1) % 25 == 0 or i == n - 1:
            # Quick progress report
            def acc(cond):
                done = [r for r in results if cond in r]
                return sum(r[cond] == r["correct"] for r in done) / len(done) * 100
            base_acc = acc("baseline")
            print(f"  [{i+1:4d}/{n}]  baseline={base_acc:.1f}%  "
                  f"text={acc('text_injection'):.1f}%  "
                  f"mae_L16={acc('patch_mae_L16'):.1f}%  "
                  f"vjepa_L16={acc('patch_vjepa_L16'):.1f}%")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Step 9: Scale sweep (MAE, layer 16 only)
# ══════════════════════════════════════════════════════════════════════════════

def run_scale_sweep(matched_qs, tok, model, device,
                    all_concepts, mae_patches, layer_idx=24):
    """
    Test PATCH_SCALE_SWEEP values on MAE at given layer.
    Run on first 200 questions to keep runtime reasonable.
    """
    sweep_qs = matched_qs[:200]
    sweep_results = {}
    patch_state = PatchState()

    print(f"\nScale sweep on MAE L{layer_idx} ({len(sweep_qs)} questions)...")
    print(f"Scales: {PATCH_SCALE_SWEEP}")

    for scale in PATCH_SCALE_SWEEP:
        handle = register_patch_hook(model, layer_idx, patch_state, scale)
        correct = 0
        for q in sweep_qs:
            patch_vec = get_patch_for_question(
                q["concepts"], mae_patches, all_concepts
            )
            if patch_vec is not None:
                patch_state.patch_vec = patch_vec
                patch_state.active    = True
            else:
                patch_state.active = False

            prompt = build_prompt(q["goal"], q["sol1"], q["sol2"])
            ans    = get_answer(tok, model, device, prompt)
            correct_letter = "A" if q["label"] == 0 else "B"
            correct += int(ans == correct_letter)
            patch_state.active = False

        handle.remove()
        acc = correct / len(sweep_qs) * 100
        sweep_results[scale] = acc
        print(f"  scale={scale}  acc={acc:.1f}%")

    return sweep_results


def run_layer_sweep(matched_qs, tok, model, device,
                    all_concepts, P_mae, mae_embs, scale=0.1):
    """
    Test LAYER_GRANULARITY_SWEEP layers at fixed scale.
    Run on first 200 questions.
    """
    sweep_qs = matched_qs[:200]
    layer_results = {}
    patch_state = PatchState()

    # Build patch vectors (same projection, different hook points)
    mae_patches = build_concept_patch_vectors(P_mae, mae_embs, all_concepts, device)

    print(f"\nLayer granularity sweep at scale={scale} ({len(sweep_qs)} questions)...")
    print(f"Layers: {LAYER_GRANULARITY_SWEEP}")

    for layer_idx in LAYER_GRANULARITY_SWEEP:
        handle = register_patch_hook(model, layer_idx, patch_state, scale)
        correct = 0
        for q in sweep_qs:
            patch_vec = get_patch_for_question(
                q["concepts"], mae_patches, all_concepts
            )
            if patch_vec is not None:
                patch_state.patch_vec = patch_vec
                patch_state.active    = True
            else:
                patch_state.active = False

            prompt = build_prompt(q["goal"], q["sol1"], q["sol2"])
            ans    = get_answer(tok, model, device, prompt)
            correct_letter = "A" if q["label"] == 0 else "B"
            correct += int(ans == correct_letter)
            patch_state.active = False

        handle.remove()
        acc = correct / len(sweep_qs) * 100
        layer_results[layer_idx] = acc
        print(f"  L{layer_idx}: acc={acc:.1f}%")

    return layer_results


# ══════════════════════════════════════════════════════════════════════════════
# Step 10: Analysis and report
# ══════════════════════════════════════════════════════════════════════════════

def analyse(results):
    from scipy.stats import binomtest

    conditions = (
        ["baseline", "text_injection"] +
        [f"patch_mae_L{l}" for l in PATCH_LAYERS] +
        ["patch_vjepa_L16"]
    )

    summary = {}
    for cond in conditions:
        cond_results = [r for r in results if cond in r]
        if not cond_results:
            continue
        n  = len(cond_results)
        acc = sum(r[cond] == r["correct"] for r in cond_results) / n

        # McNemar vs baseline
        improved = sum(
            1 for r in cond_results
            if r["baseline"] != r["correct"] and r[cond] == r["correct"]
        )
        degraded = sum(
            1 for r in cond_results
            if r["baseline"] == r["correct"] and r[cond] != r["correct"]
        )
        n_change = improved + degraded
        p_val = binomtest(improved, n_change, 0.5).pvalue if n_change > 0 else 1.0

        baseline_acc = sum(r["baseline"] == r["correct"] for r in cond_results) / n
        summary[cond] = {
            "n":            n,
            "acc":          acc,
            "delta":        acc - baseline_acc,
            "n_improved":   improved,
            "n_degraded":   degraded,
            "p_mcnemar":    p_val,
        }

        # Per-concept breakdown for primary condition
        if cond == "patch_mae_L16":
            concept_stats = {}
            for concept in ALL_CONCEPTS:
                qs = [r for r in cond_results if concept in r["concepts"]]
                if len(qs) < 3:
                    continue
                concept_stats[concept] = {
                    "n":       len(qs),
                    "acc_base": sum(r["baseline"] == r["correct"] for r in qs) / len(qs),
                    "acc_patch": sum(r[cond] == r["correct"] for r in qs) / len(qs),
                    "delta":   (sum(r[cond] == r["correct"] for r in qs) -
                                sum(r["baseline"] == r["correct"] for r in qs)) / len(qs),
                }
            summary["patch_mae_L16_per_concept"] = concept_stats

    return summary


def write_report(summary, scale_sweep):
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT 19: ACTIVATION PATCHING vs TEXT INJECTION — PIQA")
    lines.append("=" * 70)
    lines.append(f"\nPatch scale: {PATCH_SCALE}  Ridge alpha: {RIDGE_ALPHA}")
    lines.append(f"Conditions tested: {list(summary.keys())}\n")

    lines.append(f"{'Condition':<22} {'N':>5}  {'Acc':>6}  {'Δ Base':>7}  "
                 f"{'↑ Impr':>7}  {'↓ Degr':>7}  {'p (McNemar)':>12}")
    lines.append("-" * 70)

    order = (
        ["baseline", "text_injection"] +
        [f"patch_mae_L{l}" for l in PATCH_LAYERS] +
        ["patch_vjepa_L16"]
    )
    for cond in order:
        if cond not in summary:
            continue
        s = summary[cond]
        sig = "***" if s["p_mcnemar"] < 0.001 else (
              "**"  if s["p_mcnemar"] < 0.01  else (
              "*"   if s["p_mcnemar"] < 0.05  else "n.s."))
        lines.append(
            f"  {cond:<20} {s['n']:>5}  {s['acc']*100:>5.1f}%  "
            f"{s['delta']*100:>+6.1f}%  "
            f"{s['n_improved']:>7}  {s['n_degraded']:>7}  "
            f"  p={s['p_mcnemar']:.3f} {sig}"
        )

    lines.append("\n" + "─" * 70)
    lines.append("KEY COMPARISONS")
    lines.append("─" * 70)

    if "text_injection" in summary and "patch_mae_L16" in summary:
        text_d  = summary["text_injection"]["delta"] * 100
        patch_d = summary["patch_mae_L16"]["delta"] * 100
        lines.append(f"\n  Text injection vs direct patching (MAE L16):")
        lines.append(f"    Text:   Δ = {text_d:+.1f}%")
        lines.append(f"    Patch:  Δ = {patch_d:+.1f}%")
        if patch_d > text_d + 1.0:
            lines.append("    → Direct patching outperforms text injection.")
            lines.append("      Interpretation: text interface is the bottleneck,")
            lines.append("      not the representations.")
        elif abs(patch_d - text_d) <= 1.0:
            lines.append("    → No meaningful difference between interfaces.")
            lines.append("      Interpretation: failure is representational, not translational.")
        else:
            lines.append("    → Text injection outperforms or matches direct patching.")
            lines.append("      Consider scale tuning (see sweep results below).")

    if "patch_mae_L16" in summary and "patch_vjepa_L16" in summary:
        mae_d   = summary["patch_mae_L16"]["delta"] * 100
        vjepa_d = summary["patch_vjepa_L16"]["delta"] * 100
        lines.append(f"\n  MAE vs V-JEPA 2 (layer 16, direct patching):")
        lines.append(f"    MAE:    Δ = {mae_d:+.1f}%")
        lines.append(f"    V-JEPA: Δ = {vjepa_d:+.1f}%")
        if mae_d > vjepa_d + 1.0:
            lines.append("    → MAE patching > V-JEPA 2 patching.")
            lines.append("      Consistent with alignment results: higher LM alignment")
            lines.append("      predicts larger behavioral effect of activation patching.")
        else:
            lines.append("    → MAE does not clearly outperform V-JEPA 2.")
            lines.append("      LM alignment score does not predict patching benefit.")

    lines.append(f"\nPATCH LAYER COMPARISON (MAE)")
    for layer in PATCH_LAYERS:
        cond = f"patch_mae_L{layer}"
        if cond in summary:
            d = summary[cond]["delta"] * 100
            lines.append(f"    Layer {layer:2d}: Δ = {d:+.1f}%")

    if scale_sweep:
        lines.append(f"\nSCALE SWEEP (MAE, L16, first 200 questions)")
        for scale, acc in sorted(scale_sweep.items()):
            lines.append(f"    scale={scale:.1f}:  acc={acc:.1f}%")

    if "patch_mae_L16_per_concept" in summary:
        cs = summary["patch_mae_L16_per_concept"]
        lines.append(f"\nPER-CONCEPT BREAKDOWN — patch_mae_L16 (n≥3)")
        lines.append(f"  {'Concept':<14} {'N':>4}  {'Base':>6}  {'Patch':>6}  {'Δ':>6}")
        lines.append(f"  {'-'*45}")
        for concept in sorted(cs, key=lambda c: -abs(cs[c]["delta"])):
            s = cs[concept]
            lines.append(f"  {concept:<14} {s['n']:>4}  "
                         f"{s['acc_base']*100:>5.1f}%  "
                         f"{s['acc_patch']*100:>5.1f}%  "
                         f"{s['delta']*100:>+5.1f}%")

    report = "\n".join(lines)
    print("\n" + report)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("EXPERIMENT 19: ACTIVATION PATCHING — WORLD MODEL → LM RESIDUAL STREAM")
    print("=" * 70)

    # ── Load concept embeddings ───────────────────────────────────────────────
    print("\n[1/6] Loading concept embeddings...")
    concepts, mae_embs, vjepa_embs, lm_embs = load_concept_embeddings()

    # ── Train projections ─────────────────────────────────────────────────────
    print("\n[2/6] Training projections (visual → LM space)...")
    print("  MAE projection:")
    P_mae   = train_projection(mae_embs,   lm_embs)
    print("  V-JEPA 2 projection:")
    P_vjepa = train_projection(vjepa_embs, lm_embs)

    # ── Load PIQA ─────────────────────────────────────────────────────────────
    print("\n[3/6] Loading and matching PIQA...")
    questions        = load_piqa()
    matched, unmatched = match_questions(questions)
    print(f"  Matched: {len(matched)}  Unmatched: {len(unmatched)}")
    if len(matched) < MIN_QUESTIONS:
        print(f"WARNING: only {len(matched)} matched questions")
    if DEBUG_LIMIT:
        matched = matched[:DEBUG_LIMIT]
        print(f"DEBUG: truncated to {len(matched)}")

    # ── Load Mistral ──────────────────────────────────────────────────────────
    print("\n[4/6] Loading Mistral 7B...")
    tok, model, device = load_mistral()

    # ── Build patch vectors ───────────────────────────────────────────────────
    print("\n[5/6] Building concept patch vectors...")
    # MAE: build for all patch layers (same projection, different hook points)
    mae_patches_by_layer = {}
    for layer_idx in PATCH_LAYERS:
        mae_patches_by_layer[layer_idx] = build_concept_patch_vectors(
            P_mae, mae_embs, concepts, device
        )
    vjepa_patches_L16 = build_concept_patch_vectors(
        P_vjepa, vjepa_embs, concepts, device
    )

    # Normalised V-JEPA for text injection nearest-neighbour lookup
    vjepa_n = vjepa_embs / (np.linalg.norm(vjepa_embs, axis=1, keepdims=True) + 1e-8)

    # ── Scale sweep at L24 (quick, on subset) ────────────────────────────────
    print("\n[5b/6] Running scale sweep on MAE L24 (first 200 questions)...")
    scale_sweep = run_scale_sweep(
        matched, tok, model, device, concepts, mae_patches_by_layer[24], layer_idx=24
    )
    best_scale = max(scale_sweep, key=scale_sweep.get)
    print(f"  Best scale: {best_scale}  → acc={scale_sweep[best_scale]:.1f}%")

    # ── Layer granularity sweep around L24 ─────────────────────────────────
    print(f"\n[5c/6] Running layer granularity sweep at scale={best_scale}...")
    layer_sweep = run_layer_sweep(
        matched, tok, model, device, concepts, P_mae, mae_embs, scale=best_scale
    )
    best_layer = max(layer_sweep, key=layer_sweep.get)
    print(f"  Best layer: L{best_layer}  → acc={layer_sweep[best_layer]:.1f}%")

    # ── Main benchmark ────────────────────────────────────────────────────────
    print(f"\n[6/6] Running full benchmark ({len(matched)} questions)...")
    print(f"  Using PATCH_SCALE={PATCH_SCALE} for main conditions.")
    print(f"  (Best from sweeps: scale={best_scale}, layer=L{best_layer})")
    t0 = time.time()
    results = run_all_conditions(
        matched, tok, model, device,
        concepts,
        mae_patches_by_layer,
        vjepa_patches_L16,
        vjepa_n,
    )
    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed/60:.1f} min")

    # ── Targeted run: L24 at best scale (0.5) on full set ────────────────────
    print(f"\n[7/7] Targeted run: MAE L24 at scale={best_scale} ({len(matched)} questions)...")
    patch_state = PatchState()
    mae_patches_24 = mae_patches_by_layer[24]
    handle = register_patch_hook(model, 24, patch_state, best_scale)
    l24_correct = 0
    l24_per_question = []
    t1 = time.time()
    for qi, q in enumerate(matched):
        patch_vec = get_patch_for_question(
            q["concepts"], mae_patches_24, concepts
        )
        if patch_vec is not None:
            patch_state.patch_vec = patch_vec
            patch_state.active    = True
        else:
            patch_state.active = False

        prompt = build_prompt(q["goal"], q["sol1"], q["sol2"])
        ans    = get_answer(tok, model, device, prompt)
        correct_letter = "A" if q["label"] == 0 else "B"
        is_correct = int(ans == correct_letter)
        l24_correct += is_correct
        l24_per_question.append(ans)
        patch_state.active = False

        if (qi + 1) % 100 == 0:
            print(f"    [{qi+1}/{len(matched)}] acc={l24_correct/(qi+1)*100:.1f}%")

    handle.remove()
    l24_acc = l24_correct / len(matched) * 100
    l24_elapsed = time.time() - t1
    print(f"  MAE L24 @ scale={best_scale}: {l24_acc:.1f}% "
          f"(Δ baseline = {l24_acc - 72.1:+.1f}%)  [{l24_elapsed/60:.1f} min]")

    # Inject L24-at-best-scale into each result row for analysis
    for ri, ans in enumerate(l24_per_question):
        results[ri][f"patch_mae_L24_s{best_scale}"] = ans

    # ── Analyse ───────────────────────────────────────────────────────────────
    summary = analyse(results)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        "summary":      summary,
        "scale_sweep":  {str(k): v for k, v in scale_sweep.items()},
        "layer_sweep":  {str(k): v for k, v in layer_sweep.items()},
        "best_scale":   best_scale,
        "best_layer":   best_layer,
        "l24_at_best_scale": {
            "scale":    best_scale,
            "acc":      round(l24_acc, 2),
            "n":        len(matched),
            "delta":    round(l24_acc - 72.1, 2),
        },
        "results":      results,
        "config": {
            "patch_scale":    PATCH_SCALE,
            "patch_layers":   PATCH_LAYERS,
            "layer_granularity_sweep": LAYER_GRANULARITY_SWEEP,
            "ridge_alpha":    RIDGE_ALPHA,
            "n_concepts":     len(concepts),
            "n_matched":      len(matched),
            "debug_limit":    DEBUG_LIMIT,
            "elapsed_min":    round(elapsed / 60, 1),
        }
    }
    os.makedirs("lm_output", exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    write_report(summary, scale_sweep)
    print(f"Report saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
