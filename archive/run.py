"""
RSA Experiment: World Model vs LLM Representational Alignment
--------------------------------------------------------------
The core question: do a world model (DreamerV3) and an LLM (Mistral 7B),
trained completely independently on different data with different objectives,
organize the same concepts in similar relational structures?

Run modes:
    python run.py               -- simulation mode (runs immediately)
    python run.py --mode real   -- real models (requires downloads)

Output:
    rsa_results.png   -- 6-panel figure
    rsa_results.json  -- full numerical results
"""

import argparse
import json
import numpy as np
from pathlib import Path

from concepts import ALL_CONCEPTS, CATEGORY_BOUNDARIES, N_CONCEPTS
import extract_wm
import extract_lm
from rsa import cosine_similarity_matrix, run_full_analysis
from visualize import plot_all


def run(mode: str = "simulation"):
    print("="*60)
    print("RSA EXPERIMENT: World Model vs LLM")
    print(f"Mode: {mode.upper()}")
    print(f"Concepts: {N_CONCEPTS}")
    print("="*60)

    # ─── Step 1: Extract representations (independently) ───────────────
    print("\n[ STEP 1 ] Extracting world model representations...")
    wm_vectors = extract_wm.extract(ALL_CONCEPTS, mode=mode)

    print("\n[ STEP 2 ] Extracting LLM representations...")
    lm_vectors = extract_lm.extract(ALL_CONCEPTS, mode=mode)

    # ─── Step 2: Build similarity matrices ─────────────────────────────
    print("\n[ STEP 3 ] Computing representational similarity matrices...")
    rsm_wm = cosine_similarity_matrix(wm_vectors)
    rsm_lm = cosine_similarity_matrix(lm_vectors)
    print(f"  WM RSM: {rsm_wm.shape}  (range: {rsm_wm.min():.3f} to {rsm_wm.max():.3f})")
    print(f"  LM RSM: {rsm_lm.shape}  (range: {rsm_lm.min():.3f} to {rsm_lm.max():.3f})")

    # ─── Step 3: RSA analysis ───────────────────────────────────────────
    print("\n[ STEP 4 ] Running RSA analysis...")
    results = run_full_analysis(
        rsm_wm, rsm_lm,
        CATEGORY_BOUNDARIES,
        ALL_CONCEPTS
    )

    # ─── Step 4: Save results ───────────────────────────────────────────
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save JSON results
    results_json = {k: v for k, v in results.items()
                    if k != "by_category"}
    results_json["by_category"] = {
        cat: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
              for kk, vv in val.items()}
        for cat, val in results["by_category"].items()
    }
    results_json["mode"] = mode
    results_json["n_concepts"] = N_CONCEPTS

    with open(output_dir / "rsa_results.json", "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to outputs/rsa_results.json")

    # Save RSMs
    np.save(output_dir / "rsm_wm.npy", rsm_wm)
    np.save(output_dir / "rsm_lm.npy", rsm_lm)
    np.save(output_dir / "wm_vectors.npy", wm_vectors)
    np.save(output_dir / "lm_vectors.npy", lm_vectors)

    # ─── Step 5: Visualize ──────────────────────────────────────────────
    print("\n[ STEP 5 ] Generating visualizations...")
    try:
        plot_all(
            rsm_wm, rsm_lm,
            wm_vectors, lm_vectors,
            ALL_CONCEPTS, CATEGORY_BOUNDARIES,
            results,
            save_path=str(output_dir / "rsa_results.png")
        )
    except Exception as e:
        print(f"  Visualization failed: {e}")
        print("  (Results still saved to JSON)")

    # ─── Summary ────────────────────────────────────────────────────────
    r = results["overall_spearman_r"]
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Overall alignment (Spearman r): {r:.4f}")
    print()
    print("Category breakdown:")
    for cat, res in results["by_category"].items():
        bar = "█" * int(max(0, res["r"]) * 20)
        sig = "✓" if res["significant"] else " "
        print(f"  {sig} {cat:12s} r={res['r']:+.3f}  {bar}")

    print()
    if mode == "simulation":
        print("─" * 60)
        print("NOTE: Running in simulation mode.")
        print("These results verify the pipeline works correctly.")
        print()
        print("To run the real experiment:")
        print("  1. pip install transformers bitsandbytes accelerate")
        print("  2. pip install dreamerv3 minedojo")
        print("  3. python run.py --mode real")
        print()
        print("Expected real-world result timeline:")
        print("  LLM extraction:  ~20 min (first run, downloads Mistral 7B)")
        print("  WM extraction:   ~2-4 hours (depends on environment setup)")
        print("  RSA analysis:    ~2 min")
        print("─" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["simulation", "real"],
        default="simulation",
        help="'simulation' runs immediately with structured noise. "
             "'real' requires DreamerV3 + Mistral 7B."
    )
    args = parser.parse_args()
    run(mode=args.mode)
