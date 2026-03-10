"""
plot_adapter_curves.py
======================

Plots convergence curves and summary tables from finetune_adapter.py.

Produces:
  lm_output/adapter_curves.png      — convergence curves (all models, medium adapter)
  lm_output/adapter_summary.png     — bar chart: baseline vs ceiling vs MAE threshold
  lm_output/adapter_heatmap.png     — heatmap: model × adapter size alignment ceilings

Usage:
  python plot_adapter_curves.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import json
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib not found — install with: pip install matplotlib --break-system-packages")
    print("Falling back to text summary only.")

RESULTS_FILE = "lm_output/adapter_finetune_results.json"
CURVES_FILE  = "lm_output/adapter_curves.json"
OUT_DIR      = "lm_output"

# Colors per model category
COLORS = {
    "mae":       "#1d4ed8",   # blue — image reconstruction (the gold standard)
    "dinov2":    "#7c3aed",   # purple — image distillation
    "clip":      "#0891b2",   # teal — image contrastive
    "vmae_k400": "#ea580c",   # orange — video reconstruction
    "vmae_ssv2": "#dc2626",   # red — video reconstruction (temporal)
    "vjepa2":    "#be123c",   # dark red — video temporal prediction
}

LINESTYLES = {
    "mae":       "-",
    "dinov2":    "--",
    "clip":      "-.",
    "vmae_k400": "-",
    "vmae_ssv2": "--",
    "vjepa2":    "-.",
}

MAE_THRESHOLD = 17.7   # Mistral LM↔MAE natural alignment


def load_data():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    with open(CURVES_FILE) as f:
        curves = json.load(f)
    return results, curves


def plot_convergence(results, curves):
    """Main convergence curves: test alignment vs epoch for medium adapter."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1, ax2 = axes

    # ── Left: convergence curves ────────────────────────────────────────────
    ax1.axhline(MAE_THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
                label=f"MAE natural alignment ({MAE_THRESHOLD}%)", alpha=0.7)

    for key, curve_data in curves.items():
        if key not in results:
            continue
        label    = results[key]["label"]
        color    = COLORS.get(key, "#666")
        ls       = LINESTYLES.get(key, "-")
        med_curve = curve_data.get("medium", [])
        if not med_curve:
            continue
        epochs = [pt[0] for pt in med_curve]
        test   = [pt[2] for pt in med_curve]
        ax1.plot(epochs, test, color=color, linestyle=ls, linewidth=2.0,
                 label=label, marker="o", markersize=2, markevery=3)

        # Mark baseline (epoch 0)
        baseline = results[key]["baseline"]["mean"]
        ax1.scatter([0], [baseline], color=color, marker="s", s=60, zorder=5)

    ax1.set_xlabel("Training Epoch", fontsize=12)
    ax1.set_ylabel("Test Alignment (%)", fontsize=12)
    ax1.set_title("Adapter Convergence Curves\n(medium adapter, test alignment)", fontsize=13)
    ax1.legend(fontsize=9, loc="lower right")
    ax1.set_ylim(0, None)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, "■ = baseline (no adapter)", transform=ax1.transAxes,
             fontsize=8, va="top", color="gray")

    # ── Right: baseline → ceiling summary bars ──────────────────────────────
    model_order = ["mae", "dinov2", "clip", "vmae_k400", "vmae_ssv2", "vjepa2"]
    model_order = [k for k in model_order if k in results]

    labels   = [results[k]["label"] for k in model_order]
    baselines = [results[k]["baseline"]["mean"] for k in model_order]
    ceilings  = [results[k].get("medium", {}).get("final_mean", baselines[i])
                 for i, k in enumerate(model_order)]
    colors    = [COLORS.get(k, "#666") for k in model_order]

    x = np.arange(len(labels))
    w = 0.35

    bars1 = ax2.bar(x - w/2, baselines, w, label="Baseline (no adapter)",
                    color=colors, alpha=0.4, edgecolor="white")
    bars2 = ax2.bar(x + w/2, ceilings,  w, label="After adapter fine-tuning",
                    color=colors, alpha=0.9, edgecolor="white")

    ax2.axhline(MAE_THRESHOLD, color="gray", linestyle=":", linewidth=1.5,
                label=f"MAE natural ({MAE_THRESHOLD}%)", alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=22, ha="right", fontsize=10)
    ax2.set_ylabel("Test Alignment (%)", fontsize=12)
    ax2.set_title("Baseline vs Adapter Ceiling\n(medium adapter)", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3, axis="y")

    # Annotate gain
    for i, (b, c) in enumerate(zip(baselines, ceilings)):
        gain = c - b
        ax2.text(x[i] + w/2, c + 0.3, f"+{gain:.1f}%", ha="center",
                 fontsize=8, color="#333")

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adapter_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_heatmap(results):
    """Heatmap: model × adapter size alignment ceilings."""
    model_order = ["mae", "dinov2", "clip", "vmae_k400", "vmae_ssv2", "vjepa2"]
    model_order = [k for k in model_order if k in results]
    size_order  = ["small", "medium", "large"]
    size_order  = [s for s in size_order if s in results.get(model_order[0], {})]

    if not size_order:
        return

    labels = [results[k]["label"] for k in model_order]
    data   = np.zeros((len(model_order), len(size_order)))

    for i, key in enumerate(model_order):
        for j, size in enumerate(size_order):
            data[i, j] = results[key].get(size, {}).get("final_mean", 0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, cmap="Blues", aspect="auto", vmin=0)

    ax.set_xticks(range(len(size_order)))
    ax.set_xticklabels([f"{s.capitalize()} adapter" for s in size_order], fontsize=11)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(len(model_order)):
        for j in range(len(size_order)):
            val = data[i, j]
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=11, color="white" if val > data.max() * 0.6 else "black",
                    fontweight="bold")

    ax.set_title("Alignment Ceiling by Model × Adapter Size\n(test alignment after fine-tuning)",
                 fontsize=12)
    plt.colorbar(im, label="Test alignment (%)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "adapter_heatmap.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_text_summary(results):
    """Always-available text summary, regardless of matplotlib."""
    print(f"\n{'='*72}")
    print("ADAPTER FINE-TUNING RESULTS SUMMARY")
    print(f"{'='*72}")
    print(f"  MAE natural alignment threshold: {MAE_THRESHOLD}%\n")

    model_order = ["mae", "dinov2", "clip", "vmae_k400", "vmae_ssv2", "vjepa2"]
    model_order = [k for k in model_order if k in results]

    print(f"  {'Model':<20}  {'Baseline':>9}  {'Small':>7}  {'Medium':>8}  "
          f"{'Large':>7}  {'Gain(M)':>8}  {'Steps':>8}  {'Reached':>8}")
    print(f"  {'-'*80}")

    for key in model_order:
        res   = results[key]
        base  = res["baseline"]["mean"]
        small = res.get("small",  {}).get("final_mean", float("nan"))
        med   = res.get("medium", {}).get("final_mean", float("nan"))
        large = res.get("large",  {}).get("final_mean", float("nan"))
        gain  = res.get("medium", {}).get("gain", float("nan"))
        steps = res.get("medium", {}).get("steps_to_threshold")
        reach = res.get("medium", {}).get("reached_threshold_pct", 0)
        steps_s = f"{steps:.0f}" if steps else "never"
        print(f"  {res['label']:<20}  {base:>8.1f}%  {small:>6.1f}%  {med:>7.1f}%  "
              f"{large:>6.1f}%  {gain:>+7.1f}%  {steps_s:>8}  {reach:>7.0f}%")

    # Key comparison
    print(f"\n  Key comparison (same objective, different data):")
    if "mae" in results and "vmae_k400" in results:
        mae_ceil  = results["mae"].get("medium",     {}).get("final_mean", float("nan"))
        vmae_ceil = results["vmae_k400"].get("medium", {}).get("final_mean", float("nan"))
        gap       = mae_ceil - vmae_ceil
        print(f"    MAE ceiling:          {mae_ceil:.1f}%")
        print(f"    VideoMAE-K400 ceiling:{vmae_ceil:.1f}%")
        print(f"    Residual gap:         {gap:+.1f}%")
        if gap > 3:
            print(f"    → Structural gap persists after adaptation.")
        else:
            print(f"    → Adapter fully bridges the gap.")


def main():
    if not os.path.exists(RESULTS_FILE):
        print(f"Results file not found: {RESULTS_FILE}")
        print("Run finetune_adapter.py first.")
        return

    results, curves = load_data()
    print_text_summary(results)

    if HAS_PLOT:
        print(f"\nGenerating plots...")
        plot_convergence(results, curves)
        plot_heatmap(results)
        print("Done.")
    else:
        print("\nInstall matplotlib to generate plots:")
        print("  pip install matplotlib --break-system-packages")


if __name__ == "__main__":
    main()
