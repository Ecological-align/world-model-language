# Can a World Model and a Language Model Share a Vocabulary?

This repository contains the code and extracted representations for the paper/blog post:
**"Can a World Model and a Language Model Share a Vocabulary?"**

We test whether a temporal world model (V-JEPA 2) and language models (Mistral 7B, Sentence-Transformers) organize physical concepts similarly enough to share a discrete codebook — and what it takes to make that work.

**Key findings:**
- Naive cross-modal RSA shows r=+0.325 (Mistral vs CLIP) — but this is entirely a categorical artifact. Within physical concepts, alignment is near zero.
- V-JEPA 2 aligns with CLIP (r=+0.404, p=0.005); MAE does not (r=+0.087) — training objective, not architecture, is what matters.
- Reconstruction-only codebooks always collapse to 2 codes (one per modality). This is structural, not a tuning failure.
- NT-Xent contrastive loss at λ=0.5 robustly solves the collapse: 99% ± 2% cross-modal agreement across 5 seeds.
- High-polysemy, high-sensorimotor concepts (esp. *hand*) require more alignment pressure but are not categorically unalignable.

---

## Repository Structure

```
codebook/
│
├── extract_lm_standalone.py       # Extract Mistral 7B + Sentence-Transformers + CLIP representations
├── extract_wm_visual.py           # Extract V-JEPA 2 + MAE representations
│
├── rsa.py                         # RSA utilities (build RDMs, correlate, permutation test)
├── compare_all.py                 # 5-way RSA across all models
│
├── train_codebook.py              # 2-way VQ codebook: Mistral ↔ V-JEPA 2
├── train_codebook_3way.py         # 3-way VQ codebook: Mistral + CLIP + V-JEPA 2
├── train_codebook_st.py           # ST-based codebooks (2-way and 3-way)
├── train_codebook_contrastive.py  # NT-Xent contrastive codebook, single λ sweep
├── train_codebook_contrastive_multiseed.py  # 15-run multi-seed validation (3 λ × 5 seeds)
│
├── preregister_polysemy.py        # Polysemy + sensorimotor pre-registration (run before multi-seed)
│
└── lm_output/
    ├── lm_hiddens.npy             # Mistral 7B layer 16 last-token [71, 4096]
    ├── st_hiddens.npy             # Sentence-Transformers all-mpnet-base-v2 [71, 768]
    ├── clip_hiddens.npy           # CLIP ViT-L/14 visual projection [71, 768]
    ├── vjepa2_hiddens.npy         # V-JEPA 2 ViT-L mean context tokens [71, 1024]
    ├── mae_hiddens.npy            # MAE ViT-L mean patch tokens [71, 1024]
    ├── rsa_comparison.json        # Exp 1: RSA on 33 concepts (physical + abstract)
    ├── rsa_physical_only.json     # Exp 2: RSA on 17 physical concepts only
    ├── rsa_5way_physical.json     # Exp 3: 5-way RSA with V-JEPA 2 + MAE
    ├── codebook_results.json      # Exp 4: 2-way Mistral↔VJ codebook results
    ├── codebook3way_results.json  # Exp 5: 3-way Mistral+CLIP+VJ codebook results
    ├── codebook_st3way_results.json  # Exp 6: ST-based codebook results
    ├── codebook_contrastive_results.json   # Exp 7: single-run λ sweep
    ├── codebook_multiseed_results.json     # Multi-seed: per-seed results (15 runs)
    ├── codebook_multiseed_summary.json     # Multi-seed: mean ± std summary table
    └── polysemy_preregistration.json       # Pre-registered polysemy predictions
```

---

## Concept Set

17 physical concepts used throughout:

> apple, chair, water, fire, stone, rope, door, container, shadow, mirror, knife, wheel, hand, wall, hole, bridge, ladder

For Experiments 1–2, 16 abstract concepts are added as a category-level control (see paper Section 3.1).

Each concept uses its Wikipedia introductory paragraph (text models) and Wikipedia lead image (vision models), fetched via the MediaWiki API and cached locally.

---

## Hardware & Environment

```
OS:       Windows 11
GPU:      NVIDIA RTX 5090 32GB VRAM
CUDA:     12.9
Python:   3.11+
PyTorch:  2.10.0+cu128
transformers: 5.3.0
```

### Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install transformers sentence-transformers numpy scipy requests pillow
```

**Windows note:** Add this to the top of any script that prints Unicode:
```python
import sys
sys.stdout.reconfigure(encoding="utf-8")
```

---

## Reproducing the Experiments

Run experiments in order. Extracted representations are included in `lm_output/` so you can skip extraction and start from Experiment 1 if you prefer.

---

### Step 0: Extract Representations (optional — outputs already included)

**Language models + CLIP:**
```bash
python extract_lm_standalone.py
```
Extracts Mistral 7B Instruct (layer 16 last-token), Sentence-Transformers (all-mpnet-base-v2 CLS pooled), and CLIP ViT-L/14 (visual projection) for all 71 concepts. Saves to `lm_output/`.

**V-JEPA 2 + MAE:**
```bash
python extract_wm_visual.py
```
Extracts V-JEPA 2 ViT-L (mean of context tokens, static image repeated 8× to form pseudo-video) and MAE ViT-L (mean of all 197 patch tokens, mask_ratio=0.0). Saves to `lm_output/`.

**Model loading notes:**
- V-JEPA 2: use `AutoVideoProcessor`, pass `skip_predictor=True`, use `dtype=` not `torch_dtype=`
- MAE: set `config.mask_ratio = 0.0` before loading to extract all tokens
- CLIP: use `vision_model()` + `visual_projection()` directly; `get_image_features()` is broken in transformers 5.x
- Wikipedia images: 403 errors are common with direct URLs; the script uses local caching with 2s delays

---

### Experiment 1: Overall RSA (33 concepts)

```bash
python compare_all.py --concepts all
```

Expected output:
```
Mistral vs CLIP:  r=+0.325  p=1.7e-14  ✓
Mistral vs ST:    r=+0.148  p=...       ✓
ST vs CLIP:       r=+0.013  p=...       ✗
```

The r=+0.325 signal looks like cross-modal alignment but is entirely driven by physical vs. abstract category separation.

---

### Experiment 2: Physical-Only RSA (17 concepts)

```bash
python compare_all.py --concepts physical
```

Expected output:
```
Mistral vs CLIP:  r=+0.039  ✗  (was +0.325 on full set)
ST vs CLIP:       r=+0.089  ✗
Mistral vs ST:    r=+0.171  borderline
```

---

### Experiment 3: 5-Way RSA with V-JEPA 2 and MAE

```bash
python compare_all.py --concepts physical --models all5
```

Expected output:
```
V-JEPA 2 vs CLIP:    r=+0.404  p=0.005  ✓
MAE vs CLIP:         r=+0.087  p=0.272  ✗
V-JEPA 2 vs Mistral: r=−0.036  p=0.650  ✗
V-JEPA 2 vs ST:      r=−0.025  p=0.757  ✗
```

Temporal prediction (V-JEPA 2) specifically aligns with CLIP. Spatial reconstruction (MAE) does not. Same architecture (ViT-L), different training objective.

---

### Experiment 4: 2-Way VQ Codebook (Mistral ↔ V-JEPA 2)

```bash
python train_codebook.py
```

Expected: Collapse to 2 codes. Cross-modal agreement: 0/17 (0%).

---

### Experiment 5: 3-Way VQ Codebook (Mistral + CLIP + V-JEPA 2)

```bash
python train_codebook_3way.py
```

Expected: Collapse to 2 codes. V-JEPA 2 ↔ CLIP merge; Mistral isolated.

---

### Experiment 6: Sentence-Transformer Codebooks

```bash
python train_codebook_st.py
```

Despite ST having 8× more representational spread than Mistral (0.639 vs 0.074), still collapses to 2 codes. Confirms collapse is structural.

---

### (Optional) Polysemy Pre-Registration

Run **before** the multi-seed experiment to replicate pre-registration as intended:

```bash
python preregister_polysemy.py
```

Ranks all 17 concepts by combined polysemy (WordNet noun senses) + sensorimotor score (Lancaster Norms). Saves to `lm_output/polysemy_preregistration.json`.

---

### Experiment 7: Contrastive Loss — Single Run

```bash
python train_codebook_contrastive.py
```

Sweeps λ ∈ {0.0, 0.1, 0.5, 1.0} with a single seed.

| λ | Active codes | Agreement | Post-VQ RSA |
|---|---|---|---|
| 0.0 | 2 | 0% | +0.000 |
| 0.1 | 6 | 59% | +0.497 |
| 0.5 | 14 | 88% | +0.755 |
| 1.0 | 12 | 71% | +0.411 |

---

### Multi-Seed Validation (15 runs: 3 λ × 5 seeds)

```bash
python train_codebook_contrastive_multiseed.py
```

Seeds: [42, 123, 7, 99, 2025]. Runtime: ~20–40 minutes on RTX 5090.

| λ | Active codes | ST↔VJ agreement | Post-VQ RSA |
|---|---|---|---|
| 0.0 | 2 | 0% ± 0% | +0.000 ± 0.000 |
| 0.1 | 12.2 ± 4.7 | 62% ± 13% | +0.653 ± 0.149 |
| **0.5** | **15.6 ± 1.4** | **99% ± 2%** | **+1.000 ± 0.000** |
| 1.0 | 12.6 ± 1.4 | 93% ± 7% | +0.897 ± 0.094 |

λ=0.5 wins in 5/5 seeds. Results saved to `lm_output/codebook_multiseed_summary.json`.

---

## How the Contrastive Loss Works

NT-Xent is applied to the **pre-quantization projections** (continuous 256-dim vectors, before snapping to codebook):

- **Pulls:** same concept across modalities → close in shared space
- **Pushes:** different concepts across modalities → far apart

Applied pre-VQ because VQ is non-differentiable. The loss shapes the geometry the codebook will quantize. λ=0.5 is the Goldilocks zone: strong enough to overcome between-modality variance, gentle enough to preserve within-modality structure.

---

## Pre-Registration Outcome

| Concept | Predicted | Outcome |
|---|---|---|
| hand (rank 1) | Fail | ✓ Confirmed — hardest across seeds |
| fire (rank 2) | Fail | ✗ Aligns reliably at λ=0.5 |
| stone (rank 3) | Fail | ✗ Aligns reliably at λ=0.5 |
| hole (rank 6) | Not predicted | ~ Harder at low λ, not in top 3 |

Polysemy/sensorimotor score predicts which concepts need more alignment pressure, not which ones categorically fail.

---

## Citation

```bibtex
@misc{[yourname]2026worldmodel,
  title={Can a World Model and a Language Model Share a Vocabulary?},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[yourname]/world-model-language-gap}
}
```

**Key references:**
- Huh et al. (2024). The Platonic Representation Hypothesis. ICML 2024.
- Assran et al. (2025). V-JEPA 2. arXiv:2506.09985.
- Liu et al. (2022). Cross-Modal Discrete Representation Learning. ACL 2022.
- Duan et al. (2022). Multi-Modal Alignment using Representation Codebook. CVPR 2022.
- van den Oord et al. (2017). Neural Discrete Representation Learning. NeurIPS 2017.
- Kriegeskorte et al. (2008). Representational Similarity Analysis. Frontiers in Systems Neuroscience.
- Lynott et al. (2020). The Lancaster Sensorimotor Norms. Behavior Research Methods.

---

## Notes on AI Assistance

Code and analysis were developed with AI assistance (Claude, Anthropic). The research question, experimental design, hardware, and all scientific judgment calls are the author's. The author takes full responsibility for the results and their interpretation.
