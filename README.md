# The Gap Between Language Models and World Models Is Temporal, Not Visual

Code and data for the paper/blog post: **"The Gap Between Language Models and World Models Is Temporal, Not Visual"**

We investigate whether a language model (Mistral 7B) and a temporal world model (V-JEPA 2) can share a discrete representational codebook for physical concepts — and what the structure of their alignment failure reveals about the nature of the gap.

---

## Revised Finding

The most natural hypothesis — that the gap is between *language* representations and *visual* representations — is wrong. So is the refined hypothesis — that it is between *language-supervised* and *physics-supervised* representations. 

A four-modality probe (LM, V-JEPA 2, CLIP-text, MAE) shows that models trained on **static inputs** cluster together in shared codebook space regardless of modality, while only V-JEPA 2 — trained to predict masked spatio-temporal video regions — occupies a structurally distinct region.

| Model | Training signal | Temporal? | Agreement with LM |
|-------|----------------|-----------|-------------------|
| MAE | Spatial reconstruction (static images) | No | **48.8%** |
| CLIP-text | Language alignment (image-caption pairs) | No | 44.4% |
| V-JEPA 2 | Temporal prediction (masked video) | **Yes** | 40.6% |

The decrease is monotonic and stable across 6 hyperparameter configurations. **The gap is about *what happens next*, not about modality.**

---

## Key Results

- Cross-modal RSA alignment (r=+0.325, Mistral vs CLIP) is a **category structure artifact** — it disappears entirely when restricting to physical concepts only
- Reconstruction-only codebooks always collapse to 2 codes (one per modality) — this is structural, not a tuning failure
- NT-Xent contrastive loss at **λ=0.5** achieves 99% ± 2% cross-modal agreement, robust across 5/5 seeds
- Generalizes: 17-concept memorization (test 2%) → 49-concept genuine generalization (**test 92%**)
- Translating world model knowledge to natural language **hurts** PIQA performance (Δ=−6.6%, McNemar p=0.999) — a sense mismatch, not a knowledge gap
- Phrase-level event grounding achieves **89.5% test agreement** with a −1.0% generalization gap (test > train)
- The four-modality probe is **stable in 5/6 hyperparameter configurations**: LM↔MAE consistently highest (48.8%), VIS↔CLIP consistently lowest (15.6%)

---

## Repository Structure

```
codebook/
│
├── Extraction
│   ├── extract_lm_standalone.py        # Mistral 7B + Sentence-Transformers + CLIP
│   ├── extract_wm_visual.py            # V-JEPA 2 + MAE
│   ├── extract_expanded.py             # Expanded 49-concept extraction
│   ├── extract_phrase_level.py         # Phrase-level event extraction (251 events)
│   └── extract_multimodal.py           # Multi-image averaging extraction
│
├── Codebook training (concept-level)
│   ├── train_codebook_contrastive.py           # 2-way contrastive codebook, λ sweep
│   ├── train_codebook_contrastive_multiseed.py # 15-run multi-seed validation
│   ├── train_codebook_generalization.py        # 17→49 concept generalization
│   ├── generalization_balanced.py              # Batch-balanced generalization control
│   └── novq_baseline.py                        # No-VQ contrastive projection baseline
│
├── Codebook training (phrase-level)
│   ├── phrase_bank.py                  # 49 concepts × 5 phrases = 251 events
│   ├── train_phrase_codebook.py        # Phrase-level shared codebook (50 runs)
│   ├── lambda_sweep_phrase.py          # 8-config λ sweep at phrase level
│   ├── train_trimodal_codebook.py      # 3-modality: LM + VIS + CLIP-text
│   ├── train_quadmodal_codebook.py     # 4-modality: LM + VIS + CLIP-text + MAE
│   └── quadmodal_stability.py         # Stability across 6 hyperparameter configs
│
├── Analysis
│   ├── rsa_expanded.py                 # RSA on 49 concepts with bootstrap CIs
│   ├── code_analysis.py                # Codebook structure analysis
│   ├── polysemy_frequency.py           # Polysemy × frequency × alignment analysis
│   └── preregister_expanded.py         # Pre-registration of polysemy predictions
│
├── Downstream
│   ├── piqa_benchmark.py               # PIQA evaluation with world model context
│   └── diagnose_generalization.py      # Generalization failure diagnostics
│
├── Utilities
│   ├── add_clip_text.py                # Extract CLIP text embeddings for phrases
│   ├── check_phrase_images.py          # Verify phrase image coverage
│   ├── download_multi_images.py        # Multi-image concept download
│   └── download_concept_videos.py      # Video download utilities
│
├── blog_post.html                      # Full blog post (current version)
├── world_model_language_gap_paper.docx # Paper (current version)
│
└── lm_output/
    ├── *_hiddens*.npy                  # Extracted embeddings (concept-level)
    ├── phrase_level/                   # Phrase-level embeddings and results
    │   ├── *_hiddens_phrase.npy
    │   ├── event_index.json
    │   ├── codebook_results.json
    │   ├── lambda_sweep_results.json
    │   ├── trimodal_codebook_results.json
    │   ├── quadmodal_codebook_results.json
    │   └── quadmodal_stability_results.json
    ├── rsa_expanded_results.json
    ├── generalization_balanced_results.json
    ├── piqa_results.json
    └── polysemy_frequency_analysis.json
```

---

## Concept Set

**49 physical concepts** used in the full experiments:

> apple, chair, water, fire, stone, rope, door, container, shadow, mirror, knife, wheel, hand, wall, hole, bridge, ladder, spring, leaf, thread, feather, sand, ice, glass, cloud, coin, shelf, pipe, net, chain, bowl, field, bucket, fence, wave, branch, bark, gear, needle, log, hinge, lens, piston, valve, wedge, pulley, anvil, bellows, trough

The original 17 concepts (apple → ladder) were used for Experiments 1–7. Experiments 8–12 use all 49.

**251 phrase-level events** (phrase_bank.py): 49 concepts × ~5 phrases each, covering distinct physical senses per concept (e.g., *fire burning in a fireplace* vs *fire spreading through dry grass*).

---

## Experiment Summary

| Exp | Description | Key result |
|-----|-------------|------------|
| 1 | RSA, 33 concepts (physical + abstract) | r=+0.325 Mistral↔CLIP — artifact |
| 2 | RSA, physical only | r≈0 — artifact confirmed |
| 3 | RSA expanded to 49 concepts, +MAE | V-JEPA 2↔CLIP r=+0.917, MAE↔CLIP r=+0.723 |
| 4 | Codebook, no supervision | Collapse to 2 codes always |
| 5 | Codebook + contrastive λ sweep | λ=0.5: 99%±2%, robust 5/5 seeds |
| 6 | Generalization, 17 concepts | Test 2% — memorization |
| 7 | Generalization, 49 concepts | Test 92% — genuine generalization |
| 8 | PIQA downstream benchmark | Δ=−6.6%, p=0.999 — sense mismatch |
| 9 | Phrase-level codebook | Test 89.5%, gap=−1.0% |
| 10 | λ sweep (phrase level) | Binary collapse is objective-driven |
| 11 | Trimodal: LM + VIS + CLIP-text | CLIP-text closer to LM than to VIS |
| 12 | Quadmodal: + MAE; stability check | LM↔MAE highest; temporal gap confirmed |

---

## Reproducing the Key Results

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install torch transformers sentence-transformers numpy scipy

# 2. Extract representations (requires model downloads ~15GB)
python extract_expanded.py          # concept-level, 49 concepts
python extract_phrase_level.py      # phrase-level, 251 events
python add_clip_text.py             # CLIP text embeddings for phrases

# 3. Run RSA analysis
python rsa_expanded.py

# 4. Train concept-level codebook
python train_codebook_contrastive_multiseed.py  # 15 runs, λ sweep
python generalization_balanced.py               # generalization test

# 5. Run phrase-level experiments
python train_phrase_codebook.py     # 50 runs
python lambda_sweep_phrase.py       # 8-config sweep

# 6. Run multi-modality probe
python train_trimodal_codebook.py   # LM + VIS + CLIP-text
python train_quadmodal_codebook.py  # + MAE
python quadmodal_stability.py       # stability across hyperparams

# 7. PIQA benchmark
python piqa_benchmark.py
```

---

## Models

| Model | Source | Notes |
|-------|--------|-------|
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.2` | Last-token hidden state, layer 16 |
| all-mpnet-base-v2 | `sentence-transformers/all-mpnet-base-v2` | Mean-pooled |
| CLIP ViT-L/14 | `openai/clip-vit-large-patch14` | Visual projection (images) or text projection (phrases) |
| V-JEPA 2 ViT-L | Meta AI | CLS token, 8× static frame (no temporal variation — known limitation) |
| MAE ViT-L | `facebook/vit-mae-large` | CLS token, mask_ratio=0 |

**Hardware:** RTX 5090 32GB · CUDA 12.9 · PyTorch 2.10 · transformers 5.3.0

---

## Prior Work

Shared discrete codebooks with cross-modal contrastive objectives are an established technique:
- Liu et al. (ACL 2022) — cross-modal code matching for video-audio-text
- Duan et al. (CVPR 2022) — representation codebook as image-text bridge
- LG-VQ (NeurIPS 2024) — language-guided codebook learning

What is novel here: the specific modality combination, the multi-modality probe isolating the temporal axis, and the falsification of the modality/supervision-type hypotheses in favor of the temporal dynamics hypothesis.

---

## References

1. Huh et al. (2024). Position: The Platonic Representation Hypothesis. ICML 2024.
2. Kriegeskorte et al. (2008). Representational similarity analysis. Frontiers in Systems Neuroscience.
3. Assran et al. (2025). V-JEPA 2. arXiv:2506.09985.
4. He et al. (2022). Masked autoencoders are scalable vision learners. CVPR 2022.
5. van den Oord et al. (2017). Neural discrete representation learning. NeurIPS 2017.
6. Liu et al. (2022). Cross-modal discrete representation learning. ACL 2022.
7. Duan et al. (2022). Multi-modal alignment using representation codebook. CVPR 2022.
8. Bisk et al. (2020). PIQA. AAAI 2020.
