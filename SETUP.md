# Setup Guide — Latent Bridge Experiment

## What you need to buy: Nothing
## What you need to install: See below

---

## 1. System requirements check

```bash
# Check your GPU
nvidia-smi

# You need: 40GB+ VRAM, CUDA 11.8+
# Expected memory usage during experiment:
#   Mistral 7B (4-bit): ~4GB
#   World model:        ~6GB
#   Bridge + training:  ~8GB
#   Total:              ~18GB  (comfortable on 40GB)
```

---

## 2. Environment setup

```bash
# Create isolated environment
conda create -n latent-bridge python=3.11
conda activate latent-bridge

# Core dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# LLM + quantization
pip install transformers==4.40.0 bitsandbytes accelerate

# World model (DreamerV3 PyTorch port)
pip install dreamerv3

# Utilities
pip install numpy datasets sentence-transformers tqdm
```

---

## 3. Run the experiment (simulated mode first)

```bash
# This runs immediately, no downloads needed
# Uses simulated representations to verify the architecture works
python shared_codebook.py
```

Expected output after ~2 minutes:
```
Epoch  0 | Loss: 4.2341 | Code Agreement: 0.041 ...
Epoch  5 | Loss: 2.1203 | Code Agreement: 0.124 ...
Epoch 10 | Loss: 1.3401 | Code Agreement: 0.231 ...
Epoch 20 | Loss: 0.8821 | Code Agreement: 0.387 ✓ converging
...
Code agreement: 18/30 concepts (60.0%)
```

---

## 4. Upgrade to real models

### Step 4a: Real LLM representations
Uncomment in shared_codebook.py:
```python
lm_hiddens = extract_lm_representations(concepts, cfg.lm_model_id, cfg.device)
```
This downloads Mistral 7B (~14GB) and runs forward passes.
Takes ~20 minutes on first run, cached after.

### Step 4b: Real world model representations
```bash
# Install DreamerV3 dependencies
pip install gymnasium minedojo  # or habitat-sim for robotics

# Then swap simulate_wm_representations() for:
# - Run DreamerV3 on visual observations of each concept
# - Extract RSSM stochastic state vectors
# See: https://github.com/NM512/dreamerv3-torch
```

---

## 5. Key files

| File                    | Purpose                              |
|-------------------------|--------------------------------------|
| shared_codebook.py      | Main experiment — run this           |
| checkpoints/bridge.pt   | Saved model after training           |
| checkpoints/history.json| Training loss curve                  |

---

## 6. What success looks like

| Metric              | Weak result | Strong result |
|---------------------|-------------|---------------|
| Code agreement      | < 20%       | > 40%         |
| LM recon error      | > 0.5       | < 0.1         |
| WM recon error      | > 0.5       | < 0.1         |

Strong results on simulated data → run on real models.
Strong results on real models → potentially publishable.

---

## 7. The publishable claim (if it works)

"We demonstrate that a vector-quantized bottleneck trained jointly 
on world model latent states and LLM hidden representations converges 
to a shared discrete codebook, with X% concept-level code agreement, 
supporting the Platonic Representation Hypothesis and providing an 
interpretable inter-module communication protocol for multi-system AI."

That's a Nature Communications-level framing. Given your publications
history, you know what that abstract needs to look like.
