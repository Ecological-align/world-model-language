"""
extract_alt_lm.py
=================

Extract phrase-level hidden states from alternative LMs to test whether
the language-vision gap is Mistral-specific or model-general.

Models:
  qwen25_7b   — Qwen/Qwen2.5-7B-Instruct   (3584-dim, layer 16/32)
  llama31_8b  — meta-llama/Llama-3.1-8B-Instruct (4096-dim, layer 16/32)
  qwen25_32b  — Qwen/Qwen2.5-32B-Instruct   (5120-dim, layer 32/64, 4-bit)

Outputs (in lm_output/phrase_level/):
  {model}_hiddens_phrase.npy   [N_events, hidden_dim]

Usage:
  python extract/extract_alt_lm.py --model qwen25_7b
  python extract/extract_alt_lm.py --model llama31_8b
  python extract/extract_alt_lm.py --model qwen25_32b   # needs bitsandbytes
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
import time
import numpy as np
import torch
from pathlib import Path

# Add parent dir so we can import phrase_bank
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phrase_bank import PHRASE_BANK

# ── Model configs ────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "qwen25_7b": {
        "hf_id":      "Qwen/Qwen2.5-7B-Instruct",
        "layer":      16,
        "hidden_dim": 3584,
        "quantize":   False,
    },
    "llama31_8b": {
        "hf_id":      "meta-llama/Llama-3.1-8B-Instruct",
        "layer":      16,
        "hidden_dim": 4096,
        "quantize":   False,
    },
    "qwen25_32b": {
        "hf_id":      "Qwen/Qwen2.5-32B-Instruct",
        "layer":      32,
        "hidden_dim": 5120,
        "quantize":   True,
    },
}

# ── Config ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = "lm_output/phrase_level"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Build event list ─────────────────────────────────────────────────────────

events = []
for concept, pairs in PHRASE_BANK.items():
    for i, (phrase, image_query) in enumerate(pairs):
        events.append({
            "concept":     concept,
            "phrase":      phrase,
            "image_query": image_query,
            "event_id":    f"{concept}__{i}",
        })

N = len(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Which model to extract from")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    model_name = args.model
    hf_id = cfg["hf_id"]
    layer = cfg["layer"]
    hidden_dim = cfg["hidden_dim"]

    print(f"{'='*60}")
    print(f"Extracting: {model_name} ({hf_id})")
    print(f"Layer: {layer}, Hidden dim: {hidden_dim}")
    print(f"Events: {N}, Concepts: {len(PHRASE_BANK)}")
    print(f"{'='*60}")

    # ── Load model ───────────────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"\nLoading {hf_id}...")

    tokenizer = AutoTokenizer.from_pretrained(hf_id)

    load_kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
    }

    if cfg["quantize"]:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        print("  Using 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)
    model.eval()
    print(f"  Model loaded on {DEVICE}")

    # ── Extraction ───────────────────────────────────────────────────────────

    def extract_phrase(phrase):
        """Extract hidden state at target layer, last token of the prompt."""
        messages = [
            {"role": "user",
             "content": f"Describe the physical dynamics of the following event: {phrase}"}
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[layer]  # [1, seq_len, hidden_dim]
        return hidden[0, -1, :].float().cpu().numpy()

    hiddens = np.zeros((N, hidden_dim), dtype=np.float32)

    print(f"\nExtracting {N} events...\n")
    t_start = time.time()

    for i, event in enumerate(events):
        phrase = event["phrase"]
        concept = event["concept"]
        t0 = time.time()

        hiddens[i] = extract_phrase(phrase)
        elapsed = time.time() - t0

        if (i + 1) % 10 == 0 or i < 5:
            eta = (time.time() - t_start) / (i + 1) * (N - i - 1)
            print(f"  [{i+1:3d}/{N}] {concept:12s} | {phrase[:50]:<50s} "
                  f"({elapsed:.1f}s, ETA {eta/60:.0f}m)")

    # ── Save ─────────────────────────────────────────────────────────────────

    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_hiddens_phrase.npy")
    np.save(out_path, hiddens)

    total_time = (time.time() - t_start) / 60
    print(f"\nSaved: {out_path}  {hiddens.shape}")
    print(f"Total time: {total_time:.1f} min")


if __name__ == "__main__":
    main()
