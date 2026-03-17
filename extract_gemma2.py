"""
extract_gemma2.py
=================

Extracts phrase-level hidden states from Gemma-2-9B (google/gemma-2-9b-it)
at the midpoint layer (~layer 21 of 42), matching the extraction protocol
used for all other LLMs.

Gemma-2-9B is Google's architecture family — a different lineage from
Mistral, Qwen, and Llama. Adding it to the architecture control resolves
the 3/4 majority to 4/5, providing stronger evidence that the MAE vs
VideoMAE gap is not LLM-architecture-specific.

Requirements:
  - ~18GB VRAM in float16 (fits on RTX 5090)
  - HuggingFace access token if model is gated
    (set HF_TOKEN env var or pass --token flag)
  - transformers >= 4.38 for Gemma 2 support

Output:
  lm_output/phrase_level/gemma2_hiddens_phrase.npy  [323, 3584]

Run from repo root:
  PYTHONPATH=. .venv/Scripts/python.exe extract_gemma2.py
  
  If gated:
  PYTHONPATH=. HF_TOKEN=hf_xxx .venv/Scripts/python.exe extract_gemma2.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import os, json, argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID  = "google/gemma-2-9b-it"
DATA_DIR  = "lm_output/phrase_level"
OUT_FILE  = os.path.join(DATA_DIR, "gemma2_hiddens_phrase.npy")
BATCH     = 8

def get_layer_idx(model, override=None):
    """Return layer index: override if specified, else midpoint."""
    n = model.config.num_hidden_layers
    if override is not None:
        assert 0 <= override < n, f"--layer {override} out of range (model has {n} layers)"
        print(f"  Model has {n} layers → using layer {override} (explicit override)")
        return override
    mid = n // 2
    print(f"  Model has {n} layers → using layer {mid} (midpoint)")
    return mid

def extract_hidden(model, tokenizer, texts, layer_idx):
    hiddens = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        # hidden_states[layer_idx+1] because index 0 = embedding layer
        hs = out.hidden_states[layer_idx + 1]
        # mean-pool over non-padding tokens
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (hs * mask).sum(1) / mask.sum(1)
        hiddens.append(pooled.cpu().float().numpy())
        if (i // BATCH + 1) % 5 == 0:
            print(f"    {i+len(batch)}/{len(texts)}")
    return np.concatenate(hiddens, axis=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index to extract from (default: midpoint). "
                             "Diagnostic recommends layer 41 for Gemma-2-9B.")
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    # Load event index
    with open(os.path.join(DATA_DIR, "event_index.json")) as f:
        raw = json.load(f)
    event_index = raw if isinstance(raw, list) else raw["events"]
    phrases = [e["phrase"] for e in event_index]
    print(f"Loaded {len(phrases)} phrases")

    # Load model
    print(f"\nLoading {MODEL_ID}...")
    kwargs = dict(
        torch_dtype=torch.float16,
        device_map="auto",
        output_hidden_states=True,
    )
    if args.token:
        kwargs["token"] = args.token

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **({"token": args.token} if args.token else {}))
        model     = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except OSError as e:
        print(f"\nFailed to load {MODEL_ID}: {e}")
        print("\nIf the model is gated, set HF_TOKEN or pass --token hf_xxx")
        print("You may also need: huggingface-cli login")
        sys.exit(1)

    model.eval()
    layer_idx = get_layer_idx(model, override=args.layer)
    print(f"  Hidden dim: {model.config.hidden_size}")
    print(f"  Extracting from layer {layer_idx}...")

    print(f"\nExtracting {len(phrases)} phrase embeddings in batches of {BATCH}...")
    hiddens = extract_hidden(model, tokenizer, phrases, layer_idx)
    print(f"Shape: {hiddens.shape}")

    np.save(OUT_FILE, hiddens)
    print(f"\nSaved to {OUT_FILE}")
    print(f"  dtype:  {hiddens.dtype}")
    print(f"  shape:  {hiddens.shape}")
    print(f"  range:  [{hiddens.min():.3f}, {hiddens.max():.3f}]")


if __name__ == "__main__":
    main()
