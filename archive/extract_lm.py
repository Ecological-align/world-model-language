"""
LLM Hidden State Extractor
----------------------------
Extracts mean-pooled last hidden states from Mistral 7B for each concept.

TWO MODES:
    simulation  -- runs immediately, structured noise with semantic
                   clustering. Use to verify pipeline.
    real        -- requires Mistral 7B download (~14GB). This is the
                   actual experiment.

DESIGN CHOICES:
    Why last hidden state, not embeddings?
        The embedding layer (layer 0) is close to a lookup table.
        The last hidden state has been processed by all 32 transformer
        layers and encodes rich contextual/semantic structure.

    Why mean pooling over tokens?
        We want a single vector per concept. Mean pooling is more
        stable than CLS token for decoder-only models like Mistral.

    Why this prompt template?
        "The concept of {X} refers to:"
        This forces the model to activate its representation of X
        as a topic, not as a word in a sentence. The colon at the
        end primes the model's internal state for elaboration,
        which tends to produce richer hidden states.
"""

import numpy as np
import numpy.linalg as la
from pathlib import Path


PROMPT_TEMPLATE = "The concept of {concept} refers to:"

# Alternative prompts to try if results are noisy:
PROMPT_VARIANTS = [
    "The concept of {concept} refers to:",       # default
    "{concept} is a",                             # shorter, more direct
    "When I think about {concept}, I imagine:",   # more embodied framing
    "The physical properties of {concept} include:", # stress-tests grounding
]


def extract_real(concepts: list[str],
                 model_id: str = "mistralai/Mistral-7B-v0.1",
                 prompt_template: str = PROMPT_TEMPLATE,
                 device: str = "cuda",
                 cache_path: str = "lm_hiddens.pt") -> np.ndarray:
    """
    Extract last hidden states from Mistral 7B using 4-bit quantization.

    Requirements:
        pip install transformers bitsandbytes accelerate
        ~14GB disk space for model download
        ~4GB VRAM (4-bit quantized)

    First run: downloads model from HuggingFace (~14GB, one-time)
    Subsequent runs: loads from HuggingFace cache

    The extracted representations are cached to disk so you only
    run this once.
    """
    # Check cache first
    cache = Path(cache_path)
    if cache.exists():
        print(f"[LM] Loading cached representations from {cache_path}")
        data = torch.load(cache_path)
        if data["concepts"] == concepts:
            return data["hiddens"]
        else:
            print("[LM] Cache concept list mismatch, re-extracting...")

    try:
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
    except ImportError:
        raise ImportError(
            "transformers not installed.\n"
            "Run: pip install transformers bitsandbytes accelerate\n"
            "Or use mode='simulation' to test the pipeline first."
        )

    print(f"[LM] Loading {model_id} (4-bit quantized)...")
    print("[LM] First run: downloading ~14GB. This is one-time only.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    print(f"[LM] Extracting hidden states for {len(concepts)} concepts...")
    hiddens = []

    with torch.no_grad():
        for i, concept in enumerate(concepts):
            prompt = prompt_template.format(concept=concept)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=64,
            ).to(device)

            outputs = model(**inputs, output_hidden_states=True)

            # Last hidden state: [1, seq_len, hidden_dim]
            last_hidden = outputs.hidden_states[-1]

            # Mean pool over token dimension -> [hidden_dim]
            pooled = last_hidden.mean(dim=1).squeeze(0).float().cpu()
            hiddens.append(pooled)

            if (i + 1) % 20 == 0:
                print(f"[LM]   {i+1}/{len(concepts)} done")

    hiddens_tensor = torch.stack(hiddens)  # [N, hidden_dim]

    # Cache for future runs
    torch.save({"concepts": concepts, "hiddens": hiddens_tensor}, cache_path)
    print(f"[LM] Cached to {cache_path}")
    print(f"[LM] Done. Shape: {hiddens_tensor.shape}")

    return hiddens_tensor


def extract_simulated(concepts: list[str], hidden_dim: int = 4096,
                      seed: int = 99) -> np.ndarray:
    """
    Simulation mode: generates structured LLM-like representations.

    Key difference from WM simulation:
        - Abstract concepts have STRONGER clustering (LLM's home turf)
        - Physical/action concepts have WEAKER separation (text is lossy
          about physical properties)

    This creates a realistic mismatch pattern: WM and LLM representations
    should partially agree (shared reality structure) but diverge on
    abstract vs physical concepts.

    That divergence pattern -- if replicated with real models -- would be
    the finding.
    """
    from concepts import CONCEPT_CATEGORIES, CONCEPTS

    print("[LM] Running in SIMULATION mode")
    print("[LM] Replace with real Mistral extraction for actual experiment")
    print(f"[LM] Generating {len(concepts)} hidden state vectors (dim={hidden_dim})")

    # Opposite to WM: abstract concepts are better separated in LLM space
    category_separation = {
        "physical": 1.2,   # text is lossy about physical properties
        "actions": 1.5,    # some causal/temporal signal in text
        "spatial": 1.8,    # prepositions are well-represented in text
        "social": 2.2,     # rich social language signal
        "abstract": 3.0,   # LLM's home turf -- strong signal
    }

    rng = np.random.default_rng(seed)
    category_list = list(CONCEPTS.keys())
    cluster_centers = {}
    for cat in category_list:
        c_rng = np.random.default_rng(abs(hash(cat)) % 100000)
        cluster_centers[cat] = c_rng.standard_normal(hidden_dim)

    hiddens = []
    for concept in concepts:
        cat = CONCEPT_CATEGORIES[concept]
        separation = category_separation.get(cat, 1.5)
        center = cluster_centers[cat]

        concept_seed = int(abs(hash(concept + "_lm")) % 100000)
        c_rng = np.random.default_rng(concept_seed)
        concept_offset = c_rng.standard_normal(hidden_dim) * 0.4
        noise = c_rng.standard_normal(hidden_dim) * 0.3

        hidden = separation * center + concept_offset + noise
        hiddens.append(hidden)

    hiddens_arr = np.stack(hiddens)
    print(f"[LM] Done. Shape: {hiddens_arr.shape}")
    return hiddens_arr


def extract(concepts: list[str], mode: str = "simulation",
            hidden_dim: int = 4096, **kwargs) -> np.ndarray:
    """Main entry point. mode: 'simulation' or 'real'"""
    if mode == "simulation":
        return extract_simulated(concepts, hidden_dim)
    elif mode == "real":
        return extract_real(concepts, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simulation' or 'real'.")


if __name__ == "__main__":
    from concepts import ALL_CONCEPTS
    hiddens = extract(ALL_CONCEPTS, mode="simulation")
    print(f"LM hiddens: {hiddens.shape}")
    print(f"Sample norms: {hiddens.norm(dim=-1)[:5]}")
