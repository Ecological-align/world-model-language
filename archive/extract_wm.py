"""
World Model Representation Extractor
-------------------------------------
Extracts RSSM latent state vectors from DreamerV3 for each concept.

TWO MODES:
    simulation  -- runs immediately, no downloads, structured noise
                   with semantic clustering. Use to verify pipeline.
    real        -- requires DreamerV3 + MineDojo/Habitat environment.
                   This is the actual experiment.

WHAT THE RSSM LATENT ENCODES:
    DreamerV3's Recurrent State Space Model produces a state vector:
        h_t  -- deterministic recurrent state (GRU hidden)
        z_t  -- stochastic discrete latent (32 categories x 32 classes)

    Concatenated: [h_t, flatten(z_t)] = [512, 1024] -> 1536 dim typically.

    This encodes the model's *prediction* of future observations given
    current context -- it's fundamentally dynamic and causal, NOT a
    static feature descriptor like a vision encoder.

    That distinction is the whole point of this experiment.
"""

import numpy as np
from pathlib import Path


def extract_real(concepts: list[str], env_name: str = "minedojo") -> np.ndarray:
    """
    Extract RSSM latent states from a trained DreamerV3 model.

    For each concept:
        1. Load a short video / observation sequence related to the concept
        2. Run through DreamerV3's encoder + RSSM
        3. Extract the final [h_t, z_t] concatenated state
        4. Average across a few observations for stability

    Requirements:
        pip install dreamerv3  (or the torch port: pip install dreamer-pytorch)
        pip install minedojo   (for Minecraft-like environments)
        A pretrained DreamerV3 checkpoint (~400MB)

    Downloads:
        DreamerV3 checkpoint: https://github.com/danijar/dreamerv3
        MineDojo: automatically downloads on first run (~2GB)

    VRAM: ~6GB for DreamerV3-small
    """
    try:
        import dreamerv3
    except ImportError:
        raise ImportError(
            "dreamerv3 not installed.\n"
            "Run: pip install dreamerv3\n"
            "Or use mode='simulation' to test the pipeline first."
        )

    raise NotImplementedError(
        "Real DreamerV3 extraction requires:\n"
        "  1. A trained checkpoint (see README for download link)\n"
        "  2. An observation source per concept (images or video)\n"
        "  3. Setting up the RSSM forward pass\n\n"
        "See extract_real_template() below for the scaffolding.\n"
        "Use extract_simulated() to verify the pipeline first."
    )


def extract_real_template():
    """
    Template for real extraction -- fill in once you have DreamerV3 running.
    This shows the exact API calls needed.
    """
    template = '''
    import dreamerv3
    import numpy as np
    import torch

    # Load pretrained DreamerV3
    config = dreamerv3.Config().update(dreamerv3.configs["small"])
    agent = dreamerv3.Agent(config, ...)
    checkpoint = dreamerv3.Checkpoint()
    checkpoint.agent = agent
    checkpoint.load("path/to/checkpoint.ckpt", keys=["agent"])

    # For each concept, get observations and extract latent
    latents = []
    for concept in concepts:
        # Option A: MineDojo -- spawn objects/scenarios in Minecraft
        # Option B: Habitat -- navigate to objects in realistic scenes
        # Option C: Images -- run through DreamerV3 encoder only (weaker)

        obs_sequence = get_observations_for_concept(concept)  # [T, H, W, C]

        # Run through RSSM
        state = agent.initial_state(batch_size=1)
        for obs in obs_sequence:
            embed = agent.encoder(obs)
            post, prior = agent.rssm.observe(embed, action, state)
            state = post

        # Concatenate deterministic + stochastic state
        h = post["deter"]           # [1, 512]
        z = post["stoch"].flatten() # [1, 32*32] = [1, 1024]
        latent = torch.cat([h, z], dim=-1).squeeze(0)  # [1536]
        latents.append(latent)

    return torch.stack(latents)  # [N_concepts, 1536]
    '''
    print(template)


def extract_simulated(concepts: list[str], latent_dim: int = 1536,
                      seed: int = 42) -> np.ndarray:
    """
    Simulation mode: generates structured representations that mimic
    what DreamerV3 would produce, with realistic semantic clustering.

    Structure imposed:
        - Concepts in the same semantic category cluster together
        - Physical concepts have stronger "grounding" signal
          (larger cluster separation) -- mimics embodied learning
        - Abstract concepts are more diffuse -- mimics poor WM signal
          for non-physical concepts
        - Within-category similarity is higher than cross-category

    This is NOT a replacement for real extraction -- it verifies
    the pipeline logic and sets a baseline for what random agreement
    looks like vs. real agreement.
    """
    from concepts import CONCEPT_CATEGORIES, CONCEPTS

    print("[WM] Running in SIMULATION mode")
    print("[WM] Replace with real DreamerV3 extraction for actual experiment")
    print(f"[WM] Generating {len(concepts)} latent vectors (dim={latent_dim})")

    # Category cluster centers -- each category gets a distinct region
    # of latent space, with physical concepts most separated (stronger signal)
    category_list = list(CONCEPTS.keys())
    category_separation = {
        "physical": 3.0,   # strong grounding signal
        "actions": 2.5,    # temporal/causal -- also strong in WM
        "spatial": 2.0,    # geometric -- moderate
        "social": 1.5,     # agent-relative -- weaker in WM
        "abstract": 0.8,   # minimal sensory grounding -- weak in WM
    }

    rng = np.random.default_rng(seed)
    # One cluster center per category
    cluster_centers = {
        cat: rng.standard_normal(latent_dim)
        for cat in category_list
    }

    latents = []
    for concept in concepts:
        cat = CONCEPT_CATEGORIES[concept]
        separation = category_separation.get(cat, 1.5)
        center = cluster_centers[cat]

        concept_seed = int(abs(hash(concept)) % 100000)
        c_rng = np.random.default_rng(concept_seed)
        concept_offset = c_rng.standard_normal(latent_dim) * 0.4
        noise = c_rng.standard_normal(latent_dim) * 0.3

        latent = separation * center + concept_offset + noise
        latents.append(latent)

    latents = np.stack(latents)  # [N, latent_dim]
    print(f"[WM] Done. Shape: {latents.shape}")
    return latents


def extract(concepts: list[str], mode: str = "simulation",
            latent_dim: int = 1536, **kwargs) -> np.ndarray:
    """Main entry point. mode: 'simulation' or 'real'"""
    if mode == "simulation":
        return extract_simulated(concepts, latent_dim)
    elif mode == "real":
        return extract_real(concepts, **kwargs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'simulation' or 'real'.")


if __name__ == "__main__":
    from concepts import ALL_CONCEPTS
    latents = extract(ALL_CONCEPTS, mode="simulation")
    print(f"WM latents: {latents.shape}")
    print(f"Sample norms: {latents.norm(dim=-1)[:5]}")
