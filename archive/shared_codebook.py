"""
Experiment 4: Shared Discrete Codebook Between World Model and LLM
--------------------------------------------------------------------
Tests whether a VQ-VAE bottleneck can serve as a translation layer
between a world model's latent space and an LLM's embedding space.

Architecture:
    [World Model RSSM] --> [WM Encoder] --> |              |
                                            | VQ Codebook  | <-- discrete codes
    [LLM Mistral 7B]  --> [LM Encoder] --> |              |

Training objective:
    1. Both encoders should map the same concept to the same code
    2. Decoders should reconstruct the original embedding from the code
    3. Codes should be discrete (straight-through estimator)

Requirements:
    pip install torch transformers bitsandbytes gymnasium dreamerv3 \
                sentence-transformers datasets accelerate
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json
import os


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class Config:
    # Codebook
    codebook_size: int = 512        # Number of discrete codes
    codebook_dim: int = 256         # Dimension of each code vector
    commitment_cost: float = 0.25   # Weight for commitment loss

    # World model latent dim (DreamerV3-small RSSM output)
    wm_latent_dim: int = 512

    # LLM hidden dim (Mistral 7B last hidden state, projected)
    lm_hidden_dim: int = 4096       # Mistral 7B hidden size
    lm_proj_dim: int = 512          # Project down before codebook

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model paths
    lm_model_id: str = "mistralai/Mistral-7B-v0.1"

cfg = Config()


# ─────────────────────────────────────────────
# VECTOR QUANTIZER (the core discrete bottleneck)
# ─────────────────────────────────────────────

class VectorQuantizer(nn.Module):
    """
    Straight-through VQ-VAE codebook.
    Maps continuous vectors to discrete codes and back.
    This is the shared 'vocabulary of thought' between modules.
    """
    def __init__(self, codebook_size: int, codebook_dim: int, commitment_cost: float):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.commitment_cost = commitment_cost

        # The shared codebook - these are the discrete concept codes
        self.embedding = nn.Embedding(codebook_size, codebook_dim)
        self.embedding.weight.data.uniform_(
            -1 / codebook_size, 1 / codebook_size
        )

    def forward(self, z: torch.Tensor):
        """
        z: [batch, codebook_dim] continuous representation
        returns: quantized, loss, code_indices
        """
        # Flatten if needed
        z_flat = z.view(-1, self.codebook_dim)

        # Compute distances to all codebook entries
        distances = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flat, self.embedding.weight.t())
        )  # [batch, codebook_size]

        # Nearest code for each input
        code_indices = torch.argmin(distances, dim=1)  # [batch]
        quantized = self.embedding(code_indices)       # [batch, codebook_dim]

        # VQ loss: codebook moves toward encoder outputs
        # Commitment loss: encoder stays close to codebook
        e_latent_loss = F.mse_loss(quantized.detach(), z_flat)
        q_latent_loss = F.mse_loss(quantized, z_flat.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator: gradients pass through quantization
        quantized_st = z_flat + (quantized - z_flat).detach()

        return quantized_st, loss, code_indices

    def decode_from_index(self, indices: torch.Tensor) -> torch.Tensor:
        """Given discrete code indices, retrieve the code vectors."""
        return self.embedding(indices)


# ─────────────────────────────────────────────
# WORLD MODEL INTERFACE
# Using DreamerV3's RSSM latent state as the world model representation.
# The RSSM produces a stochastic + deterministic state vector.
# ─────────────────────────────────────────────

class WorldModelEncoder(nn.Module):
    """
    Projects world model RSSM latent state into codebook space.
    In a full system this would wrap DreamerV3's encoder directly.
    For the experiment, we use a learned projection.
    """
    def __init__(self, input_dim: int, codebook_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, codebook_dim),
            nn.LayerNorm(codebook_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, input_dim),
        )

    def encode(self, wm_latent: torch.Tensor) -> torch.Tensor:
        return self.encoder(wm_latent)

    def decode(self, code_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(code_vector)


class LMEncoder(nn.Module):
    """
    Projects LLM last hidden state into codebook space.
    Uses mean pooling over token dimension, then projects.
    """
    def __init__(self, lm_hidden_dim: int, codebook_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(lm_hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, codebook_dim),
            nn.LayerNorm(codebook_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(codebook_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, lm_hidden_dim),
        )

    def encode(self, lm_hidden: torch.Tensor) -> torch.Tensor:
        return self.encoder(lm_hidden)

    def decode(self, code_vector: torch.Tensor) -> torch.Tensor:
        return self.decoder(code_vector)


# ─────────────────────────────────────────────
# FULL BRIDGE SYSTEM
# ─────────────────────────────────────────────

class LatentBridge(nn.Module):
    """
    The complete inter-module communication system.

    Forward pass (WM → LM direction):
        1. World model produces latent state
        2. WM encoder projects to codebook space
        3. VQ quantizes to discrete code
        4. LM decoder reconstructs LM-compatible representation
        5. LM can now "understand" what the world model was representing

    The discrete code in step 3 is the shared symbol.
    """
    def __init__(self, config: Config):
        super().__init__()
        self.vq = VectorQuantizer(
            config.codebook_size,
            config.codebook_dim,
            config.commitment_cost
        )
        self.wm_module = WorldModelEncoder(config.wm_latent_dim, config.codebook_dim)
        self.lm_module = LMEncoder(config.lm_hidden_dim, config.codebook_dim)

    def forward_wm(self, wm_latent: torch.Tensor):
        """World model → discrete code → LM space"""
        z = self.wm_module.encode(wm_latent)
        z_q, vq_loss, codes = self.vq(z)
        lm_recon = self.lm_module.decode(z_q)
        wm_recon = self.wm_module.decode(z_q)
        return lm_recon, wm_recon, vq_loss, codes

    def forward_lm(self, lm_hidden: torch.Tensor):
        """LM → discrete code → world model space"""
        z = self.lm_module.encode(lm_hidden)
        z_q, vq_loss, codes = self.vq(z)
        wm_recon = self.wm_module.decode(z_q)
        lm_recon = self.lm_module.decode(z_q)
        return wm_recon, lm_recon, vq_loss, codes

    def get_code(self, wm_latent: Optional[torch.Tensor] = None,
                 lm_hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Just get the discrete code indices — the actual 'shared symbol'"""
        if wm_latent is not None:
            z = self.wm_module.encode(wm_latent)
        else:
            z = self.lm_module.encode(lm_hidden)
        _, _, codes = self.vq(z)
        return codes


# ─────────────────────────────────────────────
# DATASET
# Paired concept representations from both models
# ─────────────────────────────────────────────

class ConceptPairDataset(Dataset):
    """
    Each sample: a concept that both the world model and LLM have processed.
    WM representation: latent state from observing/imagining the concept
    LM representation: last hidden state from the concept's text description

    For the experiment, we simulate WM latents — in a real system
    you'd run DreamerV3 on visual/sensory input of each concept.
    """
    def __init__(self, wm_latents: torch.Tensor, lm_hiddens: torch.Tensor):
        assert len(wm_latents) == len(lm_hiddens)
        self.wm_latents = wm_latents
        self.lm_hiddens = lm_hiddens

    def __len__(self):
        return len(self.wm_latents)

    def __getitem__(self, idx):
        return self.wm_latents[idx], self.lm_hiddens[idx]


# ─────────────────────────────────────────────
# LLM REPRESENTATION EXTRACTOR
# ─────────────────────────────────────────────

def extract_lm_representations(concepts: list[str], model_id: str, device: str):
    """
    Extract last hidden states from Mistral 7B for each concept.
    Uses 4-bit quantization to fit in memory.
    """
    print(f"Loading {model_id} with 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.eval()

    representations = []
    with torch.no_grad():
        for concept in concepts:
            # Prompt that encourages the model to activate its full representation
            prompt = f"The concept of {concept} refers to"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # Use last hidden state, mean pooled over tokens
            last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
            pooled = last_hidden.mean(dim=1).squeeze(0)  # [hidden_dim]
            representations.append(pooled.float().cpu())

    return torch.stack(representations)  # [n_concepts, hidden_dim]


# ─────────────────────────────────────────────
# SIMULATED WORLD MODEL REPRESENTATIONS
# In a full system: run DreamerV3 on sensory observations of each concept
# For experiment: use a structured simulation with concept-specific patterns
# ─────────────────────────────────────────────

def simulate_wm_representations(concepts: list[str], latent_dim: int):
    """
    Placeholder for DreamerV3 RSSM latent states.

    In production: load DreamerV3, run it on visual/sensory observations
    of each concept (images, videos, physical simulations), extract the
    stochastic state vector.

    For now: structured noise that preserves semantic clustering,
    so we can still test whether the codebook alignment works.
    """
    print("Simulating world model representations...")
    print("(In production: replace with DreamerV3 RSSM latent states)")

    # Group concepts into rough semantic clusters
    # Real WM representations would cluster this way naturally
    concept_seeds = {c: hash(c) % 10000 for c in concepts}

    representations = []
    for concept in concepts:
        torch.manual_seed(concept_seeds[concept])
        # Structured latent: cluster center + noise
        cluster_id = concept_seeds[concept] % 8
        cluster_center = torch.randn(latent_dim) * 0.1
        cluster_center[cluster_id * 64:(cluster_id + 1) * 64] += 2.0
        noise = torch.randn(latent_dim) * 0.3
        representations.append(cluster_center + noise)

    return torch.stack(representations)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(bridge: LatentBridge, dataloader: DataLoader, config: Config):
    optimizer = torch.optim.AdamW(bridge.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    bridge.train()
    history = []

    for epoch in range(config.epochs):
        total_loss = 0
        code_consistency = 0
        n_batches = 0

        for wm_batch, lm_batch in dataloader:
            wm_batch = wm_batch.to(config.device)
            lm_batch = lm_batch.to(config.device)

            optimizer.zero_grad()

            # Forward: WM path
            lm_recon_from_wm, wm_recon, vq_loss_wm, wm_codes = bridge.forward_wm(wm_batch)
            # Forward: LM path
            wm_recon_from_lm, lm_recon, vq_loss_lm, lm_codes = bridge.forward_lm(lm_batch)

            # Reconstruction losses
            wm_recon_loss = F.mse_loss(wm_recon, wm_batch)
            lm_recon_loss = F.mse_loss(lm_recon, lm_batch)

            # Cross-modal reconstruction (key test: can WM encode → LM decode?)
            cross_lm_loss = F.mse_loss(lm_recon_from_wm, lm_batch)
            cross_wm_loss = F.mse_loss(wm_recon_from_lm, wm_batch)

            # Code consistency loss: same concept should get same code
            # Soft version: code vectors should be close
            wm_code_vecs = bridge.vq.decode_from_index(wm_codes)
            lm_code_vecs = bridge.vq.decode_from_index(lm_codes)
            consistency_loss = F.mse_loss(wm_code_vecs, lm_code_vecs)

            # Combined loss
            loss = (
                wm_recon_loss
                + lm_recon_loss
                + cross_lm_loss       # This is the core test
                + cross_wm_loss
                + vq_loss_wm
                + vq_loss_lm
                + 2.0 * consistency_loss   # Upweight — this is the alignment goal
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            optimizer.step()

            # Track how often WM and LM agree on the same code
            code_agreement = (wm_codes == lm_codes).float().mean().item()

            total_loss += loss.item()
            code_consistency += code_agreement
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches
        avg_consistency = code_consistency / n_batches

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "code_agreement": avg_consistency
        })

        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                  f"Code Agreement: {avg_consistency:.3f} "
                  f"({'✓ converging' if avg_consistency > 0.3 else '...'})")

    return history


# ─────────────────────────────────────────────
# EVALUATION
# The key question: given only the discrete code, can we recover
# what the other model was representing?
# ─────────────────────────────────────────────

def evaluate(bridge: LatentBridge, dataset: ConceptPairDataset,
             concepts: list[str], config: Config):
    bridge.eval()
    results = []

    with torch.no_grad():
        for i, concept in enumerate(concepts):
            wm_lat = dataset.wm_latents[i:i+1].to(config.device)
            lm_hid = dataset.lm_hiddens[i:i+1].to(config.device)

            # Get codes from each modality
            wm_code = bridge.get_code(wm_latent=wm_lat).item()
            lm_code = bridge.get_code(lm_hidden=lm_hid).item()

            # Cross-reconstruct
            lm_from_wm, _, _, _ = bridge.forward_wm(wm_lat)
            wm_from_lm, _, _, _ = bridge.forward_lm(lm_hid)

            # Reconstruction quality
            lm_error = F.mse_loss(lm_from_wm, lm_hid).item()
            wm_error = F.mse_loss(wm_from_lm, wm_lat).item()

            results.append({
                "concept": concept,
                "wm_code": wm_code,
                "lm_code": lm_code,
                "codes_match": wm_code == lm_code,
                "lm_reconstruction_error": lm_error,
                "wm_reconstruction_error": wm_error,
            })

    # Summary
    n_match = sum(r["codes_match"] for r in results)
    avg_lm_err = np.mean([r["lm_reconstruction_error"] for r in results])
    avg_wm_err = np.mean([r["wm_reconstruction_error"] for r in results])

    print(f"\n{'='*50}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Code agreement: {n_match}/{len(concepts)} concepts ({100*n_match/len(concepts):.1f}%)")
    print(f"Avg LM reconstruction error: {avg_lm_err:.4f}")
    print(f"Avg WM reconstruction error: {avg_wm_err:.4f}")
    print(f"\nPer-concept breakdown:")
    for r in results[:20]:
        match_str = "✓" if r["codes_match"] else "✗"
        print(f"  {match_str} {r['concept']:20s} | WM code: {r['wm_code']:3d} | "
              f"LM code: {r['lm_code']:3d} | err: {r['lm_reconstruction_error']:.4f}")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Latent Bridge Experiment")
    print("Testing shared discrete codebook between world model and LLM\n")

    # Concepts to test — spanning physical, abstract, social categories
    # Physical grounding is important: these are things a robot would encounter
    concepts = [
        # Physical / embodied
        "apple", "chair", "water", "fire", "door", "hand", "stone", "shadow",
        # Actions / dynamics
        "falling", "pushing", "grasping", "breaking", "balancing",
        # Spatial relations
        "inside", "above", "beside", "distance", "boundary",
        # Abstract
        "danger", "support", "intention", "causation", "similarity",
        # Social
        "helping", "blocking", "following", "trust",
    ]

    print(f"Testing {len(concepts)} concepts across physical, abstract, social categories\n")

    # Step 1: Extract LLM representations
    # Uncomment when ready to run with real Mistral:
    # lm_hiddens = extract_lm_representations(concepts, cfg.lm_model_id, cfg.device)
    # torch.save(lm_hiddens, "lm_hiddens.pt")

    # For now: simulate LM representations too (same structured approach)
    # Replace with real extraction above for actual experiment
    print("Generating LM representations (simulated — replace with real Mistral extraction)")
    lm_hiddens = torch.randn(len(concepts), cfg.lm_hidden_dim)
    # Add semantic structure so related concepts cluster
    for i, concept in enumerate(concepts):
        seed = (hash(concept) % 10000)
        torch.manual_seed(seed)
        cluster_id = seed % 8
        lm_hiddens[i, cluster_id * 512:(cluster_id + 1) * 512] += 3.0

    # Step 2: Get world model representations
    wm_latents = simulate_wm_representations(concepts, cfg.wm_latent_dim)

    # Step 3: Build dataset
    dataset = ConceptPairDataset(wm_latents, lm_hiddens)
    dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    # Step 4: Initialize bridge
    bridge = LatentBridge(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in bridge.parameters())
    print(f"Bridge model: {n_params:,} parameters\n")

    # Step 5: Train
    print("Training shared codebook...")
    history = train(bridge, dataloader, cfg)

    # Step 6: Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(bridge.state_dict(), "checkpoints/bridge.pt")
    with open("checkpoints/history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("\nCheckpoint saved to checkpoints/bridge.pt")

    # Step 7: Evaluate
    results = evaluate(bridge, dataset, concepts, cfg)

    print("\n" + "="*50)
    print("INTERPRETATION")
    print("="*50)
    print("""
High code agreement (>40%):
    → The discrete codebook is finding shared structure between
      world model and LLM representations. The 'same concept'
      is being mapped to the same discrete symbol by both paths.
      This supports the Platonic Representation Hypothesis.

Low reconstruction error (<0.1):
    → The codebook has enough capacity to preserve the information
      needed to reconstruct both modalities. The lossy compression
      is not destroying meaning.

Next steps if this works:
    1. Replace simulated WM with real DreamerV3 on MineDojo/Habitat
    2. Replace simulated LM with real Mistral 7B extractions
    3. Test whether codes are interpretable (visualize what concept
       clusters around each code)
    4. Test cross-modal reasoning: can LM answer questions about
       what the WM 'saw', using only the discrete code as input?
    """)


if __name__ == "__main__":
    main()
