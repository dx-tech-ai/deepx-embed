"""
DeepX v0.6 Full Pipeline.

Combines:
  1. Frozen Gemma 4 E2B token embedding (loaded from pretrained/gemma4_e2b_embed.pt)
  2. Pure GLA Hyperloop backbone + ColBERT head (PureGLAEmbeddingModel)

Outputs:
  - encode()         → single vector (1536-d) for fast ANN retrieval
  - encode_colbert() → token vectors (T × 128-d) for MaxSim reranking
  - encode_multi()   → both single + token vectors in one forward pass

Weight Init: ~90% of backbone can be copied from Gemma 4 E2B.
"""

import torch
import torch.nn as nn
import logging
import dataclasses
from typing import Optional, Tuple
from pathlib import Path

from config import HybridEmbeddingConfig
from .embedding_model import DeepXEmbeddingModel

logger = logging.getLogger(__name__)


class DeepXPipeline(nn.Module):
    """
    Full DeepX v0.6 embedding pipeline.

    Token embedding is frozen and loaded from a pre-extracted file.
    Only the backbone (PureGLAEmbeddingModel) is trained.
    """

    def __init__(
        self,
        config: HybridEmbeddingConfig,
        embed_path: str = "pretrained/gemma4_e2b_embed.pt",
    ):
        super().__init__()
        self.config = config

        # --- Frozen Token Embedding (Gemma 4 E2B) ---
        embed_path = Path(embed_path)
        if not embed_path.exists():
            raise FileNotFoundError(
                f"Token embedding not found at '{embed_path}'.\n"
                f"Please run: python scripts/extract_gemma_embedding.py"
            )

        logger.info(f"Loading frozen token embedding from {embed_path} ...")
        weight = torch.load(embed_path, weights_only=True)

        assert weight.shape == (config.vocab_size, config.hidden_size), (
            f"Embedding shape mismatch: expected ({config.vocab_size}, {config.hidden_size}), "
            f"got {tuple(weight.shape)}. Check config.hidden_size matches E2B."
        )

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.token_embedding.weight.data = weight.to(config.torch_dtype)
        self.token_embedding.requires_grad_(False)
        logger.info(f"Token embedding frozen. Shape: {weight.shape}, dtype: {config.torch_dtype}")

        # --- Trainable Backbone ---
        self.backbone = DeepXEmbeddingModel(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
        return_colbert: bool = False,
    ):
        """
        Full forward pass.
        
        Returns:
            If return_colbert=False: single_embed (B, D)
            If return_colbert=True: (single_embed (B, D), token_embeds (B, T, colbert_dim))
        """
        with torch.no_grad():
            hidden_states = self.token_embedding(input_ids)

        return self.backbone(
            hidden_states,
            attention_mask=attention_mask,
            normalize=normalize,
            truncate_dim=truncate_dim,
            return_colbert=return_colbert,
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        truncate_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Single vector encoding for fast ANN retrieval."""
        with torch.no_grad():
            return self.forward(input_ids, attention_mask, normalize=True, truncate_dim=truncate_dim)

    def encode_colbert(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ColBERT encoding — returns both single vector and token vectors.
        
        Returns:
            single_embed: (B, 1536) for coarse retrieval
            token_embeds: (B, T, 128) for MaxSim reranking
        """
        with torch.no_grad():
            return self.forward(input_ids, attention_mask, normalize=True, return_colbert=True)

    def encode_multi(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        truncate_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for encode_colbert with optional truncation on single vector."""
        with torch.no_grad():
            hidden_states = self.token_embedding(input_ids)
        return self.backbone(
            hidden_states,
            attention_mask=attention_mask,
            normalize=True,
            truncate_dim=truncate_dim,
            return_colbert=True,
        )

    def freeze_embedder(self):
        """Ensure token embedding stays frozen."""
        self.token_embedding.requires_grad_(False)

    def unfreeze_embedder(self):
        """Unfreeze token embedding for fine-tuning (use with very small LR ~1e-6)."""
        self.token_embedding.requires_grad_(True)
        logger.warning("Token embedding UNFROZEN. Use lr ~1e-6.")

    def count_parameters(self) -> dict:
        embed_params = self.token_embedding.weight.numel()
        backbone_counts = self.backbone.count_parameters()
        return {
            "embedding_frozen": embed_params,
            "backbone_trainable": backbone_counts["trainable"],
            "backbone_total": backbone_counts["backbone_total"],
            "grand_total": embed_params + backbone_counts["backbone_total"],
        }

    def save_backbone(self, path: str) -> None:
        """Save only the trained backbone weights."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.backbone.state_dict(),
            "config": dataclasses.asdict(self.config),
            "version": "0.7",
            "architecture": "gdn2_hyperloop_colbert",
            "embed_source": "gemma4_e2b",
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "colbert_dim": self.config.colbert_dim,
        }, out)
        size_mb = out.stat().st_size / 1024 / 1024
        logger.info(f"Backbone saved to {out} ({size_mb:.1f} MB)")

    @classmethod
    def from_pretrained(
        cls,
        config: HybridEmbeddingConfig,
        embed_path: str,
        backbone_path: str,
    ) -> "DeepXPipeline":
        """Load pipeline from 2 .pt files for deployment."""
        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone weights not found at '{backbone_path}'.")

        pipeline = cls(config, embed_path=embed_path)

        logger.info(f"Loading backbone from {backbone_path} ...")
        checkpoint = torch.load(backbone_path, weights_only=True, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        pipeline.backbone.load_state_dict(state_dict)
        pipeline.backbone.to(config.torch_dtype)
        logger.info("Backbone loaded successfully.")

        return pipeline
