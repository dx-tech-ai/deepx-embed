"""
DeepX v1.0 Full Pipeline.

Combines:
  1. Frozen token embedding (pruned 186K vocab, loaded from HuggingFace)
  2. GDN-2 Hyperloop backbone + ColBERT head

Outputs:
  - encode()         → single vector (1536-d) for fast ANN retrieval
  - encode_colbert() → token vectors (T × 128-d) for MaxSim reranking
  - encode_multi()   → both single + token vectors in one forward pass
"""

import torch
import torch.nn as nn
import logging
import dataclasses
from typing import Optional, Tuple
from pathlib import Path

from config import DeepXConfig
from .embedding_model import DeepXEmbeddingModel

logger = logging.getLogger(__name__)


class DeepXPipeline(nn.Module):
    """
    Full DeepX embedding pipeline.

    Token embedding is frozen. Only the backbone is trained.
    """

    def __init__(
        self,
        config: DeepXConfig,
        embed_path: str = None,
    ):
        super().__init__()
        self.config = config

        # --- Token Embedding (frozen) ---
        if embed_path is not None:
            embed_path = Path(embed_path)
            if not embed_path.exists():
                raise FileNotFoundError(
                    f"Token embedding not found at '{embed_path}'.\n"
                    f"Use DeepXEmbed.from_pretrained() to load from HuggingFace."
                )

            logger.info(f"Loading frozen token embedding from {embed_path} ...")
            weight = torch.load(embed_path, weights_only=True)
            if isinstance(weight, dict) and "weight" in weight:
                weight = weight["weight"]

            self.token_embedding = nn.Embedding(weight.shape[0], weight.shape[1])
            self.token_embedding.weight.data = weight.to(config.torch_dtype)
            self.token_embedding.requires_grad_(False)
            logger.info(f"Token embedding frozen. Shape: {weight.shape}, dtype: {config.torch_dtype}")
        else:
            # Placeholder — will be overridden by caller (e.g. DeepXEmbed.from_pretrained)
            self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
            self.token_embedding.requires_grad_(False)

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
        """Encode both single vector and token vectors in one forward pass."""
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
            "version": "1.0",
            "architecture": "gdn2_hyperloop_colbert",
            "hidden_size": self.config.hidden_size,
            "vocab_size": self.config.vocab_size,
            "colbert_dim": self.config.colbert_dim,
        }, out)
        size_mb = out.stat().st_size / 1024 / 1024
        logger.info(f"Backbone saved to {out} ({size_mb:.1f} MB)")

    @classmethod
    def from_pretrained(
        cls,
        config: DeepXConfig,
        embed_path: str,
        backbone_path: str,
    ) -> "DeepXPipeline":
        """Load pipeline from embedding + backbone .pt files."""
        backbone_path = Path(backbone_path)
        if not backbone_path.exists():
            raise FileNotFoundError(f"Backbone weights not found at '{backbone_path}'.")

        pipeline = cls(config, embed_path=embed_path)

        logger.info(f"Loading backbone from {backbone_path} ...")
        checkpoint = torch.load(backbone_path, weights_only=True, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        pipeline.backbone.load_state_dict(state_dict, strict=False)
        pipeline.backbone.to(config.torch_dtype)
        logger.info("Backbone loaded successfully.")

        return pipeline
