"""
DeepX v0.7: Gated DeltaNet-2 Hyperloop Backbone + ColBERT Head.

Architecture:
  Begin(4 NarrowA) → Phase1×2 [WideA + NarrowA×4] → Phase2×4 [NarrowB×4 + WideB]
  → End(1 WideB) = 35 compute passes

Unique layers: 9 (4 begin + 4 shared cores + 1 end)
Per-loop differentiation: LoRA + RoDE

Outputs:
  1. Single vector (1536-d) via attention pooling
  2. Token vectors (T × 128-d) via ColBERT head

Weight Init: ~95% from Gemma 4 E2B via direct copy + SVD LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple, Union

from config import DeepXConfig
from .hybrid_layer import (
    DeepXLayer, make_narrow_a_layer, make_narrow_b_layer,
    make_wide_a_layer, make_wide_b_layer,
)
from .hyperloop import HyperloopPhase
from .utils import RMSNorm, RoDE

logger = logging.getLogger(__name__)


class ColBERTHead(nn.Module):
    """Projects hidden states to low-dimensional token vectors for MaxSim."""

    def __init__(self, hidden_size: int, colbert_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(hidden_size, colbert_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        token_embeds = self.linear(hidden_states)
        token_embeds = F.normalize(token_embeds, p=2, dim=-1)
        if attention_mask is not None:
            token_embeds = token_embeds * attention_mask.unsqueeze(-1).to(token_embeds.dtype)
        return token_embeds


class DeepXEmbeddingModel(nn.Module):
    """
    DeepX v0.7 Backbone — Gated DeltaNet-2 Hyperloop + ColBERT.
    
    Receives hidden_states from external frozen token embedding.
    """

    def __init__(self, config: DeepXConfig):
        super().__init__()
        self.config = config

        # ═══ 1. Begin Block: 4 unique NarrowA layers (direct copy from Gemma 0-3) ═══
        self.begin_blocks = nn.ModuleList([
            make_narrow_a_layer(config, layer_idx=i)
            for i in range(config.begin_layers)
        ])

        # ═══ 2. Phase1 Loop: 2 iterations × [WideA + NarrowA×4] ═══
        self.shared_narrow_a = make_narrow_a_layer(config, layer_idx=config.begin_layers)
        self.shared_wide_a = make_wide_a_layer(config, layer_idx=config.begin_layers + 1)

        # Attach RoDE to shared cores
        if config.use_rode:
            self.shared_narrow_a.self_attn._rode = RoDE(
                dim=config.depth_rotary_dims, num_loops=config.phase1_loops
            )
            self.shared_wide_a.self_attn._rode = RoDE(
                dim=config.depth_rotary_dims, num_loops=config.phase1_loops
            )

        self.phase1 = HyperloopPhase(
            config=config,
            shared_narrow=self.shared_narrow_a,
            shared_wide=self.shared_wide_a,
            num_loops=config.phase1_loops,
            narrow_num_heads=config.narrow_a_heads,
            narrow_kv_heads=config.narrow_a_kv_heads,
            narrow_head_dim=config.narrow_a_head_dim,
            narrow_intermediate=config.narrow_a_intermediate,
            wide_num_heads=config.wide_a_heads,
            wide_kv_heads=config.wide_a_kv_heads,
            wide_head_dim=config.wide_a_head_dim,
            wide_intermediate=config.wide_a_intermediate,
            wide_first=True,  # [WideA, NarrowA×4]
        )

        # ═══ 3. Phase2 Loop: 4 iterations × [NarrowB×4 + WideB] ═══
        self.shared_narrow_b = make_narrow_b_layer(config, layer_idx=config.begin_layers + 2)
        self.shared_wide_b = make_wide_b_layer(config, layer_idx=config.begin_layers + 3)

        if config.use_rode:
            self.shared_narrow_b.self_attn._rode = RoDE(
                dim=config.depth_rotary_dims, num_loops=config.phase2_loops
            )
            self.shared_wide_b.self_attn._rode = RoDE(
                dim=config.depth_rotary_dims, num_loops=config.phase2_loops
            )

        self.phase2 = HyperloopPhase(
            config=config,
            shared_narrow=self.shared_narrow_b,
            shared_wide=self.shared_wide_b,
            num_loops=config.phase2_loops,
            narrow_num_heads=config.narrow_b_heads,
            narrow_kv_heads=config.narrow_b_kv_heads,
            narrow_head_dim=config.narrow_b_head_dim,
            narrow_intermediate=config.narrow_b_intermediate,
            wide_num_heads=config.wide_b_heads,
            wide_kv_heads=config.wide_b_kv_heads,
            wide_head_dim=config.wide_b_head_dim,
            wide_intermediate=config.wide_b_intermediate,
            wide_first=False,  # [NarrowB×4, WideB]
        )

        # ═══ 4. End Block: 1 unique WideB layer (layer 34 direct copy) ═══
        self.end_block = make_wide_b_layer(config, layer_idx=config.begin_layers + 4)

        # ═══ Final norm ═══
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # ═══ Output Head 1: Attention Pooling ═══
        self.pooling_strategy = config.pooling_strategy
        if config.pooling_strategy == "attention":
            self.pool_query = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

        # ═══ Output Head 2: ColBERT ═══
        self.use_colbert = config.use_colbert
        if config.use_colbert:
            self.colbert_head = ColBERTHead(config.hidden_size, config.colbert_dim)

        self.to(config.torch_dtype)

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.pooling_strategy == "attention":
            scale = hidden_states.shape[-1] ** -0.5
            scores = (hidden_states * self.pool_query).sum(dim=-1) * scale
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float("-inf"))
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)
            return (hidden_states * weights).sum(dim=1)
        # Mean pooling fallback
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return hidden_states.mean(dim=1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
        truncate_dim: Optional[int] = None,
        return_colbert: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = hidden_states.shape
        position_ids = torch.arange(T, device=hidden_states.device).unsqueeze(0).expand(B, -1)

        # 1. Begin blocks (4 unique NarrowA)
        for layer in self.begin_blocks:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        # 2. Phase1 loop: 2 × [WideA + NarrowA×4] = 10 passes
        hidden_states = self.phase1(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        # 3. Phase2 loop: 4 × [NarrowB×4 + WideB] = 20 passes
        # Model parallel: move tensors to phase2 device if different
        phase2_device = next(self.phase2.parameters()).device
        if hidden_states.device != phase2_device:
            hidden_states = hidden_states.to(phase2_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(phase2_device)
            position_ids = position_ids.to(phase2_device)
        hidden_states = self.phase2(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        # Move attention_mask to same device as hidden_states (for model parallel)
        if attention_mask is not None and attention_mask.device != hidden_states.device:
            attention_mask = attention_mask.to(hidden_states.device)
            position_ids = position_ids.to(hidden_states.device)

        # 4. End block (1 unique WideB)
        hidden_states = self.end_block(hidden_states, attention_mask=attention_mask, position_ids=position_ids)

        # 5. Final norm
        hidden_states = self.norm(hidden_states)

        # --- Output 1: Single vector ---
        single_embed = self._pool(hidden_states, attention_mask)
        if truncate_dim is not None:
            single_embed = single_embed[:, :truncate_dim]
        if normalize:
            single_embed = F.normalize(single_embed, p=2, dim=-1)

        # --- Output 2: ColBERT token vectors ---
        if return_colbert and self.use_colbert:
            token_embeds = self.colbert_head(hidden_states, attention_mask)
            return single_embed, token_embeds

        return single_embed

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"backbone_total": total, "trainable": trainable}
