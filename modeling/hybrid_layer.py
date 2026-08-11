"""
DeepX v0.7 Layer: GDN-2 Attention + SwiGLU MLP with pre-norm residual.

Supports both narrow (8h) and wide (16h) configurations.
"""

import torch
import torch.nn as nn
from typing import Optional

from .gdn2_attention import GatedDeltaNet2Attention
from .utils import RMSNorm, SwiGLUMLP


class DeepXLayer(nn.Module):
    """Gated DeltaNet-2 Attention + SwiGLU MLP with pre-norm residual."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        config,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.input_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.self_attn = GatedDeltaNet2Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            chunk_size=config.chunk_size,
            attention_dropout=config.attention_dropout,
            layer_idx=layer_idx,
            use_dual_path=config.use_dual_path,
            softmax_init_alpha=config.softmax_init_alpha,
            use_short_conv=config.use_short_conv,
            conv_kernel_size=config.conv_kernel_size,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_original_max_position=config.rope_original_max_position,
            rope_beta_fast=config.rope_beta_fast,
            rope_beta_slow=config.rope_beta_slow,
        )
        self.post_attn_norm = RMSNorm(hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        loop_idx: Optional[int] = None,
        lora_deltas: Optional[dict] = None,
    ) -> torch.Tensor:
        # Pre-norm + Attention + Residual
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(
            x,
            attention_mask=attention_mask,
            position_ids=position_ids,
            loop_idx=loop_idx,
            lora_deltas=lora_deltas,
        )
        x = residual + x

        # Pre-norm + MLP + Residual
        residual = x
        x = self.post_attn_norm(x)
        x = self.mlp(x, lora_deltas=lora_deltas)
        x = residual + x

        return x


def make_narrow_a_layer(config, layer_idx: int = 0) -> DeepXLayer:
    """Create NarrowA layer (8h, 1kv, MLP 6144)."""
    return DeepXLayer(
        hidden_size=config.hidden_size,
        num_heads=config.narrow_a_heads,
        num_kv_heads=config.narrow_a_kv_heads,
        head_dim=config.narrow_a_head_dim,
        intermediate_size=config.narrow_a_intermediate,
        config=config,
        layer_idx=layer_idx,
    )


def make_narrow_b_layer(config, layer_idx: int = 0) -> DeepXLayer:
    """Create NarrowB layer (8h, 1kv, MLP 12288)."""
    return DeepXLayer(
        hidden_size=config.hidden_size,
        num_heads=config.narrow_b_heads,
        num_kv_heads=config.narrow_b_kv_heads,
        head_dim=config.narrow_b_head_dim,
        intermediate_size=config.narrow_b_intermediate,
        config=config,
        layer_idx=layer_idx,
    )


def make_wide_a_layer(config, layer_idx: int = 0) -> DeepXLayer:
    """Create WideA layer (16h, 2kv, MLP 6144)."""
    return DeepXLayer(
        hidden_size=config.hidden_size,
        num_heads=config.wide_a_heads,
        num_kv_heads=config.wide_a_kv_heads,
        head_dim=config.wide_a_head_dim,
        intermediate_size=config.wide_a_intermediate,
        config=config,
        layer_idx=layer_idx,
    )


def make_wide_b_layer(config, layer_idx: int = 0) -> DeepXLayer:
    """Create WideB layer (16h, 2kv, MLP 12288)."""
    return DeepXLayer(
        hidden_size=config.hidden_size,
        num_heads=config.wide_b_heads,
        num_kv_heads=config.wide_b_kv_heads,
        head_dim=config.wide_b_head_dim,
        intermediate_size=config.wide_b_intermediate,
        config=config,
        layer_idx=layer_idx,
    )
