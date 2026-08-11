"""
DeepX v0.7 — Gated DeltaNet-2 Hyperloop + ColBERT Configuration.

Architecture:
  Begin(4 NarrowA) → Phase1 Loop×2 [WideA + NarrowA×4] → Transition(WideA)
  → Phase2 Loop×4 [WideB + NarrowB×4] → End(WideB) = 35 passes

Gemma 4 E2B layers:
  - NarrowA: 8 heads, 1 KV head, head_dim=256, MLP=6144  (layers 0-3, 5-8, 10-13)
  - NarrowB: 8 heads, 1 KV head, head_dim=256, MLP=12288 (layers 15-18, 20-23, 25-28, 30-33)
  - WideA:   16 heads, 2 KV heads, head_dim=256, MLP=6144 (layers 4, 9, 14)
  - WideB:   16 heads, 2 KV heads, head_dim=256, MLP=12288 (layers 19, 24, 29, 34)

Attention: Gated DeltaNet-2 (dual-path: softmax + delta rule)
Weight Init: Copy Q/K/V/O + MLP from Gemma 4 E2B → ~95% pretrained.
"""

from dataclasses import dataclass
from typing import Optional, List
import torch


@dataclass
class NarrowLayerConfig:
    """Config for narrow layers (8 heads, 1 KV head)."""
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 6144  # NarrowA default


@dataclass
class WideLayerConfig:
    """Config for wide layers (16 heads, 2 KV heads)."""
    num_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 256
    intermediate_size: int = 6144  # WideA default


@dataclass
class DeepXConfig:
    """
    DeepX v0.7 — Gated DeltaNet-2 Hyperloop + ColBERT.
    
    35 compute passes matching Gemma 4 E2B depth.
    9 unique layer parameter sets (4 shared cores + 4 begin + 1 end).
    """
    
    # --- Base dimensions (Gemma 4 E2B compatible) ---
    vocab_size: int = 262144
    hidden_size: int = 1536
    max_position_embeddings: int = 131072

    # --- Precision ---
    dtype: str = "float16"

    # --- Architecture: 35 passes ---
    # Begin(4 NarrowA) + Phase1×2×[1 WideA + 4 NarrowA] + Transition(WideA)
    # + Phase2×4×[1 WideB + 4 NarrowB] + End(1 WideB) = 4+10+1+20+1 = 36
    # Correction: fold transition into phase1 last wide → 4+10+20+1 = 35
    begin_layers: int = 4
    phase1_loops: int = 2           # each iter = [1 WideA + 4 NarrowA] = 5 layers
    phase2_loops: int = 4           # each iter = [4 NarrowB + 1 WideB] = 5 layers  
    end_layers: int = 1             # 1 unique WideB (layer 34)
    # Total: 4 + 2×5 + 4×5 + 1 = 35 ✓

    # --- Layer configs ---
    # NarrowA: layers 0-3, 5-8, 10-13 (8h, 1kv, MLP 6144)
    narrow_a_heads: int = 8
    narrow_a_kv_heads: int = 1
    narrow_a_head_dim: int = 256
    narrow_a_intermediate: int = 6144
    
    # NarrowB: layers 15-18, 20-23, 25-28, 30-33 (8h, 1kv, MLP 12288)
    narrow_b_heads: int = 8
    narrow_b_kv_heads: int = 1
    narrow_b_head_dim: int = 256
    narrow_b_intermediate: int = 12288
    
    # WideA: layers 4, 9, 14 (16h, 2kv, MLP 6144)
    wide_a_heads: int = 16
    wide_a_kv_heads: int = 2
    wide_a_head_dim: int = 256
    wide_a_intermediate: int = 6144
    
    # WideB: layers 19, 24, 29, 34 (16h, 2kv, MLP 12288)
    wide_b_heads: int = 16
    wide_b_kv_heads: int = 2
    wide_b_head_dim: int = 256
    wide_b_intermediate: int = 12288

    # --- Gated DeltaNet-2 attention config ---
    chunk_size: int = 64            # chunk size for parallel computation
    use_dual_path: bool = True      # GDN-2 enabled (FLA kernel)
    softmax_init_alpha: float = 2.0 # sigmoid(2) ≈ 0.88 → GDN-2 contributes ~12% initially
    use_short_conv: bool = True     # 1D conv on Q,K before attention
    conv_kernel_size: int = 4       # causal conv1d kernel

    # --- Depth Mechanism (RoDE) ---
    use_rode: bool = True
    depth_rotary_dims: int = 16

    # --- Per-loop LoRA ---
    lora_rank: int = 16
    drop_path_rate: float = 0.05

    # --- RoPE (YaRN for context extension) ---
    rope_theta: float = 1000000.0
    rope_scaling_factor: float = 32.0
    rope_original_max_position: int = 4096
    rope_beta_fast: int = 32
    rope_beta_slow: int = 1

    # --- Norm ---
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0

    # --- Embedding output ---
    pooling_strategy: str = "attention"
    matryoshka_dims: list = None

    # --- ColBERT head ---
    colbert_dim: int = 128
    use_colbert: bool = True

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[self.dtype]

    def __post_init__(self):
        if self.matryoshka_dims is None:
            self.matryoshka_dims = [256, 512, 768, 1024, 1536]


# Backward compatibility alias
HybridEmbeddingConfig = DeepXConfig
