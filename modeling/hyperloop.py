"""
Hyperloop Segment v1.0 — Per-Loop LoRA + RoDE for Gated DeltaNet-2.

Each loop iteration = [1 Wide + 4 Narrow] layers, with:
  1. RoDE: Rotary depth signal on Q/K inside attention
  2. Per-loop LoRA on all projections (Q/K/V/O + MLP gate/up/down)
  3. Stochastic depth for robustness
  4. Gradient checkpointing per iteration (saves ~5x VRAM during training)

Two phase types:
  Phase1: WideA(16h, MLP6144) + NarrowA×4(8h, MLP6144)
  Phase2: NarrowB×4(8h, MLP12288) + WideB(16h, MLP12288)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Dict


class PerLoopLoRA(nn.Module):
    """Per-loop low-rank adaptation for all projections."""

    def __init__(self, num_loops: int, proj_shapes: Dict[str, tuple], rank: int = 16):
        super().__init__()
        self.num_loops = num_loops
        self.rank = rank
        self.proj_names = list(proj_shapes.keys())

        for name, (out_dim, in_dim) in proj_shapes.items():
            a_tensors = nn.ParameterList([
                nn.Parameter(torch.zeros(out_dim, rank))
                for _ in range(num_loops)
            ])
            b_tensors = nn.ParameterList([
                nn.Parameter(torch.zeros(rank, in_dim))
                for _ in range(num_loops)
            ])
            setattr(self, f"lora_A_{name}", a_tensors)
            setattr(self, f"lora_B_{name}", b_tensors)

    def get_delta(self, proj_name: str, loop_idx: int) -> torch.Tensor:
        A = getattr(self, f"lora_A_{proj_name}")[loop_idx]
        B = getattr(self, f"lora_B_{proj_name}")[loop_idx]
        return A @ B


class HyperloopPhase(nn.Module):
    """
    Multi-iteration loop with [Wide + Narrow×4] pattern per iteration.
    
    Each iteration:
      1. Forward through shared_wide (1 pass)
      2. Forward through shared_narrow × 4 (4 passes)
    Total per iteration: 5 passes
    """

    def __init__(
        self,
        config,
        shared_narrow: nn.Module,
        shared_wide: nn.Module,
        num_loops: int,
        narrow_num_heads: int,
        narrow_kv_heads: int,
        narrow_head_dim: int,
        narrow_intermediate: int,
        wide_num_heads: int,
        wide_kv_heads: int,
        wide_head_dim: int,
        wide_intermediate: int,
        wide_first: bool = True,  # True: [Wide, Narrow×4], False: [Narrow×4, Wide]
        use_grad_checkpoint: bool = True,  # Gradient checkpointing per iteration
    ):
        super().__init__()
        self.shared_narrow = shared_narrow
        self.shared_wide = shared_wide
        self.num_loops = num_loops
        self.drop_path_rate = config.drop_path_rate
        self.wide_first = wide_first
        self.use_grad_checkpoint = use_grad_checkpoint
        H = config.hidden_size

        # Per-loop LoRA for narrow layers (applied 4× per iteration)
        narrow_proj_shapes = {
            "q_proj": (narrow_num_heads * narrow_head_dim, H),
            "k_proj": (narrow_kv_heads * narrow_head_dim, H),
            "v_proj": (narrow_kv_heads * narrow_head_dim, H),
            "o_proj": (H, narrow_num_heads * narrow_head_dim),
            "gate_proj": (narrow_intermediate, H),
            "up_proj": (narrow_intermediate, H),
            "down_proj": (H, narrow_intermediate),
        }
        # Per-loop LoRA for wide layers (applied 1× per iteration)
        wide_proj_shapes = {
            "q_proj": (wide_num_heads * wide_head_dim, H),
            "k_proj": (wide_kv_heads * wide_head_dim, H),
            "v_proj": (wide_kv_heads * wide_head_dim, H),
            "o_proj": (H, wide_num_heads * wide_head_dim),
            "gate_proj": (wide_intermediate, H),
            "up_proj": (wide_intermediate, H),
            "down_proj": (H, wide_intermediate),
        }

        # LoRA for each iteration (narrow layers share LoRA within iteration)
        self.narrow_lora = PerLoopLoRA(num_loops, narrow_proj_shapes, config.lora_rank)
        self.wide_lora = PerLoopLoRA(num_loops, wide_proj_shapes, config.lora_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for i in range(self.num_loops):
            # Stochastic depth
            if self.training and self.drop_path_rate > 0:
                drop_prob = self.drop_path_rate * (i + 1) / self.num_loops
                if torch.rand(1).item() < drop_prob:
                    continue

            # Get LoRA deltas for this iteration
            narrow_deltas = {
                name: self.narrow_lora.get_delta(name, i)
                for name in self.narrow_lora.proj_names
            }
            wide_deltas = {
                name: self.wide_lora.get_delta(name, i)
                for name in self.wide_lora.proj_names
            }

            if self.wide_first:
                # [Wide, Narrow×4]
                hidden_states = self._forward_iteration_wide_first(
                    hidden_states, attention_mask, position_ids, i, wide_deltas, narrow_deltas
                )
            else:
                # [Narrow×4, Wide]
                hidden_states = self._forward_iteration_narrow_first(
                    hidden_states, attention_mask, position_ids, i, wide_deltas, narrow_deltas
                )

        return hidden_states

    def _forward_iteration_wide_first(self, hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas):
        """Single loop iteration: [Wide, Narrow×4]. Wrapped for gradient checkpointing."""
        if self.training and self.use_grad_checkpoint:
            return grad_checkpoint(
                self._wide_first_fn,
                hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas,
                use_reentrant=False,
            )
        return self._wide_first_fn(hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas)

    def _forward_iteration_narrow_first(self, hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas):
        """Single loop iteration: [Narrow×4, Wide]. Wrapped for gradient checkpointing."""
        if self.training and self.use_grad_checkpoint:
            return grad_checkpoint(
                self._narrow_first_fn,
                hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas,
                use_reentrant=False,
            )
        return self._narrow_first_fn(hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas)

    def _wide_first_fn(self, hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas):
        hidden_states = self.shared_wide(
            hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, loop_idx=loop_idx, lora_deltas=wide_deltas,
        )
        for _ in range(4):
            hidden_states = self.shared_narrow(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, loop_idx=loop_idx, lora_deltas=narrow_deltas,
            )
        return hidden_states

    def _narrow_first_fn(self, hidden_states, attention_mask, position_ids, loop_idx, wide_deltas, narrow_deltas):
        for _ in range(4):
            hidden_states = self.shared_narrow(
                hidden_states, attention_mask=attention_mask,
                position_ids=position_ids, loop_idx=loop_idx, lora_deltas=narrow_deltas,
            )
        hidden_states = self.shared_wide(
            hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, loop_idx=loop_idx, lora_deltas=wide_deltas,
        )
        return hidden_states
