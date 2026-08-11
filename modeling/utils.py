import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(input_dtype)


class SwiGLUMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, lora_deltas: dict = None) -> torch.Tensor:
        if lora_deltas:
            gate = F.linear(x, self.gate_proj.weight + lora_deltas.get("gate_proj", 0))
            up = F.linear(x, self.up_proj.weight + lora_deltas.get("up_proj", 0))
            down_weight = self.down_proj.weight + lora_deltas.get("down_proj", 0)
            return F.linear(F.silu(gate) * up, down_weight)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float = 10000.0, max_position: int = 2048
) -> float:
    """Find correction dimension for YaRN interpolation."""
    return (dim * math.log(max_position / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def _yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: float = 10000.0, max_position: int = 2048
) -> Tuple[int, int]:
    """Find the range of dimensions to apply YaRN correction."""
    low = math.floor(_yarn_find_correction_dim(low_rot, dim, base, max_position))
    high = math.ceil(_yarn_find_correction_dim(high_rot, dim, base, max_position))
    return max(low, 0), min(high, dim - 1)


def _yarn_linear_ramp_mask(low: int, high: int, dim: int, dtype: torch.dtype) -> torch.Tensor:
    """Create linear ramp mask for smooth interpolation between dimensions."""
    if low == high:
        high += 0.001
    linear_func = (torch.arange(dim, dtype=dtype) - low) / (high - low)
    return linear_func.clamp(0, 1)


class YaRNRotaryEmbedding(nn.Module):
    """
    RoPE with YaRN (Yet another RoPE extensioN) scaling.
    
    Same approach as Gemma 4 E2B:
    - Base theta = 1,000,000
    - YaRN scaling for context extension to 128K
    - Splits dimensions into 3 regions:
      1. Low freq dims: apply NTK-aware interpolation
      2. Medium freq dims: smooth ramp between interpolation and extrapolation  
      3. High freq dims: no scaling (extrapolation)
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 131072,
        base: float = 1000000.0,
        scaling_factor: float = 4.0,
        original_max_position: int = 32768,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.original_max_position = original_max_position
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow

        self._build_yarn_cache()

    def _build_yarn_cache(self):
        """Compute YaRN-adjusted inverse frequencies."""
        dim = self.dim
        # Standard RoPE inverse frequencies
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        # YaRN correction
        low, high = _yarn_find_correction_range(
            self.beta_slow, self.beta_fast, dim, self.base, self.original_max_position
        )
        inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2, torch.float32)

        # Interpolated frequencies (for extending context)
        inv_freq_interpolated = inv_freq / self.scaling_factor

        # Blend: high freq dims keep original, low freq dims get interpolated
        inv_freq_yarn = inv_freq_interpolated * (1 - inv_freq_mask) + inv_freq * inv_freq_mask

        self.register_buffer("inv_freq", inv_freq_yarn, persistent=False)

        # Attention scaling factor (magnitude correction)
        self.attn_scale = 0.1 * math.log(self.scaling_factor) + 1.0

    def forward(
        self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, H, T, D) — used only for device/dtype
            position_ids: (B, T) or None (auto-generate 0..T-1)
        Returns:
            cos, sin: (B, T, D) in same dtype as x
        """
        B, H, T, D = x.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # Compute in float32 for precision, cast output to match x
        inv_freq = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)

        emb = torch.cat([freqs, freqs], dim=-1)

        cos = (emb.cos() * self.attn_scale).to(x.dtype)
        sin = (emb.sin() * self.attn_scale).to(x.dtype)

        return cos, sin


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE rotation to Q and K.
    
    Args:
        q: (B, H, T, D)
        k: (B, H_kv, T, D)
        cos: (B, T, D)
        sin: (B, T, D)
    Returns:
        q_rotated, k_rotated: same shapes
    """
    # (B, T, D) → (B, 1, T, D) for broadcasting with heads
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_rotated = (q * cos) + (_rotate_half(q) * sin)
    k_rotated = (k * cos) + (_rotate_half(k) * sin)

    return q_rotated, k_rotated


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: [x1, x2] → [-x2, x1]"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_depth_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply depth rotation to a subset of dimensions.
    Args:
        x: (B, T, D)
        cos, sin: (1, 1, d_rode) where d_rode is the number of rotated dimensions.
    """
    d_rode = cos.shape[-1]
    x_rode = x[..., :d_rode]
    x_rest = x[..., d_rode:]
    
    # Apply rotation to the first d_rode dimensions
    x_rotated = (x_rode * cos) + (_rotate_half(x_rode) * sin)
    
    return torch.cat([x_rotated, x_rest], dim=-1)


class RoDE(nn.Module):
    """
    Rotary Depth Embedding (RoDE).
    Provides a depth signal for shared weights in Hyperloop.
    """
    def __init__(self, dim: int, num_loops: int, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.num_loops = num_loops
        self.base = base
        
        # Pre-compute sin/cos for each loop index
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # (num_loops, dim // 2)
        loop_ids = torch.arange(num_loops).float()
        freqs = loop_ids.unsqueeze(-1) * inv_freq.unsqueeze(0)
        
        # (num_loops, dim)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, loop_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pick the pre-computed sin/cos for the current loop index
        # Shape: (1, 1, dim) for broadcasting
        cos = self.cos[loop_idx].view(1, 1, -1).to(x.dtype)
        sin = self.sin[loop_idx].view(1, 1, -1).to(x.dtype)
        return cos, sin
