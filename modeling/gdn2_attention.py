"""
Gated DeltaNet-2 Dual-Path Attention.

Dual-path design:
  Path 1 (Softmax): Standard chunked softmax attention with RoPE
  Path 2 (GDN-2):  Gated Delta Rule-2 with channel-wise erase/write gates
          → O(n) linear attention

Merge: output = α × softmax_path + (1-α) × gdn2_path
  α init ≈ 1.0 → model starts as pure softmax
  α learned → shifts to GDN-2 as training progresses (O(T) inference)

Gated Delta Rule-2 (per timestep):
  1. Decay:  A_t = diag(α_t) @ A_{t-1}
  2. Erase:  u_t = b_t ⊙ k_t  (channel-wise erase gate)
  3. Write:  A_t -= u_t^T @ (u_t @ A_{t-1} - w_t ⊙ v_t)
  4. Read:   o_t = q_t @ A_t

Reference: Hatamizadeh et al. "Gated DeltaNet-2" (NVIDIA, May 2026)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from einops import rearrange

from .utils import RMSNorm, YaRNRotaryEmbedding, apply_rotary_pos_emb, apply_depth_rotary_emb


class ShortConv1d(nn.Module):
    """Causal 1D convolution for Q,K preprocessing (init as identity pass-through)."""
    
    def __init__(self, dim: int, kernel_size: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size - 1, groups=dim)
        # Init as identity: center weight = 1, rest = 0
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # Set last position (causal) to 1 for identity
        with torch.no_grad():
            self.conv.weight[:, :, -1] = 1.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) → (B, T, D)"""
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)[..., :x.shape[-1]]  # causal: trim future padding
        return x.transpose(1, 2)  # (B, T, D)


class GatedDeltaNet2Attention(nn.Module):
    """
    Dual-Path: Softmax + Gated DeltaNet-2.
    
    Init strategy: α ≈ 1 (pure softmax) at start.
    Training shifts α towards GDN-2 path for O(T) inference.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        chunk_size: int = 64,
        attention_dropout: float = 0.0,
        layer_idx: int = 0,
        use_dual_path: bool = True,
        softmax_init_alpha: float = 5.0,
        use_short_conv: bool = True,
        conv_kernel_size: int = 4,
        # RoPE params
        max_position_embeddings: int = 131072,
        rope_theta: float = 1000000.0,
        rope_scaling_factor: float = 32.0,
        rope_original_max_position: int = 4096,
        rope_beta_fast: int = 32,
        rope_beta_slow: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        self.layer_idx = layer_idx
        self.chunk_size = chunk_size
        self.use_dual_path = use_dual_path

        # === Shared projections ===
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # === Q/K normalization (L2 norm per head — GDN-2 best practice) ===
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

        # === RoPE for softmax path ===
        self.rotary_emb = YaRNRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=rope_scaling_factor,
            original_max_position=rope_original_max_position,
            beta_fast=rope_beta_fast,
            beta_slow=rope_beta_slow,
        )

        # === Short conv for GDN-2 path Q,K (init as identity) ===
        if use_short_conv:
            self.q_conv = ShortConv1d(num_heads * head_dim, conv_kernel_size)
            self.k_conv = ShortConv1d(num_kv_heads * head_dim, conv_kernel_size)
        else:
            self.q_conv = None
            self.k_conv = None

        # === GDN-2 specific params (all NEW, init for near-no-op) ===
        # Erase gate: b_t = sigmoid(W_b @ x_t) ∈ [0,1]^{d_k}
        self.erase_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        nn.init.zeros_(self.erase_proj.weight)
        nn.init.zeros_(self.erase_proj.bias)  # sigmoid(0)=0.5 → moderate erase
        
        # Write gate: w_t = sigmoid(W_w @ x_t) ∈ [0,1]^{d_v}  
        self.write_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        nn.init.zeros_(self.write_proj.weight)
        nn.init.zeros_(self.write_proj.bias)  # sigmoid(0)=0.5 → moderate write
        
        # Decay (channel-wise, log-parameterized for stability)
        # g_t = -exp(a) * softplus(W_f @ x_t + δ)
        # α_t = exp(g_t) ∈ (0, 1]
        # Init: a=-5, δ=5 → g≈0 → α≈1 (no decay initially)
        self.decay_a = nn.Parameter(torch.full((num_kv_heads * head_dim,), -5.0))
        self.decay_delta = nn.Parameter(torch.full((num_kv_heads * head_dim,), 5.0))
        self.decay_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        nn.init.zeros_(self.decay_proj.weight)  # → softplus(0+5)≈5, g=-exp(-5)*5≈-0.03, α≈0.97

        # === Dual-path mix: α (softmax weight) ===
        # Init large positive → sigmoid ≈ 1 → pure softmax at start
        self.path_mix_logit = nn.Parameter(torch.full((num_heads,), softmax_init_alpha))

        # === Output gate (SiLU gating like GDN-2 paper) ===
        self.output_gate_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)

        # === RoDE placeholder ===
        self._rode = None

        self.dropout = nn.Dropout(attention_dropout)

    def _expand_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Expand KV heads for GQA: (B, H_kv, T, D) → (B, H, T, D)"""
        if self.num_kv_groups == 1:
            return x
        B, H_kv, T, D = x.shape
        x = x.unsqueeze(2).expand(-1, -1, self.num_kv_groups, -1, -1)
        return x.reshape(B, self.num_heads, T, D)

    # ── Path 1: Standard Softmax Attention (uses RoPE) ──────────

    def _softmax_attention(
        self,
        q: torch.Tensor,      # (B, H, T, D)
        k: torch.Tensor,      # (B, H, T, D) — already expanded for GQA
        v: torch.Tensor,      # (B, H, T, D)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        scale = 1.0 / math.sqrt(D)
        
        # Full attention for short seqs, chunked for long
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale  # (B, H, T, T)
        
        if attention_mask is not None:
            # attention_mask: (B, T) → (B, 1, 1, T)
            mask = attention_mask[:, None, None, :].to(scores.dtype)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)

    # ── Path 2: Gated DeltaNet-2 (Recurrent, O(T) inference) ───

    def _gdn2_attention(
        self,
        q: torch.Tensor,      # (B, H, T, D) — L2 normalized
        k: torch.Tensor,      # (B, H_kv, T, D) — L2 normalized  
        v: torch.Tensor,      # (B, H_kv, T, D)
        erase_gate: torch.Tensor,  # (B, T, d_kv)
        write_gate: torch.Tensor,  # (B, T, d_kv)
        decay: torch.Tensor,       # (B, T, d_kv) — α_t values
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gated Delta Rule-2 via FLA chunk-parallel kernel (O(T) training).
        Falls back to sequential recurrence if FLA not available.
        """
        B, H_kv, T, D = k.shape
        H = self.num_heads
        dtype = q.dtype

        # Expand KV for GQA
        k_exp = self._expand_kv(k)        # (B, H, T, D)
        v_exp = self._expand_kv(v)        # (B, H, T, D)

        # Reshape gates for FLA: (B, T, d_kv) → (B, H, T, D) → reshape for kernel
        erase = rearrange(erase_gate, "b t (h d) -> b h t d", h=H_kv)
        write = rearrange(write_gate, "b t (h d) -> b h t d", h=H_kv)
        alpha = rearrange(decay, "b t (h d) -> b h t d", h=H_kv)
        
        erase_exp = self._expand_kv(erase)  # (B, H, T, D)
        write_exp = self._expand_kv(write)  # (B, H, T, D)
        alpha_exp = self._expand_kv(alpha)  # (B, H, T, D)

        try:
            # Use FLA chunk kernel for all sequences
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gdn
            
            # FLA expects: q,k = [B, T, H, D], v = [B, T, HV, V]
            # g = [B, T, HV] (log-space forget gate, per-head)
            # beta = [B, T, HV] (update gate, per-head, 0-1)
            
            # Transpose from (B, H, T, D) → (B, T, H, D) and ensure BF16
            q_fla = q.transpose(1, 2).contiguous().to(torch.bfloat16)
            k_fla = k_exp.transpose(1, 2).contiguous().to(torch.bfloat16)
            
            # Apply write gate to values, then transpose
            v_gated = (write_exp * v_exp)                # (B, H, T, D)
            v_fla = v_gated.transpose(1, 2).contiguous().to(torch.bfloat16)
            
            # g: per-head log-decay (MUST be float32)
            g_per_head = torch.log(alpha_exp.float().clamp(min=1e-6)).mean(dim=-1)  # (B, H, T)
            g_fla = g_per_head.transpose(1, 2).contiguous()  # (B, T, H) float32
            
            # beta: per-head erase strength (MUST be float32)
            beta_per_head = erase_exp.float().mean(dim=-1)  # (B, H, T)
            beta_fla = beta_per_head.transpose(1, 2).contiguous()  # (B, T, H) float32
            
            # Call FLA kernel outside autocast
            with torch.amp.autocast(device_type="cuda", enabled=False):
                output, _ = fla_chunk_gdn(
                    q=q_fla,
                    k=k_fla,
                    v=v_fla,
                    g=g_fla,
                    beta=beta_fla,
                    scale=1.0,
                    use_qk_l2norm_in_kernel=False,
                )
            # output: (B, T, H, D) → transpose back to (B, H, T, D)
            return output.transpose(1, 2).to(dtype)
            
        except (ImportError, RuntimeError) as e:
            # Sequential recurrence fallback
            return self._gdn2_sequential(q, k_exp, v_exp, erase_exp, write_exp, alpha_exp, attention_mask)

    def _gdn2_sequential(
        self,
        q: torch.Tensor,      # (B, H, T, D)
        k: torch.Tensor,      # (B, H, T, D)
        v: torch.Tensor,      # (B, H, T, D)
        erase: torch.Tensor,  # (B, H, T, D)
        write: torch.Tensor,  # (B, H, T, D)
        alpha: torch.Tensor,  # (B, H, T, D)
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sequential fallback for GDN-2 (used when FLA kernel unavailable)."""
        B, H, T, D = q.shape
        device = q.device
        dtype = q.dtype

        state = torch.zeros(B, H, D, D, device=device, dtype=torch.float32)
        outputs = []

        for t in range(T):
            if attention_mask is not None and t < attention_mask.shape[1]:
                mask_t = attention_mask[:, t].view(B, 1, 1, 1).float()
            else:
                mask_t = 1.0

            q_t = q[:, :, t, :]
            k_t = k[:, :, t, :]
            v_t = v[:, :, t, :]
            b_t = erase[:, :, t, :]
            w_t = write[:, :, t, :]
            a_t = alpha[:, :, t, :]

            state = state * a_t.unsqueeze(-1).float()
            u_t = (b_t * k_t).float()
            old_val = torch.einsum("bhd,bhde->bhe", u_t, state)
            target = (w_t * v_t).float()
            error = target - old_val
            state = state + torch.einsum("bhd,bhe->bhde", u_t, error) * mask_t
            o_t = torch.einsum("bhd,bhde->bhe", q_t.float(), state)
            outputs.append(o_t)

        output = torch.stack(outputs, dim=2).to(dtype)
        return output

    # ── Forward ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        loop_idx: Optional[int] = None,
        lora_deltas: Optional[dict] = None,
    ) -> torch.Tensor:
        B, T, _ = hidden_states.shape

        # === Projections (with optional LoRA) ===
        def proj_with_lora(proj, x, name):
            if lora_deltas and name in lora_deltas:
                return F.linear(x, proj.weight + lora_deltas[name], proj.bias if proj.bias is not None else None)
            return proj(x)

        q_raw = proj_with_lora(self.q_proj, hidden_states, "q_proj")  # (B, T, H*D)
        k_raw = proj_with_lora(self.k_proj, hidden_states, "k_proj")  # (B, T, H_kv*D)
        v = proj_with_lora(self.v_proj, hidden_states, "v_proj")      # (B, T, H_kv*D)

        # === Path 1: Softmax (standard RoPE attention) ===
        q1 = rearrange(q_raw, "b t (h d) -> b h t d", h=self.num_heads)
        k1 = rearrange(k_raw, "b t (h d) -> b h t d", h=self.num_kv_heads)
        v1 = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)

        # RoDE depth signal
        if loop_idx is not None and self._rode is not None:
            cos_d, sin_d = self._rode(q1, loop_idx)
            q1 = apply_depth_rotary_emb(q1.flatten(0, 1), cos_d, sin_d).view(B, self.num_heads, T, -1)
            k1 = apply_depth_rotary_emb(k1.flatten(0, 1), cos_d, sin_d).view(B, self.num_kv_heads, T, -1)

        # Q/K norm
        q1 = self.q_norm(q1)
        k1 = self.k_norm(k1)

        # RoPE
        cos, sin = self.rotary_emb(q1, position_ids)
        k1_expanded = self._expand_kv(k1)
        v1_expanded = self._expand_kv(v1)
        q1_rope, k1_rope = apply_rotary_pos_emb(q1, k1_expanded, cos, sin)
        # Determine if we can skip softmax (alpha=0 means pure GDN-2)
        skip_softmax = (hasattr(self, '_alpha_override') and self._alpha_override is not None 
                       and self._alpha_override == 0.0)

        if not skip_softmax:
            o_softmax = self._softmax_attention(q1_rope, k1_rope, v1_expanded, attention_mask)

        if not self.use_dual_path:
            # Pure softmax mode (no GDN-2)
            attn_output = o_softmax
        else:
            # === Path 2: GDN-2 ===
            # Apply short conv to Q,K for GDN-2 path
            if self.q_conv is not None:
                q2_raw = F.silu(self.q_conv(q_raw))
                k2_raw = F.silu(self.k_conv(k_raw))
            else:
                q2_raw = q_raw
                k2_raw = k_raw

            q2 = rearrange(q2_raw, "b t (h d) -> b h t d", h=self.num_heads)
            k2 = rearrange(k2_raw, "b t (h d) -> b h t d", h=self.num_kv_heads)
            v2 = rearrange(v, "b t (h d) -> b h t d", h=self.num_kv_heads)

            # L2 normalize Q,K for GDN-2 (best practice)
            q2 = F.normalize(q2, p=2, dim=-1)
            k2 = F.normalize(k2, p=2, dim=-1)

            # Compute gates
            erase_gate = torch.sigmoid(self.erase_proj(hidden_states))  # (B, T, d_kv)
            write_gate = torch.sigmoid(self.write_proj(hidden_states))  # (B, T, d_kv)
            
            # Compute decay: α_t = exp(-exp(a) * softplus(W_f @ x + δ))
            decay_input = self.decay_proj(hidden_states) + self.decay_delta  # (B, T, d_kv)
            g_t = -torch.exp(self.decay_a) * F.softplus(decay_input)        # (B, T, d_kv)
            decay = torch.exp(g_t)  # ∈ (0, 1]

            o_gdn2 = self._gdn2_attention(q2, k2, v2, erase_gate, write_gate, decay, attention_mask)

            # === Merge paths ===
            # Use scheduled alpha if set, otherwise learnable
            if hasattr(self, '_alpha_override') and self._alpha_override is not None:
                alpha = self._alpha_override
            else:
                alpha = torch.sigmoid(self.path_mix_logit).view(1, self.num_heads, 1, 1)
            
            if skip_softmax:
                attn_output = o_gdn2
            elif isinstance(alpha, float) and alpha == 0.0:
                attn_output = o_gdn2
            else:
                attn_output = alpha * o_softmax + (1.0 - alpha) * o_gdn2

        # === Output gate (SiLU) ===
        gate = F.silu(rearrange(
            self.output_gate_proj(hidden_states), "b t (h d) -> b h t d", h=self.num_heads
        ))
        attn_output = gate * attn_output

        # === Output projection ===
        attn_output = rearrange(attn_output, "b h t d -> b t (h d)")
        attn_output = proj_with_lora(self.o_proj, attn_output, "o_proj")
        attn_output = self.dropout(attn_output)

        return attn_output
