"""
DeepX Embedding v1.0 — LoRA Fine-tuning with QLoRA support.

Supports:
  - QLoRA 4-bit (default): base model quantized to NF4, LoRA in fp16. Fits 8GB GPUs.
  - 8-bit: base model quantized to int8. Fits 10GB GPUs.
  - fp16 (quantize=None): full precision. Needs 12GB+ GPU.

Usage:
    from deepx_embed import DeepXEmbed, LoRAFineTuner
    model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

    # Default: 4-bit QLoRA (recommended, lowest VRAM)
    tuner = LoRAFineTuner(model, lr=1e-5)

    # Other options:
    # tuner = LoRAFineTuner(model, lr=1e-5, quantize=8)     # 8-bit base
    # tuner = LoRAFineTuner(model, lr=1e-5, quantize=None)  # fp16 base (needs more VRAM)

    tuner.train(triplets, epochs=3)
    tuner.save("my-model/")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path


def _quantize_model(model: nn.Module, bits: int = 4):
    """
    Quantize all frozen Linear layers to 4-bit or 8-bit using bitsandbytes.
    Only quantizes layers where requires_grad=False for all params.
    LoRA params (requires_grad=True) stay in fp16/bf16.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        raise ImportError(
            "bitsandbytes required for quantization. Install: pip install bitsandbytes"
        )

    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Only quantize fully frozen layers
            if all(not p.requires_grad for p in module.parameters()):
                if bits == 4:
                    new_layer = bnb.nn.Linear4bit(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        compute_dtype=torch.bfloat16,
                        quant_type="nf4",
                    )
                elif bits == 8:
                    new_layer = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                    )
                else:
                    continue
                
                # Copy weights for quantization
                new_layer.weight = bnb.nn.Params4bit(
                    module.weight.data, requires_grad=False, quant_type="nf4"
                ) if bits == 4 else module.weight
                
                if module.bias is not None:
                    new_layer.bias = module.bias
                
                replacements[name] = new_layer

    # Apply replacements
    for name, new_module in replacements.items():
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    n_quantized = len(replacements)
    if n_quantized > 0:
        print(f"  Quantized {n_quantized} frozen Linear layers to {bits}-bit")
    return model


class LoRAFineTuner:
    """
    Fine-tune DeepX Embedding with LoRA.
    
    Supports QLoRA (4-bit/8-bit quantized base model) for low-VRAM GPUs.
    
    VRAM estimates:
      - fp16 (no quantize):  ~4-5 GB (with gradient checkpointing)
      - 8-bit quantize:      ~3-4 GB
      - 4-bit quantize:      ~2-3 GB
    """

    def __init__(
        self,
        model,  # DeepXEmbed instance
        lr: float = 1e-5,
        lora_rank: int = 16,
        temperature: float = 0.07,
        matryoshka_dims: List[int] = [256, 512, 768, 1024, 1536],
        quantize: Optional[int] = 4,  # 4-bit QLoRA by default. Set None for fp16, 8 for 8-bit.
    ):
        self.model = model
        self.lr = lr
        self.temperature = temperature
        self.matryoshka_dims = matryoshka_dims
        
        # Freeze everything, unfreeze only LoRA + norms + heads
        for param in model.pipeline.parameters():
            param.requires_grad = False
        
        trainable_count = 0
        for name, param in model.pipeline.backbone.named_parameters():
            if any(k in name for k in ["lora", "norm", "pool_query", "colbert_head", "path_mix_logit"]):
                param.requires_grad = True
                trainable_count += param.numel()
        
        self.trainable_params = [p for p in model.pipeline.backbone.parameters() if p.requires_grad]
        print(f"Trainable parameters: {trainable_count / 1e6:.1f}M")

        # Quantize frozen layers (QLoRA)
        if quantize is not None:
            assert quantize in (4, 8), "quantize must be 4 or 8"
            print(f"Applying {quantize}-bit quantization to frozen layers...")
            _quantize_model(model.pipeline.backbone, bits=quantize)
            self.quantized = quantize
        else:
            self.quantized = None

        # Optimizer — use 8-bit Adam if bitsandbytes available
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.trainable_params, lr=lr, weight_decay=0.01)
            print(f"  Using 8-bit AdamW optimizer")
        except ImportError:
            self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=0.01)

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts, return embeddings with grad for LoRA training."""
        encoded = self.model.tokenizer(
            texts, padding=True, truncation=True,
            max_length=2048, return_tensors="pt"
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        if self.model.id_remap is not None:
            input_ids = self.model.id_remap[input_ids]

        max_actual = int(attention_mask.sum(dim=1).max().item())
        input_ids = input_ids[:, :max_actual]
        attention_mask = attention_mask[:, :max_actual]

        # Token embedding (frozen, on CPU to save VRAM)
        with torch.no_grad():
            embed_device = next(self.model.pipeline.token_embedding.parameters()).device
            hidden = self.model.pipeline.token_embedding(input_ids.to(embed_device))
        
        # Move to GPU and enable grad (needed for LoRA gradient flow)
        hidden = hidden.to(self.model.device).requires_grad_(True)
        mask = attention_mask.to(self.model.device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = self.model.pipeline.backbone(hidden, attention_mask=mask, normalize=False)
        
        return emb

    def _matryoshka_loss(self, q: torch.Tensor, p: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        """Compute Matryoshka InfoNCE loss across multiple dimensions."""
        total = 0.0
        for dim in self.matryoshka_dims:
            qd = F.normalize(q[:, :dim], p=2, dim=-1)
            pd = F.normalize(p[:, :dim], p=2, dim=-1)
            nd = F.normalize(n[:, :dim], p=2, dim=-1)
            
            candidates = torch.cat([pd, nd], dim=0)
            sim = torch.mm(qd, candidates.t()) / self.temperature
            labels = torch.arange(qd.size(0), device=sim.device)
            total += F.cross_entropy(sim, labels)
        
        return total / len(self.matryoshka_dims)

    def train(
        self,
        triplets: List[Tuple[str, str, str]],
        epochs: int = 3,
        batch_size: int = 4,
        log_every: int = 10,
        gradient_accumulation: int = 1,
    ):
        """
        Fine-tune on triplets (query, positive, negative).

        Args:
            triplets: List of (query, positive_doc, negative_doc)
            epochs: Number of training epochs
            batch_size: Batch size (reduce if OOM)
            log_every: Log every N steps
            gradient_accumulation: Accumulate gradients over N batches
        """
        import random

        self.model.pipeline.backbone.train()
        total_steps = 0
        accum_count = 0

        for epoch in range(epochs):
            random.shuffle(triplets)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(triplets), batch_size):
                batch = triplets[i:i + batch_size]
                queries = [t[0] for t in batch]
                positives = [t[1] for t in batch]
                negatives = [t[2] for t in batch]

                try:
                    q_emb = self._encode_batch(queries)
                    p_emb = self._encode_batch(positives)
                    n_emb = self._encode_batch(negatives)

                    loss = self._matryoshka_loss(q_emb, p_emb, n_emb)
                    loss = loss / gradient_accumulation
                    loss.backward()

                    accum_count += 1
                    if accum_count >= gradient_accumulation:
                        torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accum_count = 0

                    epoch_loss += loss.item() * gradient_accumulation
                    n_batches += 1
                    total_steps += 1

                    if total_steps % log_every == 0:
                        avg = epoch_loss / n_batches
                        vram = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                        print(f"  epoch {epoch+1}/{epochs} | step {total_steps} | loss {avg:.4f} | VRAM {vram:.1f}GB")

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    accum_count = 0
                    print(f"  OOM at step {total_steps}, skipping batch. Try reducing batch_size.")
                    continue

            # Flush remaining gradients
            if accum_count > 0:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                accum_count = 0

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs} done | avg loss: {avg_loss:.4f}")

        self.model.pipeline.backbone.eval()

    def save(self, path: str):
        """Save fine-tuned LoRA weights (small, ~12MB)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save only trainable params (LoRA + norms + heads)
        lora_state = {
            k: v for k, v in self.model.pipeline.backbone.state_dict().items()
            if any(x in k for x in ["lora", "norm", "pool_query", "colbert_head", "path_mix_logit"])
        }
        
        save_path = path / "lora_weights.pt"
        torch.save({
            "lora_state_dict": lora_state,
            "quantized": self.quantized,
        }, save_path)
        size_mb = save_path.stat().st_size / 1024**2
        print(f"Saved LoRA weights: {save_path} ({size_mb:.1f} MB)")
        print(f"To load: model.pipeline.backbone.load_state_dict(torch.load('{save_path}')['lora_state_dict'], strict=False)")
