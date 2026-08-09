"""
DeepX Embedding v1.0 — LoRA Fine-tuning.

Usage:
    from deepx_embed import DeepXEmbed, LoRAFineTuner
    model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")
    tuner = LoRAFineTuner(model, lr=1e-5)
    tuner.train(triplets, epochs=3)
    tuner.save("my-model/")
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from pathlib import Path


class LoRAFineTuner:
    """Fine-tune DeepX Embedding with LoRA (only adapters + norms trainable)."""

    def __init__(
        self,
        model,  # DeepXEmbed instance
        lr: float = 1e-5,
        lora_rank: int = 16,
        temperature: float = 0.07,
        matryoshka_dims: List[int] = [256, 512, 768, 1024, 1536],
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

        # Optimizer
        self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, weight_decay=0.01)

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Encode a batch of texts, return embeddings on GPU with grad."""
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

        embed_device = next(self.model.pipeline.token_embedding.parameters()).device
        hidden = self.model.pipeline.token_embedding(input_ids.to(embed_device)).to(self.model.device)
        hidden.requires_grad_(True)
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
    ):
        """
        Fine-tune on triplets (query, positive, negative).

        Args:
            triplets: List of (query, positive_doc, negative_doc)
            epochs: Number of training epochs
            batch_size: Batch size
            log_every: Log every N steps
        """
        import random

        self.model.pipeline.backbone.train()
        total_steps = 0

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
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.trainable_params, 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    epoch_loss += loss.item()
                    n_batches += 1
                    total_steps += 1

                    if total_steps % log_every == 0:
                        avg = epoch_loss / n_batches
                        print(f"  epoch {epoch+1}/{epochs} | step {total_steps} | loss {avg:.4f}")

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM at step {total_steps}, skipping batch")
                    continue

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch+1}/{epochs} done | avg loss: {avg_loss:.4f}")

        self.model.pipeline.backbone.eval()

    def save(self, path: str):
        """Save fine-tuned model state dict."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.model.pipeline.backbone.state_dict(),
            path / "backbone_finetuned.pt"
        )
        print(f"Saved to {path / 'backbone_finetuned.pt'}")
