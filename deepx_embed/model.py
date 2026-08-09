"""
DeepX Embedding v1.0 — Inference wrapper.

Usage:
    from deepx_embed import DeepXEmbed
    model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")
    emb = model.encode("Hello world")
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Optional
from pathlib import Path


class DeepXEmbed:
    """High-level inference wrapper for DeepX Embedding model."""

    def __init__(self, pipeline, tokenizer, id_remap=None, device="cuda"):
        self.pipeline = pipeline
        self.tokenizer = tokenizer
        self.id_remap = id_remap
        self.device = device

    @classmethod
    def from_pretrained(cls, model_id: str, device: str = "cuda"):
        """
        Load model from HuggingFace Hub or local path.
        
        Args:
            model_id: HuggingFace model ID (e.g. "dxtech-asia/deepx-embedding-v1")
                      or local directory path
            device: "cuda" or "cpu"
        """
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        import sys
        import os

        # Download files
        if os.path.isdir(model_id):
            model_dir = Path(model_id)
        else:
            # Download from HF
            model_dir = Path(hf_hub_download(model_id, "checkpoints/deepx_v1.pt")).parent.parent

            # Download all needed files
            hf_hub_download(model_id, "checkpoints/deepx_v1.pt")
            hf_hub_download(model_id, "deploy/pruned/token_embedding_pruned.pt")
            hf_hub_download(model_id, "deploy/pruned/tokenizer/id_remap.pt")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load id_remap
        remap_path = model_dir / "deploy" / "pruned" / "tokenizer" / "id_remap.pt"
        id_remap = torch.load(remap_path, map_location="cpu") if remap_path.exists() else None

        # Load model
        sys.path.insert(0, str(model_dir))
        from config import DeepXConfig
        from modeling.pipeline import DeepXPipeline

        config = DeepXConfig()
        embed_path = model_dir / "deploy" / "pruned" / "token_embedding_pruned.pt"
        
        pipeline = DeepXPipeline(config, embed_path=str(model_dir / "pretrained" / "gemma4_e2b_embed.pt"))
        
        # Load pruned embedding
        if embed_path.exists():
            import torch.nn as nn
            pruned_data = torch.load(embed_path, map_location="cpu")
            pruned_weight = pruned_data["weight"]
            pipeline.token_embedding = nn.Embedding(pruned_weight.shape[0], pruned_weight.shape[1])
            pipeline.token_embedding.weight = nn.Parameter(pruned_weight, requires_grad=False)

        # Load backbone
        ckpt_path = model_dir / "checkpoints" / "deepx_v1.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        pipeline.backbone.load_state_dict(sd, strict=False)

        # Setup
        pipeline = pipeline.half().eval()
        for m in pipeline.modules():
            if hasattr(m, 'path_mix_logit'):
                m._alpha_override = 0.0

        pipeline.token_embedding = pipeline.token_embedding.to(device if device != "cuda" else "cpu")
        pipeline.backbone = pipeline.backbone.to(device)

        return cls(pipeline, tokenizer, id_remap, device)

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        truncate_dim: Optional[int] = None,
        normalize: bool = True,
        max_length: int = 8192,
        batch_size: int = 8,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single string or list of strings
            truncate_dim: Matryoshka dimension (256, 512, 768, 1024, or 1536)
            normalize: L2 normalize output
            max_length: Max token length
            batch_size: Batch size for encoding

        Returns:
            numpy array of shape (N, dim) or (dim,) for single input
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            # Remap IDs for pruned vocab
            if self.id_remap is not None:
                input_ids = self.id_remap[input_ids]

            # Trim to actual max
            max_actual = int(attention_mask.sum(dim=1).max().item())
            input_ids = input_ids[:, :max_actual]
            attention_mask = attention_mask[:, :max_actual]

            # Forward
            embed_device = next(self.pipeline.token_embedding.parameters()).device
            hidden = self.pipeline.token_embedding(input_ids.to(embed_device)).to(self.device).half()
            mask = attention_mask.to(self.device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                emb = self.pipeline.backbone(hidden, attention_mask=mask, normalize=False)

            # Truncate dimension (Matryoshka)
            if truncate_dim is not None:
                emb = emb[:, :truncate_dim]

            # Normalize
            if normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            all_embs.append(emb.cpu().float().numpy())

        result = np.concatenate(all_embs, axis=0)
        return result[0] if single else result

    @torch.no_grad()
    def encode_colbert(
        self,
        texts: Union[str, List[str]],
        max_length: int = 8192,
    ) -> List[np.ndarray]:
        """
        Encode texts to ColBERT token vectors.

        Returns:
            List of arrays, each shape (num_tokens, 128)
        """
        single = isinstance(texts, str)
        if single:
            texts = [texts]

        results = []
        for text in texts:
            encoded = self.tokenizer(
                text, truncation=True, max_length=max_length, return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]

            if self.id_remap is not None:
                input_ids = self.id_remap[input_ids]

            embed_device = next(self.pipeline.token_embedding.parameters()).device
            hidden = self.pipeline.token_embedding(input_ids.to(embed_device)).to(self.device).half()
            mask = attention_mask.to(self.device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, token_embs = self.pipeline.backbone(
                    hidden, attention_mask=mask, normalize=True, return_colbert=True
                )

            # Remove padding tokens
            seq_len = int(mask.sum().item())
            token_embs = token_embs[0, :seq_len].cpu().float().numpy()
            results.append(token_embs)

        return results[0] if single else results
