# DeepX Embed

Official inference and fine-tuning code for [DeepX Embedding v1.0](https://huggingface.co/dxtech-asia/deepx-embedding-v1).

**nDCG@10 = 0.8162** on Zalo Legal Text Retrieval (Vietnamese) — New SOTA.

## Quick Start

```bash
pip install torch transformers huggingface_hub einops
pip install fla        # for FLA Triton kernel (optional, falls back to sequential)
```

```python
from deepx_embed import DeepXEmbed

model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

# Single text
embedding = model.encode("Mức phạt khi vượt đèn đỏ là bao nhiêu?")
# Shape: (1536,)

# Batch
embeddings = model.encode([
    "Điều kiện cấp giấy phép lái xe hạng B2",
    "Thời hạn nộp thuế thu nhập cá nhân",
])
# Shape: (2, 1536)

# Matryoshka (reduce dimension for faster search)
embedding_256d = model.encode("query text", truncate_dim=256)
```

## Features

- **O(n) linear attention** — Gated DeltaNet-2, no quadratic slowdown
- **8K token context** — Process long legal documents efficiently
- **Matryoshka** — Use 256d to 1536d, trade quality for speed
- **ColBERT reranking** — Token-level vectors for precise matching
- **LoRA fine-tuning** — Adapt to your domain with minimal compute

## Installation

```bash
git clone https://github.com/dx-tech-ai/deepx-embed.git
cd deepx-embed
pip install -e .
pip install fla        # optional: enables FLA Triton kernel for O(n) inference
```

> **Note**: Without `fla`, the model falls back to sequential attention (slower but functional).

## Fine-tuning

```python
from deepx_embed import DeepXEmbed, LoRAFineTuner

model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

# Default: 4-bit QLoRA (fits 8GB GPUs)
tuner = LoRAFineTuner(model, lr=1e-5)

# Other options:
# tuner = LoRAFineTuner(model, lr=1e-5, quantize=8)     # 8-bit base
# tuner = LoRAFineTuner(model, lr=1e-5, quantize=None)  # fp16 (needs 12GB+)

# Your data: list of (query, positive, negative) triplets
triplets = [
    ("query text", "relevant document", "irrelevant document"),
    ...
]

tuner.train(triplets, epochs=3, batch_size=4)
tuner.save("my-finetuned-model/")
```

See [`examples/finetune.py`](examples/finetune.py) for full example.

## API Reference

### `DeepXEmbed.encode(texts, truncate_dim=None, normalize=True)`

Encode texts to embeddings.

- `texts`: str or list of str
- `truncate_dim`: int, optional. Truncate to this dimension (Matryoshka)
- `normalize`: bool, default True. L2 normalize output
- Returns: numpy array (N, dim)

### `DeepXEmbed.encode_colbert(texts)`

Encode texts to token-level ColBERT vectors.

- Returns: list of arrays, each (T, 128)

## License

Apache 2.0
