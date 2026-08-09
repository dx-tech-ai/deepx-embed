"""
Example: Basic inference with DeepX Embedding v1.0
"""
from deepx_embed import DeepXEmbed

# Load model
model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

# === Single text encoding ===
query = "Mức phạt khi vượt đèn đỏ là bao nhiêu?"
embedding = model.encode(query)
print(f"Query embedding: shape={embedding.shape}, norm={embedding.dot(embedding):.4f}")

# === Batch encoding ===
documents = [
    "Điều 5. Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông",
    "Điều 12. Xử phạt người điều khiển xe mô tô vi phạm",
    "Luật Thuế thu nhập doanh nghiệp số 14/2008/QH12",
]
doc_embeddings = model.encode(documents)
print(f"Doc embeddings: shape={doc_embeddings.shape}")

# === Similarity search ===
import numpy as np
similarities = doc_embeddings @ embedding
print(f"\nSimilarities to query '{query}':")
for i, (doc, sim) in enumerate(zip(documents, similarities)):
    print(f"  [{sim:.4f}] {doc[:60]}...")

# === Matryoshka (reduced dimension) ===
emb_256 = model.encode(query, truncate_dim=256)
emb_512 = model.encode(query, truncate_dim=512)
print(f"\nMatryoshka dimensions:")
print(f"  256d: shape={emb_256.shape}")
print(f"  512d: shape={emb_512.shape}")
print(f"  1536d: shape={embedding.shape}")

# === ColBERT token vectors ===
token_vectors = model.encode_colbert(query)
print(f"\nColBERT: {token_vectors.shape[0]} tokens × {token_vectors.shape[1]}d")
