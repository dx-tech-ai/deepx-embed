"""
Example: Semantic search over a document corpus.

Two-stage pipeline:
  1. ANN search with single vectors (fast)
  2. ColBERT reranking on top candidates (precise)
"""
import numpy as np
from deepx_embed import DeepXEmbed

# === Load model ===
model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

# === Build corpus index ===
corpus = [
    "Điều 5. Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng đối với người điều khiển xe ô tô vượt đèn đỏ",
    "Điều 6. Phạt tiền từ 800.000 đồng đến 1.000.000 đồng đối với người điều khiển xe mô tô vượt đèn đỏ",
    "Điều 12. Xử phạt người đi bộ vi phạm quy tắc giao thông đường bộ",
    "Điều 22. Quy định về tốc độ tối đa cho phép xe cơ giới tham gia giao thông",
    "Điều 35. Điều kiện cấp giấy phép lái xe hạng B2",
    "Luật Thuế thu nhập cá nhân số 04/2007/QH12",
    "Nghị định 100/2019/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực giao thông đường bộ",
]

print("Encoding corpus...")
corpus_embeddings = model.encode(corpus)
print(f"  Corpus: {corpus_embeddings.shape}")

# === Search ===
query = "Phạt bao nhiêu tiền khi vượt đèn đỏ?"
print(f"\nQuery: {query}")

# Stage 1: Vector similarity (ANN)
query_embedding = model.encode(query)
similarities = corpus_embeddings @ query_embedding
top_k = np.argsort(similarities)[::-1][:5]

print("\n--- Stage 1: Vector Search (top 5) ---")
for rank, idx in enumerate(top_k):
    print(f"  {rank+1}. [{similarities[idx]:.4f}] {corpus[idx][:80]}...")

# Stage 2: ColBERT reranking on top candidates
print("\n--- Stage 2: ColBERT Reranking (top 3) ---")
query_tokens = model.encode_colbert(query)
candidates = [corpus[i] for i in top_k[:3]]
candidate_tokens = model.encode_colbert(candidates)

for rank, (idx, doc_tokens) in enumerate(zip(top_k[:3], candidate_tokens)):
    # MaxSim scoring
    sim_matrix = query_tokens @ doc_tokens.T  # (Q_tokens, D_tokens)
    maxsim = sim_matrix.max(axis=1).sum()  # sum of max per query token
    print(f"  {rank+1}. [MaxSim={maxsim:.2f}] {corpus[idx][:80]}...")
