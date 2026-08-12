"""
Example: Fine-tune DeepX Embedding on custom domain data.

This example shows how to adapt the model to a new domain
using QLoRA fine-tuning with triplet data (query, positive, negative).

Default uses 4-bit quantization (QLoRA) — fits 8GB GPUs.
"""
import json
from deepx_embed import DeepXEmbed, LoRAFineTuner

# === Load model ===
model = DeepXEmbed.from_pretrained("dxtech-asia/deepx-embedding-v1")

# === Prepare data ===
# Your data should be a list of (query, positive_doc, negative_doc) triplets.
# Example: load from JSONL file
triplets = []
with open("data/my_domain_triplets.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        triplets.append((obj["query"], obj["positive"], obj["negative"]))

print(f"Loaded {len(triplets)} triplets")

# === Fine-tune ===
# Default: 4-bit QLoRA (lowest VRAM, fits 8GB GPUs)
tuner = LoRAFineTuner(
    model,
    lr=1e-5,           # Learning rate (1e-5 to 5e-5 recommended)
    temperature=0.07,   # InfoNCE temperature
)

# Other quantization options:
# tuner = LoRAFineTuner(model, lr=1e-5, quantize=8)     # 8-bit base model
# tuner = LoRAFineTuner(model, lr=1e-5, quantize=None)  # fp16 (needs 12GB+ GPU)

tuner.train(
    triplets,
    epochs=3,
    batch_size=4,       # Reduce to 1-2 if OOM on 8GB GPUs
    log_every=10,
)

# === Save ===
tuner.save("output/my-finetuned-model/")

# === Test ===
print("\nTesting fine-tuned model:")
query = triplets[0][0]
positive = triplets[0][1]
negative = triplets[0][2]

q_emb = model.encode(query)
p_emb = model.encode(positive)
n_emb = model.encode(negative)

print(f"  Query-Positive similarity: {q_emb @ p_emb:.4f}")
print(f"  Query-Negative similarity: {q_emb @ n_emb:.4f}")
