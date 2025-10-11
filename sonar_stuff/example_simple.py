"""
Quick example showing how to use the Simple SONAR encoder
and the autoregressive model.
"""
import torch
from sonar_simple import SimpleSonarEncoder

# ----------------------------
# 1. Basic Encoding
# ----------------------------
print("="*60)
print("1. Basic SONAR Encoding Example")
print("="*60)

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = SimpleSonarEncoder(device=device)

# Encode some sentences
sentences = [
    'My name is SONAR.',
    'I can embed sentences into vectorial space.',
    'This is much simpler than fairseq!'
]

embeddings = encoder.encode(sentences, lang="eng_Latn")
print(f"\nInput: {len(sentences)} sentences")
print(f"Output shape: {embeddings.shape}")  # [3, 1024]
print(f"First embedding (first 5 dims): {embeddings[0, :5]}")

# ----------------------------
# 2. Normalized Embeddings
# ----------------------------
print("\n" + "="*60)
print("2. Normalized Embeddings")
print("="*60)

embeddings_norm = encoder.encode(sentences, lang="eng_Latn", norm=True)
print(f"\nNormalized embedding norms:")
for i, emb in enumerate(embeddings_norm):
    norm = torch.norm(emb).item()
    print(f"  Sentence {i+1}: {norm:.4f}")  # Should be ~1.0

# ----------------------------
# 3. Semantic Similarity
# ----------------------------
print("\n" + "="*60)
print("3. Semantic Similarity")
print("="*60)

test_sentences = [
    "The cat sits on the mat.",
    "A feline rests on the rug.",
    "The dog barks loudly.",
]

test_embs = encoder.encode(test_sentences, lang="eng_Latn", norm=True)

# Compute cosine similarities
print("\nCosine similarities:")
for i in range(len(test_sentences)):
    for j in range(i+1, len(test_sentences)):
        sim = torch.cosine_similarity(test_embs[i], test_embs[j], dim=0)
        print(f"  '{test_sentences[i][:30]}...' <-> '{test_sentences[j][:30]}...': {sim.item():.3f}")

# ----------------------------
# 4. Batch Processing
# ----------------------------
print("\n" + "="*60)
print("4. Batch Processing")
print("="*60)

# Large batch of sentences
large_batch = [f"This is test sentence number {i}." for i in range(100)]
print(f"\nEncoding {len(large_batch)} sentences...")

with torch.inference_mode():
    batch_embs = encoder.encode(large_batch, lang="eng_Latn")

print(f"Output shape: {batch_embs.shape}")  # [100, 1024]
print(f"Memory used: {batch_embs.element_size() * batch_embs.nelement() / 1024 / 1024:.2f} MB")

# ----------------------------
# 5. Sequence for Autoregressive Model
# ----------------------------
print("\n" + "="*60)
print("5. Preparing Sequences for Autoregressive Training")
print("="*60)

# Simulate a multi-sentence context
context = [
    "Let's solve this problem step by step.",
    "First, we need to identify the key variables.",
    "Next, we establish the constraints.",
    "Then we can formulate the solution.",
]

context_embs = encoder.encode(context, lang="eng_Latn")
print(f"\nContext: {len(context)} sentences")
print(f"Context embeddings shape: {context_embs.shape}")  # [4, 1024]

# For autoregressive training: input is [:-1], target is [1:]
input_seq = context_embs[:-1]   # First 3 sentences
target_seq = context_embs[1:]   # Last 3 sentences

print(f"\nInput sequence shape: {input_seq.shape}")    # [3, 1024]
print(f"Target sequence shape: {target_seq.shape}")   # [3, 1024]

print("\nThe model learns to predict target from input:")
print("  Input: sentences 0, 1, 2")
print("  Target: sentences 1, 2, 3")
print("  (each position predicts the next)")

print("\n" + "="*60)
print("Example complete!")
print("="*60)
print("\nNext steps:")
print("  1. Run: python prepare_sonar_data.py")
print("  2. Run: python train_sonar.py")
print("  3. Run: python test_sonar_model.py")
