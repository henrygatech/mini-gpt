
# %%

import torch
import torch.nn as nn

# Model parameters from your context
n_embd = 32       # embedding dimension
block_size = 8    # sequence length (number of tokens)
vocab_size = 50   # arbitrary vocab size for this toy example

# Create a toy token embedding table
embedding_table = nn.Embedding(vocab_size, n_embd)

# Create a toy sequence (a single sequence of tokens, shape: [1, block_size])
# For example, let's assume these token indices represent a sentence.
toy_sequence = torch.tensor([[1, 5, 12, 25, 33, 7, 8, 15]])

# Get token embeddings: shape (1, block_size, n_embd)
token_embeddings = embedding_table(toy_sequence)
print("Original token embeddings (for each token in the sequence):")
print(token_embeddings[0,1])

# Create a LayerNorm instance that normalizes over the last dimension (the feature dimension)
layer_norm = nn.LayerNorm(n_embd)

# Apply LayerNorm: Each token's 32-dimensional vector is normalized independently.
normalized_embeddings = layer_norm(token_embeddings)
print("\nNormalized token embeddings:")
print(normalized_embeddings[0,1])


#%%

# Verify the normalization per token: mean should be ~0 and std ~1 for each token's feature vector.
print("\nStatistics for each token after LayerNorm:")
for i in range(block_size):
    token_norm = normalized_embeddings[0, i]
    print(f"Token {i} - Mean: {token_norm.mean().item():.3f}, Std: {token_norm.std().item():.3f}")
# %%
