import numpy as np
import torch

# Supongamos que ya cargaste embeddings de productos
embeddings_products = np.load("products_clip_embeddings.npy")  # (N, 512)

# Embedding del bundle (1 imagen)
bundle_embedding = ...  # torch tensor o np.array (512,)
bundle_embedding = np.array(bundle_embedding)  # por si es torch
bundle_embedding = bundle_embedding.reshape(1, -1)  # (1,512)

# Normalizar
bundle_embedding = bundle_embedding / np.linalg.norm(bundle_embedding, axis=1, keepdims=True)

# Cosine similarity (producto punto porque ya están normalizados)
similarities = bundle_embedding @ embeddings_products.T  # (1,N)

best_index = np.argmax(similarities)
print("Mejor producto:", product_ids[best_index])
print("Similitud:", similarities[0][best_index])