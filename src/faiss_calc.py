import faiss
import numpy as np

# Cargar embeddings
embeddings = np.load("products_clip_embeddings.npy").astype(np.float32)

# Normalizar
faiss.normalize_L2(embeddings)

# Dimensión
d = embeddings.shape[1]

# Crear índice
index = faiss.IndexFlatIP(d)
index.add(embeddings)

# Guardar índice
faiss.write_index(index, "products_clip_faiss.index")
print("FAISS index guardado con", index.ntotal, "vectores")