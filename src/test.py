import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Cargar embeddings y nombres
embeddings = np.load("products_clip_embeddings.npy")
with open("products_ids.pkl", "rb") as f:
    ids = pickle.load(f)

# Reducir dimensionalidad
tsne = TSNE(n_components=2, perplexity=50, random_state=42)
emb_2d = tsne.fit_transform(embeddings)

# Graficar
plt.figure(figsize=(12,8))
plt.scatter(emb_2d[:,0], emb_2d[:,1], s=5)

# Opcional: mostrar algunos nombres de productos
for i in range(0, len(ids), 500):  # cada 500 puntos
    plt.text(emb_2d[i,0], emb_2d[i,1], ids[i], fontsize=8)

plt.title("Visualización de embeddings CLIP de productos (t-SNE)")
plt.show()