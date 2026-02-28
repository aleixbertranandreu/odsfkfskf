from pathlib import Path
import torch

PRODUCTS_IMAGES_DIR = Path("../data/images/products")
PRODUCTS_DATASET_CSV = Path("../data/raw/product_dataset.csv")

OUTPUT_EMBEDDINGS = Path("../data/embeddings/products_clip_embeddings.npy")
OUTPUT_IDS = Path("../data/embeddings/products_ids.pkl")
OUTPUT_DESCRIPTIONS = Path("../data/embeddings/products_descriptions.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"