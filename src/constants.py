from pathlib import Path
import torch

PRODUCTS_IMAGES_DIR = Path("../data/images/products")
PRODUCTS_DATASET_CSV = Path("../data/raw/product_dataset.csv")

BUNDLES_IMAGES_DIR = Path("../data/images/bundles")
BUNDLE_DATASET_CSV = Path("../data/raw/bundles_dataset.csv")

OUTPUT_EMBEDDINGS = Path("../data/embeddings/products_clip_embeddings.npy")
OUTPUT_IDS = Path("../data/embeddings/products_ids.pkl")
OUTPUT_DESCRIPTIONS = Path("../data/embeddings/products_descriptions.pkl")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_EMB_PATH = "../data/processed/product_embeddings.npy"
OUTPUT_META_PATH = "../data/processed/product_metadata.csv"

BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"