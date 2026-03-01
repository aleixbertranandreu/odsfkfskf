"""
dataset.py — Triplet Dataset for Fashion Metric Learning
Generates (anchor, positive, negative) triplets with hard-negative-aware sampling.
Augmentations simulate real-world bundle conditions (outdoor lighting, occlusion, angles).
"""
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────
# Augmentation pipelines
# ─────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 224):
    """
    Streamlined augmentation for maximum GPU throughput (avoiding CPU bottleneck).
    Removed ColorJitter, Blur, and Perspective as they starve the GPU on large datasets.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            img_size, 
            scale=(0.7, 1.0), 
            ratio=(0.8, 1.2),
            interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)), # Fast tensor-level occlusion
    ])


def get_eval_transform(img_size: int = 224):
    """Clean transform for evaluation / index building."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────────────────────────────
# Triplet Dataset
# ─────────────────────────────────────────────────────────────────

class TripletFashionDataset(Dataset):
    """
    Produces (anchor, positive, negative) triplets for metric learning.
    
    Strategy:
    - Anchor: a catalog product image (clean, white bg)
    - Positive: the bundle image that contains this product (noisy, real-world)
    - Negative: a different product from the SAME category (hard negative)
    
    This forces the model to learn fine-grained distinctions within categories,
    not just "shirt vs. trousers" but "THIS specific linen shirt vs. THAT cotton shirt."
    """
    
    def __init__(
        self,
        matches_csv: str,
        products_csv: str,
        bundles_csv: str,
        product_images_dir: str,
        bundle_images_dir: str,
        transform=None,
        img_size: int = 224,
    ):
        self.transform = transform or get_train_transform(img_size)
        self.eval_transform = get_eval_transform(img_size)
        self.product_images_dir = product_images_dir
        self.bundle_images_dir = bundle_images_dir
        
        # Load CSV data
        df_matches = pd.read_csv(matches_csv)
        df_products = pd.read_csv(products_csv)
        df_bundles = pd.read_csv(bundles_csv)
        
        # Build product category lookup
        self.product_category = dict(
            zip(df_products['product_asset_id'], df_products['product_description'])
        )
        
        # Build bundle section lookup (for which part of the outfit this is)
        self.bundle_section = dict(
            zip(df_bundles['bundle_asset_id'], df_bundles['bundle_id_section'])
        )
        
        # Build the match pairs: (bundle_id, product_id)
        self.pairs = []
        for _, row in df_matches.iterrows():
            bundle_id = row['bundle_asset_id']
            product_id = row['product_asset_id']
            
            # Verify images exist
            product_path = os.path.join(product_images_dir, f"{product_id}.jpg")
            bundle_path = os.path.join(bundle_images_dir, f"{bundle_id}.jpg")
            
            if os.path.exists(product_path) and os.path.exists(bundle_path):
                self.pairs.append((bundle_id, product_id))
        
        # Build category → product_ids mapping for hard negative mining
        self.category_to_products = defaultdict(list)
        for product_id, cat in self.product_category.items():
            product_path = os.path.join(product_images_dir, f"{product_id}.jpg")
            if os.path.exists(product_path):
                self.category_to_products[cat].append(product_id)
        
        # All product IDs that have images
        self.all_product_ids = [
            pid for pid in df_products['product_asset_id']
            if os.path.exists(os.path.join(product_images_dir, f"{pid}.jpg"))
        ]
        
        print(f"INFO: Dataset: {len(self.pairs)} valid pairs | "
              f"{len(self.all_product_ids)} products with images | "
              f"{len(self.category_to_products)} categories")
    
    def __len__(self):
        return len(self.pairs)
    
    def _load_image(self, path: str) -> Image.Image:
        """Load and convert to RGB, with fallback for corrupt images."""
        try:
            img = Image.open(path).convert("RGB")
            return img
        except Exception:
            # Return a blank image if corrupt
            return Image.new("RGB", (224, 224), (128, 128, 128))
    
    def _get_hard_negative(self, anchor_product_id: str) -> str:
        """
        Pick a hard negative: SAME category, DIFFERENT product.
        This is what makes the model sweat — differentiating shirts from shirts.
        Falls back to random product if category has only one item.
        """
        category = self.product_category.get(anchor_product_id, "UNKNOWN")
        same_cat = self.category_to_products.get(category, [])
        
        # Filter out the anchor itself
        candidates = [pid for pid in same_cat if pid != anchor_product_id]
        
        if candidates:
            return random.choice(candidates)
        else:
            # Fallback: random product from entire catalog
            neg_id = anchor_product_id
            while neg_id == anchor_product_id:
                neg_id = random.choice(self.all_product_ids)
            return neg_id
    
    def __getitem__(self, idx: int):
        bundle_id, product_id = self.pairs[idx]
        
        # Anchor: clean catalog image of the product
        anchor_path = os.path.join(self.product_images_dir, f"{product_id}.jpg")
        anchor_img = self._load_image(anchor_path)
        
        # Positive: the bundle image that contains this product
        positive_path = os.path.join(self.bundle_images_dir, f"{bundle_id}.jpg")
        positive_img = self._load_image(positive_path)
        
        # Negative: hard negative from same category
        neg_product_id = self._get_hard_negative(product_id)
        negative_path = os.path.join(self.product_images_dir, f"{neg_product_id}.jpg")
        negative_img = self._load_image(negative_path)
        
        # Apply augmentations
        anchor = self.transform(anchor_img)
        positive = self.transform(positive_img)
        negative = self.transform(negative_img)
        
        return anchor, positive, negative


class CatalogDataset(Dataset):
    """
    Simple dataset for encoding ALL catalog images into embeddings.
    Used by build_index.py.
    """
    
    def __init__(self, products_csv: str, product_images_dir: str, img_size: int = 224):
        self.product_images_dir = product_images_dir
        self.transform = get_eval_transform(img_size)
        
        df = pd.read_csv(products_csv)
        
        self.items = []
        for _, row in df.iterrows():
            pid = row['product_asset_id']
            path = os.path.join(product_images_dir, f"{pid}.jpg")
            if os.path.exists(path):
                self.items.append((pid, path))
        
        print(f"INFO: Catalog: {len(self.items)} products ready for indexing")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        pid, path = self.items[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        
        return self.transform(img), pid


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    ds = TripletFashionDataset(
        matches_csv=os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv"),
        products_csv=os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv"),
        bundles_csv=os.path.join(BASE_DIR, "data", "raw", "bundles_dataset.csv"),
        product_images_dir=os.path.join(BASE_DIR, "data", "images", "products"),
        bundle_images_dir=os.path.join(BASE_DIR, "data", "images", "bundles"),
    )
    
    print(f"Dataset length: {len(ds)}")
    if len(ds) > 0:
        a, p, n = ds[0]
        print(f"Anchor shape: {a.shape}, Positive: {p.shape}, Negative: {n.shape}")
