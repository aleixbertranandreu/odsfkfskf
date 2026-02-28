"""
reranker.py — Multi-Signal Re-Ranking for Fashion Visual Search
Takes Top-50 candidates from FAISS and re-ranks to Top-15 using:
  1. Embedding similarity (from metric learning)
  2. Color histogram consistency (HSV regional comparison)
  3. Category consistency (YOLO class → product category filter)
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


# ─────────────────────────────────────────────────────────────────
# Color Histogram Re-Ranking
# ─────────────────────────────────────────────────────────────────

def extract_color_histogram(
    image_rgb: np.ndarray, 
    n_regions: int = 3,
    bins: Tuple[int, int, int] = (8, 8, 8)
) -> np.ndarray:
    """
    Extract regional HSV color histograms.
    
    Split image into horizontal regions (top/mid/bottom for 3 regions).
    This captures color distribution per zone — e.g., a white collar
    on a blue shirt will have different top vs. mid histograms.
    
    Returns a concatenated, normalized histogram vector.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    
    histograms = []
    region_h = h // n_regions
    
    for i in range(n_regions):
        start_y = i * region_h
        end_y = (i + 1) * region_h if i < n_regions - 1 else h
        region = hsv[start_y:end_y, :, :]
        
        hist = cv2.calcHist(
            [region], 
            [0, 1, 2],  # H, S, V channels
            None,
            list(bins),
            [0, 180, 0, 256, 0, 256]
        )
        hist = cv2.normalize(hist, hist).flatten()
        histograms.append(hist)
    
    return np.concatenate(histograms)


def compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Compare two histograms using intersection method.
    Returns similarity score in [0, 1].
    """
    # Histogram intersection: sum of element-wise minimums
    if len(hist1) != len(hist2):
        return 0.0
    
    intersection = np.minimum(hist1, hist2).sum()
    total = max(hist1.sum(), 1e-10)
    
    return float(intersection / total)


# ─────────────────────────────────────────────────────────────────
# Category Consistency Scoring
# ─────────────────────────────────────────────────────────────────

# YOLO detection class → allowed Inditex product categories
YOLO_TO_CATEGORIES = {
    "top": [
        'T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP',
        'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT', 'SWEATSHIRT',
        'OVERSHIRT', 'KNITTED WAISTCOAT', 'TOPS AND OTHERS', 'WIND-JACKET',
        'ANORAK', 'BLAZER', 'SLEEVELESS PAD. JACKET'
    ],
    "trousers": [
        'TROUSERS', 'JEANS', 'SHORTS', 'BERMUDA', 'LEGGINGS'
    ],
    "skirt": [
        'SKIRT'
    ],
    "dress": [
        'DRESS', 'JUMPSUIT', 'OVERALL', 'BIB OVERALL'
    ],
    "outwear": [
        'JACKET', 'COAT', 'BLAZER', 'TRENCH RAINCOAT', 'ANORAK',
        'WIND-JACKET', 'SLEEVELESS PAD. JACKET'
    ],
    # Fallback for unknown YOLO classes — allow anything clothing-related
    "unknown": None  # No filtering
}


def category_score(
    yolo_class: Optional[str],
    product_category: str,
) -> float:
    """
    Score how well a candidate product matches the detected garment type.
    Returns 1.0 for match, 0.1 for mismatch (soft penalty, not zero).
    """
    if yolo_class is None or yolo_class == "unknown":
        return 0.5  # Neutral if we don't know what the crop is
    
    allowed = YOLO_TO_CATEGORIES.get(yolo_class.lower())
    if allowed is None:
        return 0.5
    
    # Check if product category matches any allowed category
    product_upper = product_category.upper().strip()
    for allowed_cat in allowed:
        if allowed_cat in product_upper or product_upper in allowed_cat:
            return 1.0
    
    return 0.1  # Soft penalty for mismatch


# ─────────────────────────────────────────────────────────────────
# Main Re-Ranker
# ─────────────────────────────────────────────────────────────────

class Reranker:
    """
    Takes Top-K candidates and re-ranks using a weighted combination:
      final_score = w_embed * embed_sim + w_color * color_sim + w_cat * cat_score
    """
    
    def __init__(
        self,
        w_embed: float = 0.4,
        w_color: float = 0.3,
        w_cat: float = 0.3,
    ):
        self.w_embed = w_embed
        self.w_color = w_color
        self.w_cat = w_cat
    
    def rerank(
        self,
        crop_image: np.ndarray,
        yolo_class: Optional[str],
        candidate_ids: List[str],
        candidate_scores: np.ndarray,
        candidate_images: Dict[str, np.ndarray],
        product_categories: Dict[str, str],
        top_k: int = 15,
    ) -> List[Tuple[str, float]]:
        """
        Re-rank candidates for a single crop.
        
        Args:
            crop_image: RGB numpy array of the detected crop
            yolo_class: YOLO class name (e.g., "top", "trousers")
            candidate_ids: list of product IDs (top-K from FAISS)
            candidate_scores: embedding similarity scores from FAISS
            candidate_images: dict mapping product_id → RGB numpy array
            product_categories: dict mapping product_id → category string
            top_k: how many to return after re-ranking
        
        Returns:
            List of (product_id, final_score) sorted descending, length top_k
        """
        # Extract color histogram for the query crop
        crop_hist = extract_color_histogram(crop_image)
        
        # Normalize embedding scores to [0, 1]
        embed_min = candidate_scores.min()
        embed_max = candidate_scores.max()
        embed_range = embed_max - embed_min
        if embed_range < 1e-10:
            embed_normalized = np.ones_like(candidate_scores)
        else:
            embed_normalized = (candidate_scores - embed_min) / embed_range
        
        results = []
        for i, pid in enumerate(candidate_ids):
            # Embedding component
            e_score = embed_normalized[i]
            
            # Color histogram component
            if pid in candidate_images and candidate_images[pid] is not None:
                cand_hist = extract_color_histogram(candidate_images[pid])
                c_score = compare_histograms(crop_hist, cand_hist)
            else:
                c_score = 0.5  # Neutral if no image available
            
            # Category consistency component
            cat_str = product_categories.get(pid, "UNKNOWN")
            cat_s = category_score(yolo_class, cat_str)
            
            # Weighted combination
            final = (
                self.w_embed * e_score +
                self.w_color * c_score +
                self.w_cat * cat_s
            )
            
            results.append((pid, final))
        
        # Sort by final score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def rerank_simple(
        self,
        candidate_ids: List[str],
        candidate_scores: np.ndarray,
        yolo_class: Optional[str],
        product_categories: Dict[str, str],
        top_k: int = 15,
    ) -> List[Tuple[str, float]]:
        """
        Simplified re-ranking without color histograms (faster, no candidate images needed).
        Uses only embedding scores + category consistency.
        Useful when catalog images aren't loaded in memory.
        """
        embed_min = candidate_scores.min()
        embed_max = candidate_scores.max()
        embed_range = embed_max - embed_min
        if embed_range < 1e-10:
            embed_normalized = np.ones_like(candidate_scores)
        else:
            embed_normalized = (candidate_scores - embed_min) / embed_range
        
        w_embed_adj = self.w_embed + self.w_color  # redistribute color weight to embed
        w_cat_adj = self.w_cat
        
        results = []
        for i, pid in enumerate(candidate_ids):
            e_score = embed_normalized[i]
            cat_str = product_categories.get(pid, "UNKNOWN")
            cat_s = category_score(yolo_class, cat_str)
            
            final = w_embed_adj * e_score + w_cat_adj * cat_s
            results.append((pid, final))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
