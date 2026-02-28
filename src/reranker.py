"""
reranker.py — Kaggle Grandmaster Re-Ranking for Fashion Visual Search
═══════════════════════════════════════════════════════════════════════

Three-stage re-ranking pipeline:
  Stage 1: HARD CATEGORY MASK — nuke anything that doesn't match YOLO class
  Stage 2: Block HSV histogram — 3-stripe regional color comparison
  Stage 3: Weighted fusion — embed_sim × color_sim × category_bonus

Designed to squeeze every decimal of MRR@15.
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


# ═════════════════════════════════════════════════════════════════
# CATEGORY MASK — The Nuclear Option
# ═════════════════════════════════════════════════════════════════

# Exhaustive YOLO → Inditex category mapping.
# If YOLO says "top", ONLY these product types survive.
# Everything else gets a death penalty.
YOLO_TO_CATEGORIES = {
    "top": {
        'T-SHIRT', 'SHIRT', 'SWEATER', 'JACKET', 'COAT', 'BLOUSE', 'TOP',
        'WAISTCOAT', 'CARDIGAN', 'POLO SHIRT', 'BODYSUIT', 'SWEATSHIRT',
        'OVERSHIRT', 'KNITTED WAISTCOAT', 'TOPS AND OTHERS', 'WIND-JACKET',
        'ANORAK', 'BLAZER', 'SLEEVELESS PAD. JACKET', 'VEST', 'HOODIE',
        'CROP TOP', 'TANK TOP', 'LONG SLEEVE T-SHIRT', 'RUGBY POLO SHIRT',
    },
    "trousers": {
        'TROUSERS', 'JEANS', 'SHORTS', 'BERMUDA', 'LEGGINGS', 'JOGGER',
        'CARGO TROUSERS', 'CHINO', 'SWEATPANTS', 'CARGO SHORTS',
    },
    "skirt": {
        'SKIRT', 'MINI SKIRT', 'MIDI SKIRT', 'LONG SKIRT',
    },
    "dress": {
        'DRESS', 'JUMPSUIT', 'OVERALL', 'BIB OVERALL', 'ROMPER',
    },
    "outwear": {
        'JACKET', 'COAT', 'BLAZER', 'TRENCH RAINCOAT', 'ANORAK',
        'WIND-JACKET', 'SLEEVELESS PAD. JACKET', 'PUFFER JACKET',
        'LEATHER JACKET', 'DENIM JACKET', 'BOMBER JACKET', 'PARKA',
        'RAINCOAT', 'VEST',
    },
}

# Reverse index: product_category → set of YOLO classes that allow it
# Built once at import time for O(1) lookups during inference
_CATEGORY_TO_YOLO = {}
for _yolo_cls, _cats in YOLO_TO_CATEGORIES.items():
    for _cat in _cats:
        if _cat not in _CATEGORY_TO_YOLO:
            _CATEGORY_TO_YOLO[_cat] = set()
        _CATEGORY_TO_YOLO[_cat].add(_yolo_cls)


def category_mask_score(
    yolo_class: Optional[str],
    product_category: str,
) -> float:
    """
    HARD category mask. This is not a soft penalty — it's a binary gate.
    
    Returns:
        1.0  → category matches YOLO detection (PASS)
        0.05 → category does NOT match (DEATH PENALTY, pushed to bottom)
        0.5  → unknown YOLO class or unknown product (NEUTRAL)
    
    The 0.05 instead of 0.0 ensures that if ALL candidates are wrong-category
    (edge case), we still return something ranked by embedding quality.
    """
    if yolo_class is None or yolo_class == "unknown":
        return 0.5  # Can't filter without detection info

    allowed_set = YOLO_TO_CATEGORIES.get(yolo_class.lower())
    if allowed_set is None:
        return 0.5

    product_upper = product_category.upper().strip()

    # Direct match
    if product_upper in allowed_set:
        return 1.0

    # Substring match (handles "CARGO TROUSERS" matching "TROUSERS")
    for allowed_cat in allowed_set:
        if allowed_cat in product_upper or product_upper in allowed_cat:
            return 1.0

    return 0.05  # Death penalty


# ═════════════════════════════════════════════════════════════════
# BLOCK HSV HISTOGRAM — 3-Stripe Color Fingerprint
# ═════════════════════════════════════════════════════════════════

def extract_block_histogram(
    image_rgb: np.ndarray,
    n_blocks: int = 3,
    h_bins: int = 12,
    s_bins: int = 8,
    v_bins: int = 4,
) -> np.ndarray:
    """
    Extract per-block HSV histograms from horizontal stripes.
    
    Splits the image into `n_blocks` horizontal stripes and extracts
    a joint HSV histogram from each. This captures spatial color distribution:
    - Block 0 (top): collar/neckline area
    - Block 1 (mid): main body/torso
    - Block 2 (bottom): hem/waist area

    Using fewer V bins and more H bins prioritizes HUE (actual color)
    over brightness (lighting variance between street and catalog).
    
    Returns:
        Concatenated normalized histogram vector, shape (n_blocks * h_bins * s_bins * v_bins,)
    """
    # Handle edge cases
    if image_rgb is None or image_rgb.size == 0:
        return np.zeros(n_blocks * h_bins * s_bins * v_bins, dtype=np.float32)

    # Ensure minimum size
    if image_rgb.shape[0] < n_blocks or image_rgb.shape[1] < 2:
        return np.zeros(n_blocks * h_bins * s_bins * v_bins, dtype=np.float32)

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    img_h, img_w = hsv.shape[:2]

    histograms = []
    block_h = img_h // n_blocks

    for i in range(n_blocks):
        y_start = i * block_h
        y_end = (i + 1) * block_h if i < n_blocks - 1 else img_h
        block = hsv[y_start:y_end, :, :]

        # Compute 3D histogram for this block
        hist = cv2.calcHist(
            [block],
            [0, 1, 2],
            None,
            [h_bins, s_bins, v_bins],
            [0, 180, 0, 256, 0, 256]
        )
        # L1 normalize per block (makes comparison lighting-invariant)
        block_sum = hist.sum()
        if block_sum > 0:
            hist = hist / block_sum
        histograms.append(hist.flatten())

    return np.concatenate(histograms).astype(np.float32)


def compare_block_histograms(hist_query: np.ndarray, hist_candidate: np.ndarray) -> float:
    """
    Compare two block histograms using Bhattacharyya-inspired similarity.
    
    Uses sqrt(h1 * h2) which is more discriminative than min(h1, h2)
    for fashion items where subtle color differences matter.
    
    Returns: similarity in [0, 1]
    """
    if len(hist_query) != len(hist_candidate) or len(hist_query) == 0:
        return 0.0

    # Bhattacharyya coefficient: sum of sqrt(p * q)
    bc = np.sum(np.sqrt(np.maximum(hist_query * hist_candidate, 0)))
    return float(np.clip(bc, 0.0, 1.0))


# ═════════════════════════════════════════════════════════════════
# MAIN RE-RANKER
# ═════════════════════════════════════════════════════════════════

class Reranker:
    """
    Three-stage re-ranking pipeline.
    
    The key insight: category filtering is a HARD gate (multiplicative),
    NOT a soft weighted component. A perfectly matching embedding for
    TROUSERS when we detected TOP should be killed, not gently penalized.
    
    Final formula:
        score = embed_sim * category_gate * (1 + color_bonus * w_color)
    
    Where:
        - embed_sim ∈ [0, 1]: normalized FAISS similarity
        - category_gate ∈ {0.05, 0.5, 1.0}: hard mask
        - color_bonus ∈ [0, 1]: block HSV similarity (additive boost)
    """

    def __init__(
        self,
        w_color: float = 0.25,
        color_enabled: bool = True,
    ):
        self.w_color = w_color
        self.color_enabled = color_enabled

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
        Full re-ranking with color histograms.
        
        Args:
            crop_image: RGB numpy array of the detected garment crop
            yolo_class: YOLO detection class ("top", "trousers", etc.)
            candidate_ids: product IDs from FAISS search
            candidate_scores: raw FAISS similarity scores
            candidate_images: product_id → RGB numpy array
            product_categories: product_id → category string
            top_k: number of results to return
        """
        # Normalize embedding scores to [0, 1]
        embed_norm = self._normalize_scores(candidate_scores)

        # Pre-compute query histogram
        query_hist = extract_block_histogram(crop_image) if self.color_enabled else None

        results = []
        for i, pid in enumerate(candidate_ids):
            # 1. Embedding similarity (base signal)
            e = embed_norm[i]

            # 2. Category hard mask (multiplicative gate)
            cat_str = product_categories.get(pid, "UNKNOWN")
            gate = category_mask_score(yolo_class, cat_str)

            # 3. Color bonus (additive boost for matching colors)
            color_bonus = 0.0
            if self.color_enabled and query_hist is not None:
                cand_img = candidate_images.get(pid)
                if cand_img is not None:
                    cand_hist = extract_block_histogram(cand_img)
                    color_bonus = compare_block_histograms(query_hist, cand_hist)

            # Final score: embed × gate × (1 + color_boost)
            score = e * gate * (1.0 + self.w_color * color_bonus)
            results.append((pid, float(score)))

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
        Fast re-ranking without color histograms.
        Uses embedding scores × category hard mask.
        For when catalog images aren't loaded in memory.
        """
        embed_norm = self._normalize_scores(candidate_scores)

        results = []
        for i, pid in enumerate(candidate_ids):
            e = embed_norm[i]
            cat_str = product_categories.get(pid, "UNKNOWN")
            gate = category_mask_score(yolo_class, cat_str)
            score = e * gate  # Multiplicative: wrong category → score ≈ 0
            results.append((pid, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def rerank_with_color(
        self,
        crop_image_rgb: np.ndarray,
        yolo_class: Optional[str],
        candidate_ids: List[str],
        candidate_scores: np.ndarray,
        product_images_dir: str,
        product_categories: Dict[str, str],
        top_k: int = 15,
    ) -> List[Tuple[str, float]]:
        """
        Re-ranking with lazy image loading from disk.
        Loads candidate images on-the-fly (slower but memory-efficient).
        """
        import os

        embed_norm = self._normalize_scores(candidate_scores)
        query_hist = extract_block_histogram(crop_image_rgb)

        results = []
        for i, pid in enumerate(candidate_ids):
            e = embed_norm[i]
            cat_str = product_categories.get(pid, "UNKNOWN")
            gate = category_mask_score(yolo_class, cat_str)

            # Only compute color for category-passing candidates (saves time)
            color_bonus = 0.0
            if gate > 0.1:  # Skip color for dead-penalized candidates
                img_path = os.path.join(product_images_dir, f"{pid}.jpg")
                if os.path.exists(img_path):
                    try:
                        cand_img = cv2.imread(img_path)
                        if cand_img is not None:
                            cand_img = cv2.cvtColor(cand_img, cv2.COLOR_BGR2RGB)
                            cand_hist = extract_block_histogram(cand_img)
                            color_bonus = compare_block_histograms(query_hist, cand_hist)
                    except Exception:
                        pass

            score = e * gate * (1.0 + self.w_color * color_bonus)
            results.append((pid, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0.01, 1.0]. Floor at 0.01 to avoid zero multiplication."""
        s_min = scores.min()
        s_max = scores.max()
        rng = s_max - s_min
        if rng < 1e-10:
            return np.ones_like(scores, dtype=np.float64)
        normalized = (scores - s_min) / rng
        return np.clip(normalized, 0.01, 1.0)
