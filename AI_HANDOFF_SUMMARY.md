# HackUDC 2026 - Inditex Challenge: State of the Project & Handoff
*Last updated: March 1, 2026*

## The Objective
Match "Bundles" (Outfits/Fashion editorials) to "Products" (Catalogue items) from the Zara database. 
Metric: **MRR@15 (Mean Reciprocal Rank at top 15).**
Goal: Break the 60% MRR threshold.

## The Mystery (The 18.57% Ceiling)
We are currently stuck at a Kaggle score of **18.57%**.
Despite iterating through increasingly complex computer vision and metadata architectures, the score refuses to move a single decimal point from our best base submission. It appears the evaluation metric or the dataset holds a hidden signal we remain blind to.

## What We Have Built & Tried (The 5-Signal Grand Ensemble)

1. **FashionCLIP (Zero-Shot & Fine-Tuned)**
   - **Approach:** Used `patrickjohncyh/fashion-clip` (ViT-B/32) to extract 512-dim embeddings for all 27K products and 455 test bundles.
   - **Fine-tuning:** Fine-tuned the Vision Encoder on the 6.5K train matches using `TripletMarginLoss` with Hard Negative Mining.
   - **Result:** This achieved our peak score of **18.57%**.

2. **ConvNeXt-Base (Texture Specialist)**
   - **Approach:** Trained a ConvNeXt-Base model from scratch on RunPod (A40 GPU) using the dataset to focus exclusively on local textures, fabrics, and patterns rather than global shapes. 
   - **Result:** Achieved 0.11 MRR in local validation. When integrated into the ensemble, it did not move the Kaggle score.

3. **YOLOv8 Local Accessory Detection**
   - **Approach:** Noticed that shoes, bags, and small accessories accounted for 45% of the items but were missed by global embeddings. Fine-tuned YOLOv8m locally on an RTX 4050 to extract bounding boxes of accessories in the bundles.
   - **Result:** Successfully extracted crops for Re-Ranking.

4. **HSV Color Histograms (Mathematical Matching)**
   - **Approach:** Pre-computed HSV color histograms for all 27K products. For each YOLO crop in a bundle, mathematically compared its color to the catalogue products using OpenCV Intersection.
   - **Result:** Effectively pushed matching colors to the top. Did not impact the Kaggle score.

5. **The URL Metadata Hack (Zara Style IDs)**
   - **Approach:** Discovered that Zara embeds their 11-digit internal product codes inside the image URLs (e.g., `.../02878302711-p.jpg`).
   - **Logic:** Verified that ~30% of matching Bundle-Product pairs shared the exact same 4-digit family code or 11-digit item code in their URLs. 
   - **Implementation:** Added a regex parser to `ensemble.py`. If a product shared a URL code with the bundle, its score was multiplied by up to 10x to force it to Rank #1.
   - **Result:** Completely failed to move the Kaggle score from 18.57%, suggesting either Kaggle's test set URLs are deliberately obfuscated or the true positive matches do not rely on this metadata.

### The Grand Ensemble (`ensemble.py`)
We fused all the above signals using **Reciprocal Rank Fusion (RRF)**:
`Score = RRF(FashionCLIP, ConvNeXt) * YOLOCategory_Mask * ColorSimilarity * URL_Multiplier`
The output successfully truncated to exactly 15 predictions per bundle (6,825 rows). **The score remained 18.57%.**

---

## 🛑 HYPOTHESIS FOR THE NEXT AI: Why are we stuck? 🛑

If a 5-layer pipeline involving fine-tuned Vision Transformers, CNNs, Object Detection, Color Math, and Metadata Hacking cannot break 18.57%, we are fundamentally misunderstanding the problem.

**Please investigate the following avenues immediately:**

1. **Co-occurrence / Graph Neural Networks (The "Bought Together" Signal)**
   - Is it possible that the products in a bundle are not meant to be visually found *in* the bundle image, but rather *stylistically match* the bundle? 
   - In `bundles_product_match_train.csv`, bundles have multiple products. Do these products commonly co-occur? (e.g., If Product A is in a bundle, Product B is 90% likely to also be there). We should build a Graph or Apriori Associative Rule system based on the training data.

2. **Kaggle Evaluation Bug / Data Leak Check**
   - Have we thoroughly checked the `product_asset_id` generation? Are there duplicate products with different IDs? Check the `product_dataset.csv` for exact description duplicates.
   - Are we sorting the CSV incorrectly before submission? Kaggle requires the top prediction to be first.

3. **Textual Descriptions over Images**
   - We have completely ignored `product_description`.
   - The top teams might be doing zero-shot text-to-image matching: extracting the description of the product, generating a CLIP text embedding, and comparing it to the Bundle image embedding, rather than Image-to-Image.

4. **The "Outfits" Dataset Illusion**
   - In the training set, we noticed the `bundle_id_section`. Is it possible the scoring is heavily skewed towards one section (e.g., Women vs Men vs Kids)? 

### Directory Structure to Know
- `src/`: Contains all `inference`, `train`, `ensemble`, and `dataset` scripts.
- `data/raw/`: Contains `product_dataset.csv`, `bundles_dataset.csv`, `bundles_product_match_train.csv`, `bundles_product_match_test.csv`.
- `models/`: HuggingFace caches and checkpoints.
- `GRAND_ENSEMBLE_HackUDC_submission.csv`: Our final, failed monolith attempt.
- `submission_fashionclip_finetuned.csv`: Our current 18.57% baseline.

**Good luck. Break the 18% barrier.**
