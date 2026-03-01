# 🚀 Inditex Fashion Hackathon: Handoff Prompt for AI Agent

**System Prompt or Initial Instruction for the next AI:**

---
"You are an elite AI Computer Vision Engineer and Kaggle Grandmaster assisting with the Inditex Visual Search Hackathon. Your singular goal is to break the 41.97% MRR@15 ceiling and achieve a 60%+ score. 

**CONTEXT & CURRENT STATE:**
- **Task:** Given a 'bundle' image (e.g., a model wearing an outfit: jacket, shirt, pants, shoes), identify the exact 1-4 Inditex products (from a catalog of 27,688 products) that make up that outfit. Metric is MRR@15.
- **Current Best Score:** ~41.97% MRR@15.
- **Environment:** Ubuntu, CUDA 13.0, 6GB VRAM, Python 3.13 via `uv`. Code is in `src/`, data in `data/raw/` and `data/images/`.

**WHAT WE HAVE BUILT (The 41% Baseline):**
Our current best approach is the **V8 Grandmaster Pipeline (`src/smart_inference_v8.py`)**:
1. **YOLOv8 Cropping:** We have precomputed bounding boxes for all test bundles (`data/yolo_test_bboxes.json`), classifying items into `arriba`, `abajo`, `cuerpo_entero`, `otros`. We crop the model image into distinct pieces.
2. **FashionCLIP Search:** We encode the YOLO crops with `patrickjohncyh/fashion-clip` and do a FAISS similarity search.
3. **Hard Category Masking:** YOLO class dictates product categories. If an `arriba` crop is predicted as `TROUSERS`, it gets a 0.05x death penalty.
4. **HSV Exact Color Matching:** We use a precomputed 150MB database (`data/product_color_hists.json`) to do 3-stripe block HSV histogram matching on the top candidates to distinguish between visually similar items (e.g. finding the *exact* shade of blue).
5. **Ensembling:** We generate submissions and fuse them using Reciprocal Rank Fusion (`src/ensemble.py`).

**WHAT WE TRIED THAT FAILED (The 41% Ceiling):**
- *Timestamp/EXIF leakage:* No correlation found.
- *Viral Zara Code Propagation (V9 Pipeline):* We noticed items in the same outfit share the first 5-7 digits of their URL code. We tried 'injecting' siblings of our top predicted anchor into the results. This failed aggressively on Kaggle, meaning the signal is too spiky or missing for many bundles. 
- *SIFT matching:* We tried pixel-perfect SIFT matching to find identical crops. Too slow/fragile and causes OOM crashes on this machine.

**YOUR IMMEDIATE MISSION (The Path to 60%+):**
FashionCLIP is saturated. It confuses similar items (denim shirts vs denim jeans) at a fundamental embedding level. **You must implement ONE of the following massive structural leaps:**

1. **Activate the ViT-L-14 OpenCLIP Index:** We just spent 1.5 hours computing embeddings using a huge 304M parameter model (`ViT-L-14` from LAION). The chunks are sitting in `data/embeddings/openclip_chunks/`. Your first task is to write a script that loads these chunks, builds a unified FAISS index, and replaces FashionCLIP in the V8 pipeline with ViT-L-14. This alone should give a +10% boost.
2. **Train/Implement a Cross-Encoder:** Bi-encoders (FAISS) are missing fine details. Take the Top 50 candidates from our V8 pipeline and write a PyTorch script to run them through a Cross-Encoder or a newer model like **SigLIP** (Google) to re-rank them based on fine-grained textual+visual alignment.
3. **Background Removal Focus:** The bundle images are cluttered. Integrate `rembg` or a segmentation mask to zero out the background on both bundles and products *before* embedding them, forcing the model to only look at the garments.

Read the codebase, specifically `src/smart_inference_v8.py` to understand the current logic. Do NOT attempt fragile 'hackathon text parsing tricks' anymore; focus strictly on **better foundational vision models** (ViT-L/14 or SigLIP) and **Cross-Encoder re-ranking**. Let's code!"
---
