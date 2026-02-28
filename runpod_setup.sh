#!/bin/bash
# ==============================================================================
# 🚀 HACKUDC 2026 - RUNPOD ELITE FINE-TUNING DEPLOYMENT
# ==============================================================================
# This script sets up a blank RunPod instance with a 24GB+ GPU to run the 
# hyper-optimized FashionCLIP contrastive learning pipeline (NT-Xent).
# ==============================================================================

set -e # Exit immediately if a command exits with a non-zero status

echo "🔥 Initializing RunPod Environment for Inditex Visual Search..."

# 1. Update and install basic system dependencies
echo "📦 Installing system dependencies (unzip)..."
apt-get update -y && apt-get install -y unzip

# 2. Install Python dependencies
echo "🐍 Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 3. Create dataset directories
echo "📁 Setting up project structure..."
mkdir -p data/raw data/images/bundles data/images/products data/embeddings checkpoints

# ------------------------------------------------------------------------------
# ☁️ DATASET DOWNLOAD (Action Required)
# ------------------------------------------------------------------------------
# NOTE: Upload your 'data.zip' to Google Drive, right click -> Get Link.
# Copy the ID (the long string of letters/numbers in the URL) and replace the ID below.
echo "⬇️ Downloading dataset from Google Drive..."
# Uncomment and replace YOUR_DRIVE_FILE_ID:
# gdown --id YOUR_DRIVE_FILE_ID -O data.zip

# 4. Decompress the dataset directly into the data/ folder
# echo "🗜️ Decompressing dataset..."
# unzip -q data.zip -d data/
# rm data.zip

# 5. Execute Elite Training Pipeline
echo "🚀 INICIANDO ENTRENAMIENTO CONTRASTIVO DE ÉLITE (Batch 128)..."
echo "   Model: patrickjohncyh/fashion-clip"
echo "   Loss: NT-Xent (InfoNCE) | Augmentation: Aggressive"

# We use batch_size 128 (requires ~16-20GB VRAM on ViT-B/32)
# num_workers=8 for blazing fast DataLoader (crucial for batch 128)
python src/finetune_clip.py \
    --batch_size 128 \
    --num_workers 8 \
    --epochs 10 \
    --save_every 20

echo "✅ ENTRENAMIENTO COMPLETADO. ¡PESOS GUARDADOS EN checkpoints/inditex_fashion_clip.pt!"
