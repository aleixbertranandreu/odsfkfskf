"""
finetune_clip.py — Test-Time Fine-Tuning of FashionCLIP
═══════════════════════════════════════════════════════════════

The top teams use the ~6,500 training pairs provided by Inditex
to teach the model the specific visual language and domain gap
of this dataset (Model/Street vs Studio/Flatlay).

We fine-tune the patrickjohncyh/fashion-clip vision encoder using
TripletMarginLoss:
- Anchor: Bundle image
- Positive: Correct Product image
- Negative: Random (or hard) Product image
"""
import argparse
import os
import random
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


class InditexTripletDataset(Dataset):
    def __init__(self, train_csv, bundles_dir, products_dir, processor, transform=None):
        self.df = pd.read_csv(train_csv)
        self.bundles_dir = bundles_dir
        self.products_dir = products_dir
        self.processor = processor
        
        # Keep track of all unique products to sample negatives
        self.all_products = self.df['product_asset_id'].unique().tolist()
        
        # Filter out rows where images don't exist
        print("🔍 Verifying images exist...")
        valid_rows = []
        for _, row in self.df.iterrows():
            b_path = os.path.join(bundles_dir, f"{row['bundle_asset_id']}.jpg")
            p_path = os.path.join(products_dir, f"{row['product_asset_id']}.jpg")
            if os.path.exists(b_path) and os.path.exists(p_path):
                valid_rows.append(row)
        
        self.df = pd.DataFrame(valid_rows)
        print(f"📦 Valid training pairs: {len(self.df)}")

    def __len__(self):
        return len(self.df)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        # Standard CLIP preprocessing
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        b_id = row['bundle_asset_id']
        p_id = row['product_asset_id']
        
        b_path = os.path.join(self.bundles_dir, f"{b_id}.jpg")
        p_path = os.path.join(self.products_dir, f"{p_id}.jpg")
        
        # Positive Pair
        try:
            anchor = self._load_img(b_path)
            positive = self._load_img(p_path)
        except Exception as e:
            # Fallback for corrupt images during training
            anchor = torch.zeros(3, 224, 224)
            positive = torch.zeros(3, 224, 224)
            
        # Negative Pair (random product that is not the positive)
        neg_id = random.choice(self.all_products)
        while neg_id == p_id:
            neg_id = random.choice(self.all_products)
            
        neg_path = os.path.join(self.products_dir, f"{neg_id}.jpg")
        try:
            negative = self._load_img(neg_path)
        except Exception:
            negative = torch.zeros(3, 224, 224)
            
        return anchor, positive, negative


def train_clip(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    bundles_dir = os.path.join(BASE_DIR, "data", "images", "bundles")
    products_dir = os.path.join(BASE_DIR, "data", "images", "products")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Fine-tuning on device: {device}")
    
    # ─── 1. LOAD MODEL ───
    print(f"🧠 Loading base model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.model_name)
    
    # Freeze text encoder and base vision layers to save memory & prevent catastrophic forgetting
    for param in model.text_model.parameters():
        param.requires_grad = False
        
    for name, param in model.vision_model.named_parameters():
        # Unfreeze only the last N layers of the vision encoder
        if "layers" in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            if layer_num >= (model.vision_model.config.num_hidden_layers - args.unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif "post_layernorm" in name or "visual_projection" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable parameters: {trainable_params:,}")

    # ─── 2. DATASET & LOADER ───
    dataset = InditexTripletDataset(train_csv, bundles_dir, products_dir, processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    # ─── 3. LOSS & OPTIMIZER ───
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # ─── 4. TRAINING LOOP ───
    scaler = torch.amp.GradScaler('cuda')
    
    best_loss = float('inf')
    save_path = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_path, exist_ok=True)
    model_save_file = os.path.join(save_path, "inditex_fashion_clip.pt")
    
    print("🔥 Starting training loop...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (anchor, positive, negative) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Pass pixel values through vision model
                a_out = model.vision_model(pixel_values=anchor)
                a_emb = model.visual_projection(a_out.pooler_output)
                a_emb = a_emb / a_emb.norm(dim=-1, keepdim=True)
                
                p_out = model.vision_model(pixel_values=positive)
                p_emb = model.visual_projection(p_out.pooler_output)
                p_emb = p_emb / p_emb.norm(dim=-1, keepdim=True)
                
                n_out = model.vision_model(pixel_values=negative)
                n_emb = model.visual_projection(n_out.pooler_output)
                n_emb = n_emb / n_emb.norm(dim=-1, keepdim=True)
                
                # Triplet Loss!
                loss = criterion(a_emb, p_emb, n_emb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            # Save periodic checkpoints every 50 steps to not wait for full epochs
            global_step = epoch * len(loader) + batch_idx
            if (global_step + 1) % 50 == 0:
                step_avg = total_loss / (batch_idx + 1)
                print(f"\n💾 Saving periodic checkpoint at step {global_step+1} (Loss: {step_avg:.4f})")
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': step_avg,
                }, model_save_file)
            
        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f"📈 Epoch {epoch+1}/{args.epochs} - Avg Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f"💾 Saving new best model to {model_save_file}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_save_file)

    print("✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="patrickjohncyh/fashion-clip")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--unfreeze_layers", type=int, default=4, help="Number of final vision layers to unfreeze")
    args = parser.parse_args()
    
    train_clip(args)
