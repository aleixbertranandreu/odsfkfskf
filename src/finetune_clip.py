"""
finetune_clip.py — Elite Contrastive Learning Pipeline for Visual Search
════════════════════════════════════════════════════════════════════════
Kaggle Grandmaster Techniques:
1. In-Batch Hard Negative Mining (InfoNCE / NT-Xent Loss) with low temperature (tau=0.05).
   Forces model to learn fine-grained details by penalizing confusion between similar items.
2. Aggressive Data Augmentation (RandomErasing, ColorJitter, RandomAffine).
   Destroys street lighting and forces texture focus via occlusion.
3. Multi-Scale / Patch Learning via aggressive 70% central cropping.
   Ignores background context and faces.
4. Discriminative Learning Rates.
   1e-6 for pre-trained backbone features, 1e-4 for the projection head.
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


class CentralCropRatio(object):
    """Crop the central X% of the image to remove background and faces."""
    def __init__(self, ratio=0.7):
        self.ratio = ratio

    def __call__(self, img):
        w, h = img.size
        new_w, new_h = int(w * self.ratio), int(h * self.ratio)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return img.crop((left, top, left + new_w, top + new_h))


class InditexPairDataset(Dataset):
    def __init__(self, train_csv, bundles_dir, products_dir):
        self.df = pd.read_csv(train_csv)
        self.bundles_dir = bundles_dir
        self.products_dir = products_dir
        
        # ─── 1 & 2. Aggressive Data Augmentation & Multi-Scale ───
        self.bundle_transform = transforms.Compose([
            CentralCropRatio(0.7),  # Central crop to ignore background/face
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Deform shape
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Destroy lighting
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'), # Occlusion / Cutout
        ])

        # Product is usually clean flat-lay; apply only slight transforms to prevent overfitting
        self.product_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        
        # Verify images exist
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

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        b_path = os.path.join(self.bundles_dir, f"{row['bundle_asset_id']}.jpg")
        p_path = os.path.join(self.products_dir, f"{row['product_asset_id']}.jpg")
        
        try:
            b_img = Image.open(b_path).convert('RGB')
            p_img = Image.open(p_path).convert('RGB')
            anchor = self.bundle_transform(b_img)
            positive = self.product_transform(p_img)
        except Exception:
            anchor = torch.zeros(3, 224, 224)
            positive = torch.zeros(3, 224, 224)
            
        return anchor, positive


class InfoNCELoss(nn.Module):
    """
    3. In-Batch Hard Negative Mining (InfoNCE / NT-Xent)
    Computes similarities across ALL items in the batch.
    """
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features_a, features_p):
        B = features_a.size(0)
        # Similarity matrix: (B, D) @ (D, B) -> (B, B)
        sim_matrix = torch.matmul(features_a, features_p.t()) / self.temperature
        
        # Labels are simply the diagonal (anchor matches its exact positive)
        labels = torch.arange(B, device=features_a.device)
        
        # Symmetric Cross Entropy
        loss_a = F.cross_entropy(sim_matrix, labels)
        loss_p = F.cross_entropy(sim_matrix.t(), labels)
        return (loss_a + loss_p) / 2


def train_clip(args):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Use CLI args if provided, otherwise fallback to default structure
    train_csv = args.train_csv or os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    bundles_dir = args.bundles_dir or os.path.join(BASE_DIR, "data", "images", "bundles")
    products_dir = args.products_dir or os.path.join(BASE_DIR, "data", "images", "products")
    
    if not os.path.exists(train_csv):
        print(f"❌ Error: CSV de entrenamiento no encontrado en {train_csv}")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Elite Fine-tuning on device: {device}")
    print(f"   CSV: {train_csv}")
    print(f"   Bundles: {bundles_dir}")
    print(f"   Products: {products_dir}")
    
    # ─── LOAD MODEL ───
    # ... rest remains identical ...
    print(f"🧠 Loading base model: {args.model_name}")
    model = CLIPModel.from_pretrained(args.model_name).to(device)
    
    # Freeze text encoder
    for param in model.text_model.parameters():
        param.requires_grad = False
        
    # Unfreeze only the last N vision layers and the projection
    for name, param in model.vision_model.named_parameters():
        if "layers" in name:
            layer_num = int(name.split("layers.")[1].split(".")[0])
            if layer_num >= (model.vision_model.config.num_hidden_layers - args.unfreeze_layers):
                param.requires_grad = True
            else:
                param.requires_grad = False
        elif "post_layernorm" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    for param in model.visual_projection.parameters():
        param.requires_grad = True

    # ─── 4. Discriminative Learning Rates ───
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual_projection" in name or "post_layernorm" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': args.lr_backbone},
        {'params': head_params, 'lr': args.lr_head}
    ], weight_decay=args.weight_decay)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔧 Trainable parameters: {trainable_params:,}")

    # ─── DATASET & LOADER ───
    # drop_last=True ensures consistent InfoNCE batch size behavior
    dataset = InditexPairDataset(train_csv, bundles_dir, products_dir)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    # ─── LOSS & SCHEDULER ───
    criterion = InfoNCELoss(temperature=args.tau)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs * len(loader))
    
    # ─── TRAINING LOOP ───
    scaler = torch.amp.GradScaler('cuda')
    
    save_path = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(save_path, exist_ok=True)
    model_save_file = os.path.join(save_path, "inditex_fashion_clip.pt")
    
    print("🔥 Starting Contrastive INFO-NCE Training...")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, (anchor, positive) in enumerate(pbar):
            anchor = anchor.to(device)
            positive = positive.to(device)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                # Extract embeddings
                a_out = model.vision_model(pixel_values=anchor)
                a_emb = model.visual_projection(a_out.pooler_output)
                a_emb = F.normalize(a_emb, p=2, dim=-1)
                
                p_out = model.vision_model(pixel_values=positive)
                p_emb = model.visual_projection(p_out.pooler_output)
                p_emb = F.normalize(p_emb, p=2, dim=-1)
                
                # InfoNCE In-Batch Hard Negative Loss
                loss = criterion(a_emb, p_emb)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'lr_backbone': f"{optimizer.param_groups[0]['lr']:.1e}",
                'lr_head': f"{optimizer.param_groups[1]['lr']:.1e}"
            })
            
            # Save periodic checkpoints
            global_step = epoch * len(loader) + batch_idx
            if (global_step + 1) % args.save_every == 0:
                step_avg = total_loss / (batch_idx + 1)
                print(f"\n💾 Checkpoint @ step {global_step+1} (Avg Loss: {step_avg:.4f})")
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': step_avg,
                }, model_save_file)

    print("✅ Training complete! Rebuild index and run inference next.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="patrickjohncyh/fashion-clip")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_backbone", type=float, default=1e-6)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05, help="Temperature for InfoNCE")
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--unfreeze_layers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=50)
    
    # Data paths
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--bundles_dir", type=str, default=None)
    parser.add_argument("--products_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    train_clip(args)
