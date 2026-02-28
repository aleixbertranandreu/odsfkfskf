"""
train_metric.py — Two-Phase Metric Learning Training for Fashion Visual Search
Phase 1: Frozen backbone, train embedding head only (10 epochs, LR=1e-3)
Phase 2: Full fine-tuning with cosine annealing (20 epochs, LR=1e-5 backbone / 1e-4 head)

Optimized for 6GB VRAM (RTX 4050):
- Mixed Precision (FP16)
- Batch size 16 with gradient accumulation (effective 32)
- ConvNeXt-Tiny (28M params)
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import FashionEmbedder
from dataset import TripletFashionDataset


def online_hard_negative_mining(anchor_emb, positive_emb, negative_emb, margin=0.3):
    """
    Compute triplet loss with semi-hard / hard negative selection within the batch.
    
    For each anchor-positive pair, we find the negative that is:
    - Closer to the anchor than the positive + margin (hard)
    - Or the hardest available negative in the batch
    
    This is more effective than random negatives because it forces the model
    to learn the most difficult distinctions.
    """
    # Standard triplet margin loss as baseline
    loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    base_loss = loss_fn(anchor_emb, positive_emb, negative_emb)
    
    # Additionally, mine harder negatives within the batch
    # Compute all pairwise distances between anchors and negatives
    batch_size = anchor_emb.size(0)
    
    if batch_size < 4:
        return base_loss
    
    # Distance from each anchor to all negatives in the batch
    dist_an = torch.cdist(anchor_emb, negative_emb, p=2)  # (B, B)
    dist_ap = torch.norm(anchor_emb - positive_emb, dim=1, keepdim=True)  # (B, 1)
    
    # For each anchor, find the hardest negative (closest to anchor)
    # Exclude the diagonal (same index = may be too easy or identity)
    mask = torch.eye(batch_size, device=anchor_emb.device).bool()
    dist_an_masked = dist_an.masked_fill(mask, float('inf'))
    
    # Hardest negative for each anchor
    hardest_neg_idx = dist_an_masked.argmin(dim=1)
    hardest_negatives = negative_emb[hardest_neg_idx]
    
    hard_loss = loss_fn(anchor_emb, positive_emb, hardest_negatives)
    
    # Blend: 50% base + 50% hard-mined
    return 0.5 * base_loss + 0.5 * hard_loss


def compute_val_mrr(model, val_loader, device):
    """Compute MRR@15 on validation set as a proxy for retrieval quality."""
    model.eval()
    
    all_anchor = []
    all_positive = []
    
    with torch.no_grad():
        for anchor, positive, _ in val_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            
            with autocast():
                a_emb = model(anchor)
                p_emb = model(positive)
            
            all_anchor.append(a_emb.cpu())
            all_positive.append(p_emb.cpu())
    
    all_anchor = torch.cat(all_anchor)
    all_positive = torch.cat(all_positive)
    
    # Compute cosine similarity matrix
    sim = torch.mm(all_anchor, all_positive.t()).numpy()
    
    # For each anchor (row), find the rank of its correct positive (diagonal)
    n = sim.shape[0]
    reciprocal_ranks = []
    for i in range(n):
        scores = sim[i]
        ranked = np.argsort(-scores)
        rank = np.where(ranked == i)[0]
        if len(rank) > 0 and rank[0] < 15:
            reciprocal_ranks.append(1.0 / (rank[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks)
    model.train()
    return mrr


def train(args):
    """Main training loop."""
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # ─── Paths ───
    matches_csv = args.train_csv if getattr(args, 'train_csv', None) else os.path.join(BASE_DIR, "data", "raw", "bundles_product_match_train.csv")
    products_csv = os.path.join(BASE_DIR, "data", "raw", "product_dataset.csv")
    bundles_csv = os.path.join(BASE_DIR, "data", "raw", "bundles_dataset.csv")
    product_images = args.products_dir if getattr(args, 'products_dir', None) else os.path.join(BASE_DIR, "data", "images", "products")
    bundle_images = args.bundles_dir if getattr(args, 'bundles_dir', None) else os.path.join(BASE_DIR, "data", "images", "bundles")
    checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")
    
    # ─── Dataset ───
    print("📊 Loading dataset...")
    full_dataset = TripletFashionDataset(
        matches_csv=matches_csv,
        products_csv=products_csv,
        bundles_csv=bundles_csv,
        product_images_dir=product_images,
        bundle_images_dir=bundle_images,
        img_size=args.img_size,
    )
    
    # Train/Val split (80/20)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * 0.2))
    n_train = n_total - n_val
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"   Train: {n_train} samples | Val: {n_val} samples")
    print(f"   Batch: {args.batch_size} | Effective: {args.batch_size * args.grad_accum}")
    
    # ─── Model ───
    print("🧠 Building model...")
    model = FashionEmbedder(embed_dim=args.embed_dim, pretrained=True).to(device)
    
    # Phase tracking
    total_phase1_epochs = args.phase1_epochs
    total_phase2_epochs = args.epochs - total_phase1_epochs
    
    # ─── Mixed Precision ───
    scaler = GradScaler()
    
    best_mrr = 0.0
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # ─── Phase transition ───
        if epoch == 0:
            model.freeze_backbone()
            optimizer = torch.optim.AdamW(
                model.embed_head.parameters(),
                lr=args.lr_head,
                weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=total_phase1_epochs)
            print(f"\n{'='*60}")
            print(f"📌 PHASE 1: Training embedding head only ({total_phase1_epochs} epochs)")
            print(f"{'='*60}")
        
        elif epoch == total_phase1_epochs:
            model.unfreeze_backbone()
            optimizer = torch.optim.AdamW(
                model.get_param_groups(
                    lr_backbone=args.lr_backbone,
                    lr_head=args.lr_head * 0.1
                ),
                weight_decay=1e-4
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=total_phase2_epochs)
            print(f"\n{'='*60}")
            print(f"🔥 PHASE 2: Full fine-tuning ({total_phase2_epochs} epochs)")
            print(f"{'='*60}")
        
        # ─── Training loop ───
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            anchor = anchor.to(device, non_blocking=True)
            positive = positive.to(device, non_blocking=True)
            negative = negative.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda'):
                a_emb = model(anchor)
                p_emb = model(positive)
                n_emb = model(negative)
                
                loss = online_hard_negative_mining(
                    a_emb, p_emb, n_emb, margin=args.margin
                )
                loss = loss / args.grad_accum  # Scale for accumulation
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                global_step += 1
            
            epoch_loss += loss.item() * args.grad_accum
            n_batches += 1
            
            if args.quick_test and batch_idx >= 5:
                break
        
        # Step scheduler
        scheduler.step()
        
        avg_loss = epoch_loss / max(n_batches, 1)
        
        # ─── Validation ───
        val_mrr = 0.0
        if not args.quick_test and (epoch + 1) % 2 == 0:
            val_mrr = compute_val_mrr(model, val_loader, device)
        
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        phase = "P1" if epoch < total_phase1_epochs else "P2"
        print(
            f"  [{phase}] Epoch {epoch+1:3d}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"MRR@15: {val_mrr:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
        
        # ─── Save best model ───
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            save_path = os.path.join(checkpoint_dir, "best_metric_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mrr': best_mrr,
                'embed_dim': args.embed_dim,
            }, save_path)
            print(f"  💾 New best model saved! MRR@15 = {best_mrr:.4f}")
        
        if args.quick_test and epoch >= 1:
            print("⚡ Quick test completed successfully!")
            break
    
    # ─── Final save ───
    final_path = os.path.join(checkpoint_dir, "final_metric_model.pt")
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'best_mrr': best_mrr,
        'embed_dim': args.embed_dim,
    }, final_path)
    print(f"\n🏆 Training complete! Best MRR@15: {best_mrr:.4f}")
    print(f"   Best model: {os.path.join(checkpoint_dir, 'best_metric_model.pt')}")
    print(f"   Final model: {final_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Metric Learning Model for Fashion Retrieval")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30, help="Total epochs (phase1 + phase2)")
    parser.add_argument("--phase1_epochs", type=int, default=10, help="Epochs for phase 1 (frozen backbone)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 fits 6GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--margin", type=float, default=0.3, help="Triplet margin")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--embed_dim", type=int, default=256, help="Embedding dimension")
    
    # Learning rates
    parser.add_argument("--lr_head", type=float, default=1e-3, help="LR for embedding head")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="LR for backbone in phase 2")
    
    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--train_csv", type=str, default=None)
    parser.add_argument("--products_dir", type=str, default=None)
    parser.add_argument("--bundles_dir", type=str, default=None)
    
    # Debug
    parser.add_argument("--quick_test", action="store_true", help="Quick 2-epoch smoke test")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
