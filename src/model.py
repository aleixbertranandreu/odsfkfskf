"""
model.py — ConvNeXt-Tiny Metric Learning Backbone
Designed for fine-grained fashion retrieval (Inditex Visual Search)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class FashionEmbedder(nn.Module):
    """
    ConvNeXt-Tiny backbone with a 256-dim embedding head.
    Outputs L2-normalized embeddings for metric learning.
    
    Why ConvNeXt:
    - Convolution hierarchy preserves local texture patterns
      (weave, fabric type, stitching) better than ViT's patch attention
    - 28M params → fits comfortably in 6GB VRAM with FP16
    - ImageNet-22K pretraining provides strong foundation
    """
    
    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        # Load ConvNeXt-Base with best available pretrained weights for cloud GPU
        if pretrained:
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1
            self.backbone = models.convnext_base(weights=weights)
        else:
            self.backbone = models.convnext_base(weights=None)
        
        # ConvNeXt-Base feature dim = 1024
        backbone_dim = 1024
        
        # Remove the original classification head
        self.backbone.classifier = nn.Identity()
        
        # Embedding projection head
        self.embed_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        
        # Initialize the projection layer
        nn.init.kaiming_normal_(self.embed_head[2].weight)
        nn.init.zeros_(self.embed_head[2].bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) image tensor, normalized for ImageNet
        Returns:
            (B, embed_dim) L2-normalized embeddings
        """
        # Extract features through ConvNeXt stages
        # We access the features BEFORE the classifier
        features = self.backbone.features(x)
        
        # Project to embedding space
        embeddings = self.embed_head(features)
        
        # L2 normalize — critical for cosine similarity / inner product search
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def freeze_backbone(self):
        """
        Phase 1: Freeze all backbone parameters.
        Only train the embedding head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.embed_head.parameters():
            param.requires_grad = True
        print("INFO: Backbone FROZEN — training only embed_head")
    
    def unfreeze_backbone(self):
        """
        Phase 2: Unfreeze full backbone for end-to-end fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("INFO: Backbone UNFROZEN — full fine-tuning active")
    
    def get_param_groups(self, lr_backbone: float = 1e-5, lr_head: float = 1e-3):
        """
        Returns parameter groups with different learning rates.
        Backbone gets a smaller LR to preserve pretrained features.
        """
        backbone_params = list(self.backbone.parameters())
        head_params = list(self.embed_head.parameters())
        
        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]


if __name__ == "__main__":
    # Quick sanity check
    model = FashionEmbedder(embed_dim=256, pretrained=True)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"INFO: SUCCESS: Output shape: {out.shape}")  # Expected: (4, 256)
    print(f"INFO: SUCCESS: L2 norm check: {torch.norm(out, dim=1)}")  # Should be ~1.0
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"INFO: Total params: {total_params/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")
