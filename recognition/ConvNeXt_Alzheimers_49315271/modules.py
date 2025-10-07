# modules.py
# ConvNeXt model for Alzheimer's classification

import torch
import torch.nn as nn
import torchvision.models as models

class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt model for binary classification of Alzheimer's Disease
    Uses pretrained ConvNeXt-Tiny with modified classifier head
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Number of output classes (2 for AD/NC)
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: If True, freeze feature extraction layers
        """
        super(ConvNeXtClassifier, self).__init__()
        
        # Load pretrained ConvNeXt-Tiny
        if pretrained:
            for param in list(self.model.features.parameters())[:100]:
                param.requires_grad = False
            print("Loaded pretrained ConvNeXt-Tiny weights")
        else:
            self.model = models.convnext_tiny(weights=None)
            print("Initialized ConvNeXt-Tiny without pretrained weights")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
            print("Frozen backbone layers")
        
        # Modify classifier head with dropout for regularization
        in_features = self.model.classifier[2].in_features
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # LayerNorm
            self.model.classifier[1],  # Flatten
            nn.Dropout(0.7),           # Added dropout
            nn.Linear(in_features, num_classes)
        )
        print(f"Modified classifier: {in_features} -> {num_classes} classes")
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("Unfrozen all layers")


def get_model(device='cuda', pretrained=True, freeze_backbone=False):
    """
    Helper function to create and initialize model
    
    Args:
        device: Device to load model on ('cuda' or 'cpu')
        pretrained: Use pretrained weights
        freeze_backbone: Freeze feature extraction layers
    
    Returns:
        model: ConvNeXt model ready for training
    """
    model = ConvNeXtClassifier(
        num_classes=2,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    
    return model