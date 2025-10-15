# modules.py
# ConvNeXt-Small model for Alzheimer's classification
# Includes Label Smoothing loss and MixUp augmentation utilities

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    Instead of using hard labels [0, 1] or [1, 0], we use soft labels
    like [0.05, 0.95] or [0.95, 0.05]. This prevents the model from
    becoming overconfident and improves generalization.
    
    Formula for label smoothing:
        y_smooth = (1 - smoothing) * y_true + smoothing / num_classes
    
    Where:
        - y_true is the one-hot encoded true label
        - smoothing is the smoothing parameter (e.g., 0.1)
        - num_classes is the number of classes (2 for AD/NC)
    
    Example with smoothing=0.1:
        Hard label: [0, 1] → Soft label: [0.05, 0.95]
        Hard label: [1, 0] → Soft label: [0.95, 0.05]
    
    This helps because:
    1. Prevents overconfidence on training data
    2. Improves calibration of predictions
    3. Better generalization to test distribution
    
    Args:
        smoothing: Smoothing factor (default: 0.1)
                   Higher = more smoothing, less confident predictions
        num_classes: Number of classes (default: 2 for binary classification)
    """
    
    def __init__(self, smoothing=0.1, num_classes=2):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) of shape (batch_size, num_classes)
            target: Ground truth labels of shape (batch_size,)
        
        Returns:
            Smoothed cross entropy loss
        """
        # Convert logits to log probabilities
        pred = F.log_softmax(pred, dim=1)
        
        # Create smoothed label distribution
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Calculate loss
        loss = torch.mean(torch.sum(-true_dist * pred, dim=1))
        
        return loss


class MixUpAugmentation:
    """
    MixUp data augmentation utility.
    
    MixUp creates virtual training examples by mixing pairs of examples
    and their labels. This forces the model to learn more robust features
    that generalize better to new distributions.
    
    Formula:
        x_mixed = lambda * x_i + (1 - lambda) * x_j
        y_mixed = lambda * y_i + (1 - lambda) * y_j
    
    Where:
        - x_i, x_j are two random training images
        - y_i, y_j are their corresponding labels
        - lambda is sampled from Beta(alpha, alpha) distribution
    
    Example with lambda=0.7:
        - Mixed image = 70% of image1 + 30% of image2
        - Mixed label = 70% of label1 + 30% of label2
    
    Args:
        alpha: Beta distribution parameter (default: 0.4)
               Higher alpha = more mixing (less conservative)
               Lower alpha = less mixing (more conservative)
    """
    
    def __init__(self, alpha=0.4):
        self.alpha = alpha
    
    def mixup_data(self, x, y):
        """
        Apply MixUp augmentation to a batch.
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
            y: Labels of shape (batch_size,)
        
        Returns:
            mixed_x: Mixed images
            y_a: Original labels
            y_b: Permuted labels
            lam: Mixing coefficient
        """
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        batch_size = x.size(0)
        
        # Generate random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix images: x_mixed = lam * x + (1 - lam) * x[shuffled]
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Return both sets of labels for loss calculation
        y_a = y
        y_b = y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Calculate MixUp loss.
        
        Loss = lambda * loss(pred, y_a) + (1 - lambda) * loss(pred, y_b)
        
        Args:
            criterion: Loss function
            pred: Model predictions
            y_a: First set of labels
            y_b: Second set of labels
            lam: Mixing coefficient
        
        Returns:
            Mixed loss value
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Small model for binary classification of Alzheimer's Disease.
    
    ConvNeXt is a modern CNN architecture that achieves strong performance
    by using design principles from Vision Transformers while maintaining
    the efficiency of CNNs.
    
    Architecture:
        - Backbone: ConvNeXt-Small pretrained on ImageNet-1K
        - Parameters: ~50M (more capacity than Tiny's ~28M)
        - Classifier: Custom head with dropout for regularization
    
    The model is initialized with ImageNet pretrained weights and
    fine-tuned for Alzheimer's classification.
    
    Args:
        num_classes: Number of output classes (default: 2 for AD/NC)
        pretrained: Use ImageNet pretrained weights (default: True)
        dropout: Dropout rate in classifier head (default: 0.7)
    """
    
    def __init__(self, num_classes=2, pretrained=True, dropout=0.7):
        super(ConvNeXtClassifier, self).__init__()
        
        # Load ConvNeXt-Small with pretrained weights
        if pretrained:
            self.model = models.convnext_small(weights='IMAGENET1K_V1')
            print("✓ Loaded ConvNeXt-Small with ImageNet pretrained weights")
        else:
            self.model = models.convnext_small(weights=None)
            print("✓ Initialized ConvNeXt-Small without pretrained weights")
        
        # Get the number of input features to the classifier
        in_features = self.model.classifier[2].in_features
        
        # Replace the classifier head
        # Original: [LayerNorm, Flatten, Linear(768 -> 1000)]
        # Modified: [LayerNorm, Flatten, Dropout, Linear(768 -> 2)]
        self.model.classifier = nn.Sequential(
            self.model.classifier[0],  # LayerNorm
            self.model.classifier[1],  # Flatten
            nn.Dropout(p=dropout),     # High dropout for regularization
            nn.Linear(in_features, num_classes)
        )
        
        print(f"✓ Modified classifier head: {in_features} -> {num_classes} classes")
        print(f"✓ Dropout rate: {dropout}")
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, 3, 224, 224)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def freeze_backbone(self):
        """
        Freeze the feature extraction layers (backbone).
        Only the classifier head will be trainable.
        
        Used in Stage 1 training for faster convergence.
        """
        for param in self.model.features.parameters():
            param.requires_grad = False
        print("✓ Froze backbone layers (features)")
    
    def unfreeze_backbone(self):
        """
        Unfreeze all layers for full model fine-tuning.
        
        Used in Stage 2 training for better adaptation to target domain.
        """
        for param in self.model.parameters():
            param.requires_grad = True
        print("✓ Unfroze all layers")


def get_model(model_size='small', num_classes=2, dropout=0.7, pretrained=True, device='cuda'):
    """
    Helper function to create and initialize the model.
    
    This function creates a ConvNeXt model, moves it to the specified device,
    and prints a summary of model parameters.
    
    Args:
        model_size: Model variant - 'tiny', 'small', or 'base' (default: 'small')
        num_classes: Number of output classes (default: 2)
        dropout: Dropout rate in classifier (default: 0.7)
        pretrained: Use ImageNet pretrained weights (default: True)
        device: Device to load model on - 'cuda' or 'cpu'
    
    Returns:
        model: Initialized ConvNeXt model ready for training
    """
    
    print("\n" + "="*80)
    print("MODEL INITIALIZATION")
    print("="*80)
    
    # Create model based on size
    if model_size == 'tiny':
        # ConvNeXt-Tiny: ~28M parameters
        base_model = models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None)
        print("Using ConvNeXt-Tiny (~28M parameters)")
    elif model_size == 'small':
        # ConvNeXt-Small: ~50M parameters
        base_model = models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None)
        print("Using ConvNeXt-Small (~50M parameters)")
    elif model_size == 'base':
        # ConvNeXt-Base: ~89M parameters
        base_model = models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None)
        print("Using ConvNeXt-Base (~89M parameters)")
    else:
        raise ValueError(f"Unknown model size: {model_size}. Choose 'tiny', 'small', or 'base'")
    
    # Modify classifier head
    in_features = base_model.classifier[2].in_features
    base_model.classifier = nn.Sequential(
        base_model.classifier[0],  # LayerNorm
        base_model.classifier[1],  # Flatten
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    
    # Move to device
    model = base_model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Dropout rate: {dropout}")
    print(f"  Device: {device}")
    print("="*80 + "\n")
    
    return model


def get_loss_function(loss_type='label_smoothing', smoothing=0.15):
    """
    Get the loss function for training.
    
    Args:
        loss_type: Type of loss - 'cross_entropy' or 'label_smoothing'
        smoothing: Smoothing parameter for label smoothing (default: 0.15)
    
    Returns:
        Loss function
    """
    if loss_type == 'cross_entropy':
        print(f"Using CrossEntropyLoss")
        return nn.CrossEntropyLoss()
    elif loss_type == 'label_smoothing':
        print(f"Using Label Smoothing Loss (smoothing={smoothing})")
        return LabelSmoothingLoss(smoothing=smoothing, num_classes=2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def count_parameters(model):
    """
    Count trainable and total parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        total: Total number of parameters
        trainable: Number of trainable parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


if __name__ == "__main__":
    """
    Test the model components
    """
    print("Testing model components...\n")
    
    # Test model creation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(model_size='small', dropout=0.7, device=device)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    
    # Test label smoothing loss
    print("\n" + "="*80)
    print("Testing Label Smoothing Loss...")
    criterion = LabelSmoothingLoss(smoothing=0.15, num_classes=2)
    dummy_labels = torch.tensor([0, 1, 0, 1]).to(device)
    loss = criterion(output, dummy_labels)
    print(f"Loss value: {loss.item():.4f}")
    
    # Test MixUp
    print("\n" + "="*80)
    print("Testing MixUp Augmentation...")
    mixup = MixUpAugmentation(alpha=0.4)
    mixed_x, y_a, y_b, lam = mixup.mixup_data(dummy_input, dummy_labels)
    print(f"Original input shape: {dummy_input.shape}")
    print(f"Mixed input shape: {mixed_x.shape}")
    print(f"Lambda (mixing coefficient): {lam:.4f}")
    print(f"Original labels: {dummy_labels.tolist()}")
    print(f"Labels A: {y_a.tolist()}")
    print(f"Labels B: {y_b.tolist()}")
    
    # Test MixUp loss
    output_mixed = model(mixed_x)
    loss_mixed = mixup.mixup_criterion(criterion, output_mixed, y_a, y_b, lam)
    print(f"MixUp loss value: {loss_mixed.item():.4f}")
    
    print("\n✓ All model components tested successfully!")
