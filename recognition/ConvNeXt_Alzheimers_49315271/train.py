# train.py
# Training script for Alzheimer's classification with ConvNeXt and Focal Loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import time
import copy


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Returns:
        avg_loss: Average training loss
        accuracy: Training accuracy
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Returns:
        avg_loss: Average validation loss
        accuracy: Validation accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, patience=10, save_path='best_model.pth'):
    """
    Main training loop with early stopping
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function (Focal Loss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of epochs
        device: Device to train on
        patience: Early stopping patience
        save_path: Path to save best model
    
    Returns:
        history: Dictionary containing training history
        best_model: Best model state dict
    """
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print(f"Training on {device}")
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12}")
    print("-" * 80)
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduler step
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<12.6f}")
        
        # Check if best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, save_path)
            print(f"    → New best model saved! Val Acc: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
            break
    
    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return history, model


def train_two_stage(model, train_loader, val_loader, num_epochs_frozen=10, num_epochs_finetune=10, 
                    lr_frozen=1e-3, lr_finetune=1e-4, device='cuda', focal_gamma=2.0):
    """
    Two-stage training: freeze backbone then fine-tune
    Now uses Focal Loss to address class imbalance
    
    Stage 1: Train only classifier with frozen backbone (10 epochs)
    Stage 2: Unfreeze and fine-tune entire model (10 epochs)
    
    Args:
        focal_gamma: Gamma parameter for Focal Loss (2.0 is standard, higher = more focus on hard examples)
    """
    from modules import get_focal_loss
    
    # Use Focal Loss instead of CrossEntropyLoss
    # Gamma = 2.0 is standard, can increase to 3.0 or 4.0 for more focus on hard examples
    criterion = get_focal_loss(alpha=2.0, gamma=2.0)
    
    print(f"\nUsing Focal Loss with gamma={focal_gamma}")
    print("This will help fix the class imbalance (NC: 93% recall, AD: 55% recall)")
    
    # STAGE 1: Frozen backbone
    print("\n" + "="*80)
    print("STAGE 1: Training classifier with frozen backbone (10 epochs)")
    print("="*80)
    
    # Freeze backbone
    for param in model.model.features.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=lr_frozen, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_frozen)
    
    history_stage1, _ = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs_frozen, device=device, patience=5,
        save_path='stage1_model.pth'
    )
    
    # STAGE 2: Fine-tune entire model
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning entire model (10 epochs)")
    print("="*80)
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = optim.AdamW(model.parameters(), lr=lr_finetune, weight_decay=0.1)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs_finetune)
    
    history_stage2, model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        num_epochs=num_epochs_finetune, device=device, patience=8,
        save_path='best_convnext_model.pth'
    )
    
    # Combine histories
    history = {
        'train_loss': history_stage1['train_loss'] + history_stage2['train_loss'],
        'train_acc': history_stage1['train_acc'] + history_stage2['train_acc'],
        'val_loss': history_stage1['val_loss'] + history_stage2['val_loss'],
        'val_acc': history_stage1['val_acc'] + history_stage2['val_acc'],
        'lr': history_stage1['lr'] + history_stage2['lr']
    }
    
    return history, model


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot learning rate
    axes[2].plot(history['lr'], marker='o', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    """
    Main execution block for training
    """
    from dataset import get_data_loaders
    from modules import get_model
    
    # Configuration
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Focal Loss gamma parameter
    # 2.0 = standard (recommended starting point)
    # 3.0 = more focus on hard examples
    # 4.0 = even more focus on hard examples
    FOCAL_GAMMA = 2.0
    
    print("="*80)
    print("ADNI Alzheimer's Classification Training with Focal Loss")
    print("="*80)
    
    # Load data with proper train/val split
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_PATH, 
        batch_size=BATCH_SIZE,
        use_val_split=True,
        val_split=0.2,
        random_seed=42
    )
    
    # Create model
    print("\nInitializing model...")
    model = get_model(device=DEVICE, pretrained=True, freeze_backbone=False)
    
    # Use 2-stage training with Focal Loss
    print("\nStarting training with Focal Loss...")
    print(f"Stage 1: 10 epochs (frozen backbone)")
    print(f"Stage 2: 10 epochs (fine-tuning)")
    print(f"Focal Loss gamma: {FOCAL_GAMMA}\n")
    
    history, best_model = train_two_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs_frozen=10,
        num_epochs_finetune=10,
        lr_frozen=1e-3,
        lr_finetune=1e-4,
        device=DEVICE,
        focal_gamma=FOCAL_GAMMA
    )
    
    # Plot results
    print("\nGenerating training plots...")
    plot_training_history(history)
    
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    print(f"\nBest model saved as: best_convnext_model.pth")
    print(f"Now run predict.py to evaluate on test set!")

