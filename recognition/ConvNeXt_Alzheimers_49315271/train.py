# train.py
# Training script for Alzheimer's classification with ConvNeXt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import copy

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, patience=10, save_path='best_model.pth'):
    """
    Main training loop with early stopping
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
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

import matplotlib.pyplot as plt

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
    DATA_PATH = '/content/drive/MyDrive/AD_NC'
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("ADNI Alzheimer's Classification Training")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    train_loader, test_loader = get_data_loaders(DATA_PATH, batch_size=BATCH_SIZE)
    
    # Create model
    print("\nInitializing model...")
    model = get_model(device=DEVICE, pretrained=True, freeze_backbone=False)
    
    # Loss function with class weights (if needed for imbalance)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Train model
    print("\nStarting training...\n")
    history, best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,  # Using test as validation for now
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=NUM_EPOCHS,
        device=DEVICE,
        patience=PATIENCE,
        save_path='best_convnext_model.pth'
    )
    
    # Plot results
    print("\nGenerating training plots...")
    plot_training_history(history)
    
    print("\nTraining complete!")