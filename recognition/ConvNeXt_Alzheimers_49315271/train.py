# train.py
# Training script for ADNI Alzheimer's classification
# Implements two-stage training with MixUp augmentation and proper evaluation

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from sklearn.metrics import classification_report, confusion_matrix
import os

from dataset import get_data_loaders
from modules import get_model, get_loss_function, MixUpAugmentation, count_parameters


def train_one_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True, mixup_alpha=0.4):
    """
    Train the model for one epoch.
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        criterion: Loss function (Label Smoothing)
        optimizer: Optimizer
        device: Device (cuda/cpu)
        use_mixup: Whether to use MixUp augmentation (default: True)
        mixup_alpha: MixUp alpha parameter (default: 0.4)
    
    Returns:
        avg_loss: Average training loss for the epoch
        accuracy: Training accuracy for the epoch
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize MixUp if enabled
    mixup = MixUpAugmentation(alpha=mixup_alpha) if use_mixup else None
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Apply MixUp augmentation
        if use_mixup and mixup is not None:
            images, labels_a, labels_b, lam = mixup.mixup_data(images, labels)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate MixUp loss
            loss = mixup.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            
            # For accuracy calculation, use the dominant label
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            # Approximate accuracy with MixUp (use labels_a as primary)
            correct += (lam * (predicted == labels_a).sum().item() + 
                       (1 - lam) * (predicted == labels_b).sum().item())
        else:
            # Standard forward pass without MixUp
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    
    Args:
        model: ConvNeXt model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device
    
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
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item() * images.size(0)
    
    avg_loss = running_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, stage_name, use_mixup=True, patience=10, 
                save_path='model_checkpoint.pth'):
    """
    Train the model for one stage (either frozen or full fine-tuning).
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device
        stage_name: Name of the training stage (for logging)
        use_mixup: Whether to use MixUp augmentation
        patience: Early stopping patience
        save_path: Path to save best model checkpoint
    
    Returns:
        history: Dictionary containing training metrics
        best_model: Model with best validation accuracy
    """
    print("\n" + "="*80)
    print(f"{stage_name}")
    print("="*80)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    since = time.time()
    
    # Print header
    print(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<12} {'Status':<15}")
    print("-" * 100)
    
    for epoch in range(num_epochs):
        # Training phase
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_mixup=use_mixup
        )
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Determine status
        status = ""
        
        # Check if this is the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            status = "✓ Best"
            
            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
            }, save_path)
        else:
            epochs_no_improve += 1
        
        # Print epoch results
        print(f"{epoch+1:<8} {train_loss:<12.4f} {train_acc:<12.4f} {val_loss:<12.4f} {val_acc:<12.4f} {current_lr:<12.6f} {status:<15}")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {patience} epochs without improvement")
            print(f"Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}")
            break
    
    time_elapsed = time.time() - since
    print(f'\n{stage_name} completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f} at epoch {best_epoch}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return history, model


def train_two_stage(model, train_loader, val_loader, device='cuda',
                    num_epochs_stage1=15, num_epochs_stage2=50,
                    lr_stage1=1e-3, lr_stage2=1e-4,
                    use_mixup=True, smoothing=0.15):
    """
    Two-stage training strategy for transfer learning.
    
    Stage 1: Train only the classifier head with frozen backbone
        - Faster convergence
        - Prevents catastrophic forgetting of pretrained features
        - Higher learning rate since we're only training the head
    
    Stage 2: Fine-tune the entire model
        - Adapts all layers to the target domain
        - Lower learning rate to avoid destroying pretrained features
        - Uses MixUp for better generalization
    
    Args:
        model: ConvNeXt model
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device (cuda/cpu)
        num_epochs_stage1: Epochs for stage 1 (default: 15)
        num_epochs_stage2: Max epochs for stage 2 (default: 50)
        lr_stage1: Learning rate for stage 1 (default: 1e-3)
        lr_stage2: Learning rate for stage 2 (default: 1e-4)
        use_mixup: Use MixUp augmentation (default: True)
        smoothing: Label smoothing parameter (default: 0.15)
    
    Returns:
        history: Combined training history from both stages
        model: Best trained model
    """
    
    # Get loss function
    criterion = get_loss_function(loss_type='label_smoothing', smoothing=smoothing)
    
    # ========== STAGE 1: Frozen Backbone ==========
    print("\n" + "="*80)
    print("STAGE 1: Training classifier head with frozen backbone")
    print("="*80)
    print(f"Epochs: {num_epochs_stage1}, Learning rate: {lr_stage1}, MixUp: {use_mixup}")
    
    # Freeze backbone (features)
    # Access model.features directly since get_model() returns ConvNeXt object
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Print trainable parameters
    total, trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    # Optimizer for stage 1 (only classifier parameters)
    optimizer_stage1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr_stage1,
        weight_decay=0.1
    )
    
    # Scheduler for stage 1
    scheduler_stage1 = CosineAnnealingLR(optimizer_stage1, T_max=num_epochs_stage1, eta_min=1e-6)
    
    # Train stage 1
    history_stage1, model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage1,
        scheduler=scheduler_stage1,
        num_epochs=num_epochs_stage1,
        device=device,
        stage_name="STAGE 1: FROZEN BACKBONE TRAINING",
        use_mixup=use_mixup,
        patience=8,
        save_path='stage1_checkpoint.pth'
    )
    
    # ========== STAGE 2: Full Fine-tuning ==========
    print("\n" + "="*80)
    print("STAGE 2: Fine-tuning entire model")
    print("="*80)
    print(f"Epochs: {num_epochs_stage2}, Learning rate: {lr_stage2}, MixUp: {use_mixup}")
    
    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True
    
    # Print trainable parameters
    total, trainable = count_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    
    # Optimizer for stage 2 (all parameters)
    optimizer_stage2 = optim.AdamW(
        model.parameters(),
        lr=lr_stage2,
        weight_decay=0.1
    )
    
    # Scheduler for stage 2
    scheduler_stage2 = CosineAnnealingLR(optimizer_stage2, T_max=num_epochs_stage2, eta_min=1e-7)
    
    # Train stage 2
    history_stage2, model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_stage2,
        scheduler=scheduler_stage2,
        num_epochs=num_epochs_stage2,
        device=device,
        stage_name="STAGE 2: FULL MODEL FINE-TUNING",
        use_mixup=use_mixup,
        patience=15,
        save_path='best_model.pth'
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
    Plot training and validation metrics over epochs.
    
    Creates a figure with 3 subplots:
    1. Loss curves (train and validation)
    2. Accuracy curves (train and validation)
    3. Learning rate schedule
    
    Args:
        history: Dictionary containing training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Plot 3: Learning Rate
    axes[2].plot(epochs, history['lr'], 'g-', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training history plot saved to {save_path}")
    plt.close()


def test_model(model, test_loader, device='cuda'):
    """
    Evaluate the model on the test set.
    
    This performs final evaluation on the completely held-out test set.
    Prints detailed metrics including per-class accuracy, confusion matrix,
    and classification report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
    
    Returns:
        test_acc: Test accuracy
        all_preds: Predicted labels
        all_labels: True labels
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n" + "="*80)
    print("FINAL TEST SET EVALUATION")
    print("="*80)
    print("Evaluating on held-out test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate overall accuracy
    test_acc = (all_preds == all_labels).sum() / len(all_labels)
    
    print(f"\n{'='*80}")
    print(f"TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*80}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print("-" * 80)
    print(classification_report(
        all_labels, 
        all_preds,
        target_names=['NC (Normal)', 'AD (Alzheimer)'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("-" * 80)
    print(f"                    Predicted NC    Predicted AD")
    print(f"Actual NC           {cm[0,0]:<15} {cm[0,1]:<15}")
    print(f"Actual AD           {cm[1,0]:<15} {cm[1,1]:<15}")
    print()
    print(f"True Negatives (NC correctly classified):  {cm[0,0]}")
    print(f"False Positives (NC predicted as AD):      {cm[0,1]}")
    print(f"False Negatives (AD predicted as NC):      {cm[1,0]}")
    print(f"True Positives (AD correctly classified):  {cm[1,1]}")
    print("="*80)
    
    return test_acc, all_preds, all_labels


if __name__ == "__main__":
    """
    Main training script.
    
    This script performs the complete training workflow:
    1. Load data (train/val/test splits)
    2. Initialize model
    3. Train with two-stage approach
    4. Evaluate on validation set
    5. Test on held-out test set
    6. Save results and plots
    """
    
    # ========== CONFIGURATION ==========
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    BATCH_SIZE = 32
    VAL_SPLIT = 0.2
    NUM_WORKERS = 4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Training hyperparameters
    NUM_EPOCHS_STAGE1 = 15
    NUM_EPOCHS_STAGE2 = 50
    LR_STAGE1 = 1e-3
    LR_STAGE2 = 1e-4
    USE_MIXUP = True
    LABEL_SMOOTHING = 0.15
    DROPOUT = 0.7
    
    print("="*80)
    print("ADNI ALZHEIMER'S DISEASE CLASSIFICATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation split: {VAL_SPLIT}")
    print(f"MixUp augmentation: {USE_MIXUP}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    print(f"Dropout: {DROPOUT}")
    print("="*80)
    
    # ========== LOAD DATA ==========
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        num_workers=NUM_WORKERS
    )
    
    # ========== INITIALIZE MODEL ==========
    print("\nInitializing model...")
    model = get_model(
        model_size='small',
        num_classes=2,
        dropout=DROPOUT,
        pretrained=True,
        device=DEVICE
    )
    
    # ========== TRAIN MODEL ==========
    print("\nStarting training...\n")
    history, trained_model = train_two_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs_stage1=NUM_EPOCHS_STAGE1,
        num_epochs_stage2=NUM_EPOCHS_STAGE2,
        lr_stage1=LR_STAGE1,
        lr_stage2=LR_STAGE2,
        use_mixup=USE_MIXUP,
        smoothing=LABEL_SMOOTHING
    )
    
    # ========== PLOT TRAINING HISTORY ==========
    print("\nGenerating training plots...")
    plot_training_history(history, save_path='training_history.png')
    
    # ========== FINAL TEST EVALUATION ==========
    print("\nLoading best model for final testing...")
    checkpoint = torch.load('best_model.pth')
    trained_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_acc, test_preds, test_labels = test_model(trained_model, test_loader, device=DEVICE)
    
    # ========== SAVE FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("TRAINING COMPLETE - FINAL SUMMARY")
    print("="*80)
    print(f"Best validation accuracy: {checkpoint['val_acc']:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    print(f"\nSaved files:")
    print(f"  - best_model.pth (best model checkpoint)")
    print(f"  - stage1_checkpoint.pth (stage 1 checkpoint)")
    print(f"  - training_history.png (training curves)")
    print(f"\nTo visualize predictions and generate detailed reports:")
    print(f"  python predict.py")
    print("="*80)
