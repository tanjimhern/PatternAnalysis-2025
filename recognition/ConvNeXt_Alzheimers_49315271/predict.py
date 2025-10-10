# predict.py
# Prediction and evaluation script for trained ConvNeXt model

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
from PIL import Image
import os


def load_trained_model(model_path, device='cuda'):
    """
    Load trained model from checkpoint
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
    
    Returns:
        model: Loaded model ready for inference
    """
    from modules import get_model
    
    # Create model architecture
    model = get_model(device=device, pretrained=False)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Checkpoint from epoch {checkpoint['epoch']}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set
    
    Returns:
        all_preds: All predictions
        all_labels: All true labels
        all_probs: All prediction probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Evaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NC (Normal)', 'AD (Alzheimer)'],
                yticklabels=['NC (Normal)', 'AD (Alzheimer)'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.show()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """
    Plot ROC curve
    """
    # Get probabilities for positive class (AD)
    y_probs_ad = y_probs[:, 1]
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs_ad)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.show()
    
    return roc_auc


def visualize_predictions(model, test_loader, device='cuda', num_samples=16):
    """
    Visualize sample predictions
    """
    model.eval()
    
    # Get one batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
    
    # Move to CPU for plotting
    images = images.cpu()
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    class_names = ['NC (Normal)', 'AD (Alzheimer)']
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        
        # Denormalize image for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to displayable format
        img = img.permute(1, 2, 0).numpy()
        
        # Plot
        axes[i].imshow(img)
        axes[i].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i][preds[i]] * 100
        
        color = 'green' if labels[i] == preds[i] else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                         color=color, fontsize=9)
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    print("Sample predictions saved to sample_predictions.png")
    plt.show()


if __name__ == "__main__":
    """
    Main execution block
    """
    from dataset import get_data_loaders
    
    # Configuration
    MODEL_PATH = 'best_convnext_model.pth'
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("ADNI Alzheimer's Classification - Prediction & Evaluation")
    print("="*80)
    
    # Load data
    print("\nLoading test data...")
    _,_, test_loader = get_data_loaders(DATA_PATH, batch_size=BATCH_SIZE,use_val_split=True,val_split=0.2)
    
    # Load trained model
    print("\nLoading trained model...")
    model = load_trained_model(MODEL_PATH, device=DEVICE)
    
    # Evaluate on test set
    print("\n" + "="*80)
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device=DEVICE)
    
    # Classification report
    print("\n" + "="*80)
    print("Classification Report:")
    print("="*80)
    print(classification_report(y_true, y_pred, 
                                target_names=['NC (Normal)', 'AD (Alzheimer)']))
    
    # Confusion matrix
    print("\nGenerating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred)
    
    # ROC curve
    print("\nGenerating ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_probs)
    
    # Sample predictions
    print("\nVisualizing sample predictions...")
    visualize_predictions(model, test_loader, device=DEVICE)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)
