# predict.py
# Prediction and visualization script for trained ConvNeXt model
# Generates detailed evaluation metrics and visualizations for README

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
import numpy as np
from PIL import Image
import os

from dataset import get_data_loaders
from modules import get_model


def load_trained_model(model_path, device='cuda'):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to saved model checkpoint (.pth file)
        device: Device to load model on
    
    Returns:
        model: Loaded model ready for inference
        checkpoint: Checkpoint dictionary with training info
    """
    print(f"\nLoading model from {model_path}...")
    
    # Create model architecture
    model = get_model(
        model_size='small',
        num_classes=2,
        dropout=0.7,
        pretrained=False,  # Don't need pretrained weights, we're loading trained model
        device=device
    )
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✓ Model loaded successfully")
    print(f"  Trained for epoch: {checkpoint['epoch'] + 1}")
    print(f"  Best validation accuracy: {checkpoint['val_acc']:.4f}")
    
    return model, checkpoint


def evaluate_model(model, test_loader, device='cuda'):
    """
    Evaluate model on test set and collect predictions.
    
    Args:
        model: Trained ConvNeXt model
        test_loader: Test data loader
        device: Device
    
    Returns:
        all_preds: Predicted labels (numpy array)
        all_labels: True labels (numpy array)
        all_probs: Prediction probabilities (numpy array, shape: N x 2)
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # Collect results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate accuracy
    accuracy = (all_preds == all_labels).sum() / len(all_labels)
    print(f"✓ Evaluation complete")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return all_preds, all_labels, all_probs


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    
    Returns:
        cm: Confusion matrix array
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=['NC (Normal)', 'AD (Alzheimer)'],
                yticklabels=['NC (Normal)', 'AD (Alzheimer)'],
                cbar_kws={'label': 'Count'},
                ax=ax)
    
    # Add text annotations with both counts and percentages
    for i in range(2):
        for j in range(2):
            text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   color='white' if cm[i, j] > cm.max() / 2 else 'black',
                   fontsize=14, fontweight='bold')
    
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved to {save_path}")
    plt.close()
    
    return cm


def plot_roc_curve(y_true, y_probs, save_path='roc_curve.png'):
    """
    Plot ROC curve for binary classification.
    
    The ROC curve shows the trade-off between true positive rate (sensitivity)
    and false positive rate (1 - specificity) at various classification thresholds.
    
    AUC (Area Under Curve): 
    - 1.0 = Perfect classifier
    - 0.5 = Random classifier
    
    Args:
        y_true: True labels
        y_probs: Prediction probabilities (N x 2 array)
        save_path: Path to save the plot
    
    Returns:
        roc_auc: AUC score
    """
    # Get probabilities for positive class (AD = class 1)
    y_probs_ad = y_probs[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_probs_ad)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    
    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier (AUC = 0.500)')
    
    # Styling
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved to {save_path}")
    plt.close()
    
    return roc_auc


def plot_per_class_metrics(y_true, y_pred, y_probs, save_path='per_class_metrics.png'):
    """
    Plot detailed per-class performance metrics.
    
    Shows precision, recall, F1-score, and sample counts for each class.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate metrics for each class
    classes = ['NC (Normal)', 'AD (Alzheimer)']
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Sample counts
    support = [np.sum(y_true == 0), np.sum(y_true == 1)]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision, Recall, F1-Score
    x = np.arange(len(classes))
    width = 0.25
    
    axes[0, 0].bar(x - width, precision, width, label='Precision', color='#3498db')
    axes[0, 0].bar(x, recall, width, label='Recall', color='#e74c3c')
    axes[0, 0].bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
    
    axes[0, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Per-Class Performance Metrics', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(classes)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        axes[0, 0].text(i - width, p + 0.02, f'{p:.3f}', ha='center', fontsize=9)
        axes[0, 0].text(i, r + 0.02, f'{r:.3f}', ha='center', fontsize=9)
        axes[0, 0].text(i + width, f + 0.02, f'{f:.3f}', ha='center', fontsize=9)
    
    # Plot 2: Sample Distribution
    colors = ['#3498db', '#e74c3c']
    axes[0, 1].bar(classes, support, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (cls, cnt) in enumerate(zip(classes, support)):
        axes[0, 1].text(i, cnt + 50, f'{cnt}\n({cnt/sum(support)*100:.1f}%)', 
                       ha='center', fontsize=10, fontweight='bold')
    
    # Plot 3: Confidence Distribution for Correct Predictions
    correct_mask = y_true == y_pred
    correct_probs = np.max(y_probs[correct_mask], axis=1)
    
    axes[1, 0].hist(correct_probs, bins=30, color='#2ecc71', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title(f'Confidence Distribution - Correct Predictions (n={correct_mask.sum()})', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].axvline(np.mean(correct_probs), color='red', linestyle='--', linewidth=2,
                      label=f'Mean: {np.mean(correct_probs):.3f}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Confidence Distribution for Incorrect Predictions
    incorrect_mask = y_true != y_pred
    if incorrect_mask.sum() > 0:
        incorrect_probs = np.max(y_probs[incorrect_mask], axis=1)
        
        axes[1, 1].hist(incorrect_probs, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
        axes[1, 1].set_title(f'Confidence Distribution - Incorrect Predictions (n={incorrect_mask.sum()})', 
                           fontsize=12, fontweight='bold')
        axes[1, 1].axvline(np.mean(incorrect_probs), color='red', linestyle='--', linewidth=2,
                          label=f'Mean: {np.mean(incorrect_probs):.3f}')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 1].text(0.5, 0.5, 'No Incorrect Predictions!', 
                       ha='center', va='center', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics saved to {save_path}")
    plt.close()


def visualize_sample_predictions(model, test_loader, device='cuda', num_samples=16, save_path='sample_predictions.png'):
    """
    Visualize sample predictions with images.
    
    Shows actual brain MRI images with their true labels, predicted labels,
    and prediction confidence. Correct predictions are shown in green,
    incorrect predictions in red.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device
        num_samples: Number of samples to visualize (default: 16)
        save_path: Path to save the plot
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
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.ravel()
    
    class_names = ['NC (Normal)', 'AD (Alzheimer)']
    
    # ADNI normalization values (for denormalization)
    mean = torch.tensor([0.115, 0.115, 0.115]).view(3, 1, 1)
    std = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1)
    
    for i in range(min(num_samples, len(images))):
        img = images[i]
        
        # Denormalize image for display
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        
        # Convert to displayable format (use only first channel since it's grayscale)
        img = img[0].numpy()  # Take first channel
        
        # Plot
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
        
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        confidence = probs[i][preds[i]] * 100
        
        # Green if correct, red if wrong
        color = 'green' if labels[i] == preds[i] else 'red'
        
        # Create title with true label, prediction, and confidence
        title = f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%'
        axes[i].set_title(title, color=color, fontsize=10, fontweight='bold', pad=10)
        
        # Add border
        for spine in axes[i].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Sample predictions saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    """
    Main execution block.
    
    This script:
    1. Loads the trained model
    2. Evaluates on test set
    3. Generates all visualizations
    4. Creates detailed report
    """
    
    # Configuration
    MODEL_PATH = 'best_model.pth'
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    BATCH_SIZE = 32
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("ADNI ALZHEIMER'S CLASSIFICATION - PREDICTION & VISUALIZATION")
    print("="*80)
    print(f"Device: {DEVICE}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model file '{MODEL_PATH}' not found!")
        print("Please run train.py first to train the model.")
        exit(1)
    
    # Load test data
    print("\nLoading test data...")
    _, _, test_loader = get_data_loaders(
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
        num_workers=2
    )
    
    # Load trained model
    model, checkpoint = load_trained_model(MODEL_PATH, device=DEVICE)
    
    # Evaluate on test set
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    y_pred, y_true, y_probs = evaluate_model(model, test_loader, device=DEVICE)
    
    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Confusion Matrix
    print("\n1. Creating confusion matrix...")
    cm = plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png')
    
    # 2. ROC Curve
    print("\n2. Creating ROC curve...")
    roc_auc = plot_roc_curve(y_true, y_probs, save_path='roc_curve.png')
    
    # 3. Per-class metrics
    print("\n3. Creating per-class metrics visualization...")
    plot_per_class_metrics(y_true, y_pred, y_probs, save_path='per_class_metrics.png')
    
    # 4. Sample predictions
    print("\n4. Creating sample predictions visualization...")
    visualize_sample_predictions(model, test_loader, device=DEVICE, 
                                 num_samples=16, save_path='sample_predictions.png')
    
    # Final summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  - confusion_matrix.png")
    print("  - roc_curve.png")
    print("  - per_class_metrics.png")
    print("  - sample_predictions.png")
    print("="*80)
