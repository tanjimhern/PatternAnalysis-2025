# Alzheimer's Disease Classification with ConvNeXt

Binary classification of brain MRI images to distinguish between Alzheimer's Disease (AD) and Cognitively Normal (NC) patients using ConvNeXt-Small deep learning architecture with transfer learning.

## Problem Description

Alzheimer's Disease is a progressive neurodegenerative disorder and the most common cause of dementia, affecting millions worldwide. Early and accurate diagnosis from brain MRI scans is crucial for treatment planning and patient care.

This project implements an automated classification system using the ADNI (Alzheimer's Disease Neuroimaging Initiative) dataset containing 21,520 training images and 9,000 test images of brain MRI scans. The goal is to classify each scan as either AD (Alzheimer's Disease) or NC (Normal Control).

## Model Architecture

### ConvNeXt-Small

ConvNeXt is a modern CNN architecture that incorporates design principles from Vision Transformers while maintaining the efficiency of traditional CNNs. We use **ConvNeXt-Small** (~50M parameters) pretrained on ImageNet-1K.

**Architecture:**
- Backbone: ConvNeXt-Small feature extractor (pretrained on ImageNet)
- Classifier Head: LayerNorm → Flatten → Dropout (p=0.7) → Linear (768 → 2)

### How It Works

The model uses a **two-stage training strategy**:

**Stage 1: Frozen Backbone Training (15 epochs)**
- Freeze the pretrained ConvNeXt backbone
- Train only the classifier head with LR=1e-3
- Prevents catastrophic forgetting of ImageNet features
- Fast convergence to baseline performance

**Stage 2: Full Fine-Tuning (50 epochs)**  
- Unfreeze all layers
- Fine-tune entire model with LR=1e-4
- Adapts all layers to medical imaging domain
- Early stopping with patience=15

**Key Techniques:**
- **MixUp Augmentation (α=0.4):** Blends pairs of training images and their labels to create virtual training examples, forcing the model to learn more robust features
- **Label Smoothing (ε=0.15):** Softens hard labels from [0,1] to [0.05, 0.95], preventing overconfidence
- **Heavy Data Augmentation:** Geometric and intensity transforms to improve generalization
- **High Dropout (p=0.7):** Strong regularization in classifier head

![Training History](training_history.png)

## Data Preprocessing

### Image Preprocessing
1. Load grayscale brain MRI images (JPEG format)
2. Convert to RGB (3 channels) by channel replication for pretrained model compatibility
3. Resize to 224×224 pixels
4. Normalize using ADNI-specific statistics:
   - Mean: [0.115, 0.115, 0.115]
   - Std: [0.225, 0.225, 0.225]

### Training Augmentation
Strong spatial augmentation to improve robustness across different MRI scanners:
- Random rotation (±30°)
- Random affine transforms (translation, scale 0.75-1.25x, shear)
- Random perspective (distortion=0.2)
- Random resized crop (70-100% of image)
- Random horizontal/vertical flips
- Color jitter (brightness, contrast ±30%)
- Gaussian blur
- Random erasing (p=0.3)
- **MixUp:** Creates virtual training examples by blending image pairs

**Validation/Test Augmentation:**
- Resize to 224×224
- Normalize (no random transforms)

## Train/Validation/Test Split

### Data Distribution
```
Training Data (from train folder):
├── Total: 21,520 images
├── AD: 10,400 (48.3%)
└── NC: 11,120 (51.7%)

Split (80/20 stratified):
├── Training: 17,216 images (48.3% AD, 51.7% NC)
└── Validation: 4,304 images (48.3% AD, 51.7% NC)

Test Data (from test folder - held out):
├── Total: 9,000 images  
├── AD: 4,460 (49.6%)
└── NC: 4,540 (50.4%)
```

### Split Justification - after consult with shakes 

**80/20 Split:**
- Provides sufficient validation data (4,304 samples) for reliable performance estimation
- Maximizes training data (17,216 samples) for learning
- Standard practice in deep learning

**Stratified Sampling:**
- Maintains class balance in both training and validation sets
- Ensures representative evaluation across both classes
- Critical for medical datasets with binary outcomes

**Separate Test Set:**
- Test data completely held out during training
- Never used for model selection or hyperparameter tuning
- Provides unbiased estimate of real-world performance
- Prevents data leakage and overfitting to test distribution

## Results

### Training Performance
- Stage 1 (Frozen): Best validation accuracy **71.91%** at epoch 10
- Stage 2 (Fine-tuning): Best validation accuracy **99.47%** at epoch 43
- Total training time: 117 minutes on NVIDIA A100 GPU

### Test Performance

**Overall Metrics:**
- **Test Accuracy: 77.51%**
- **ROC AUC: 0.849**

**Per-Class Performance:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| NC (Normal) | 0.6972 | 0.9797 | 0.8147 | 4,540 |
| AD (Alzheimer) | 0.9649 | 0.5668 | 0.7141 | 4,460 |
| **Accuracy** | | | **0.7751** | **9,000** |

**Confusion Matrix:**

![Confusion Matrix](confusion_matrix.png)

**Key Observations:**
- High NC recall (97.97%) but lower AD recall (56.68%)
- Very high AD precision (96.49%) - when model predicts AD, it's usually correct
- Model exhibits conservative behavior, preferring NC predictions
- 1,932 false negatives (missed AD cases) vs. 92 false positives

### ROC Curve

![ROC Curve](roc_curve.png)

The ROC curve shows the model's ability to discriminate between AD and NC cases 
across different decision thresholds. The AUC of **0.849** indicates strong 
performance, significantly better than random classification (0.5). This suggests 
the model has good separability between classes - it can rank predictions well 
even though the fixed threshold yields 77.51% accuracy.

### Sample Predictions
![Sample Predictions](sample_predictions.png)

The visualization shows 16 random test samples with their predictions. Green text 
indicates correct predictions, red text indicates errors. This particular batch 
shows 7 incorrect predictions (43.75% error rate), and notably, **all 7 errors 
are false negatives** (AD cases predicted as NC) with zero false positives. This 
extreme pattern strongly demonstrates the model's conservative bias toward NC 
predictions, consistent with the confusion matrix showing 1,932 false negatives 
versus only 92 false positives on the full test set.