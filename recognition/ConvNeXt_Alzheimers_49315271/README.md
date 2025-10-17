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