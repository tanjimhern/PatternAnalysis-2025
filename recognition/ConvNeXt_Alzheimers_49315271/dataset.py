# dataset.py
# Data loader for ADNI Alzheimer's classification
# Proper train/val split with stratification, no test set leakage

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split


class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI images.
    
    Loads grayscale brain MRI images (stored as JPEG) from AD and NC folders.
    Images are converted to RGB format for compatibility with pretrained models.
    
    Directory structure expected:
        data_path/
            train/
                AD/
                    image1.jpg
                    image2.jpg
                NC/
                    image1.jpg
                    image2.jpg
            test/
                AD/
                NC/
    """
    
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to folder containing AD/ and NC/ subfolders
            transform: torchvision transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load AD images (label = 1)
        ad_path = os.path.join(root_dir, 'AD')
        if os.path.exists(ad_path):
            for img_name in os.listdir(ad_path):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(ad_path, img_name))
                    self.labels.append(1)
        
        # Load NC (Normal Control) images (label = 0)
        nc_path = os.path.join(root_dir, 'NC')
        if os.path.exists(nc_path):
            for img_name in os.listdir(nc_path):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.images.append(os.path.join(nc_path, img_name))
                    self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"  AD samples: {sum(self.labels)}")
        print(f"  NC samples: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor of shape (3, 224, 224)
            label: 0 for NC (Normal), 1 for AD (Alzheimer's)
        """
        img_path = self.images[idx]
        
        # Load as grayscale first to check
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Convert to RGB by repeating channels (required for pretrained models)
        image = Image.merge('RGB', (image, image, image))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_train_transform():
    """
    Training augmentation with heavy spatial transforms.
    
    Since intensity distributions between train/test are identical,
    we focus on geometric augmentation to improve spatial robustness.
    
    Key augmentations:
    - Rotation (±30°): Brain orientation can vary
    - Affine transforms: Different positioning and scales
    - Random crops: Force model to work with partial views
    - Random erasing: Prevent reliance on specific regions
    - Moderate intensity jitter: Scanner variations
    
    Returns:
        Composed transform pipeline for training
    """
    
    # ADNI-specific normalization
    # These values are computed from the actual training data
    ADNI_MEAN = [0.115, 0.115, 0.115]
    ADNI_STD = [0.225, 0.225, 0.225]
    
    return transforms.Compose([
        # Start with larger size for augmentation
        transforms.Resize((256, 256)),
        
        # === HEAVY GEOMETRIC AUGMENTATION ===
        # Rotation: Brain can be oriented differently
        transforms.RandomRotation(degrees=30),
        
        # Affine: Simulate different positioning and scales
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),    # 15% shifts
            scale=(0.75, 1.25),        # 25% zoom variation
            shear=15                    # Perspective differences
        ),
        
        # Perspective: Different viewing angles
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        
        # Random resized crop: See different brain regions
        # Scale 0.7-1.0 means we crop 70%-100% of image
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        
        # Flips: Anatomical variations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        
        # === MODERATE INTENSITY AUGMENTATION ===
        # Brightness and contrast variations (scanner differences)
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        
        # Gaussian blur: Different image quality
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Convert to tensor (scales to [0, 1])
        transforms.ToTensor(),
        
        # Random erasing: Force model to not rely on specific landmarks
        # This prevents the model from memorizing specific anatomical markers
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        
        # Normalize using ADNI statistics
        transforms.Normalize(mean=ADNI_MEAN, std=ADNI_STD)
    ])


def get_test_transform():
    """
    Test/validation transform without augmentation.
    
    Only resize and normalize - no random transforms.
    This ensures consistent evaluation.
    
    Returns:
        Composed transform pipeline for testing/validation
    """
    ADNI_MEAN = [0.115, 0.115, 0.115]
    ADNI_STD = [0.225, 0.225, 0.225]
    
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ADNI_MEAN, std=ADNI_STD)
    ])


def get_data_loaders(data_path, batch_size=32, val_split=0.2, num_workers=4, random_seed=42):
    """
    Create train, validation, and test data loaders.
    
    IMPORTANT: This implements proper data splitting:
    1. Load training data from train folder
    2. Split into 80% train, 20% validation (stratified)
    3. Load test data separately (completely held out)
    
    The test set is NEVER used during training or validation.
    
    Args:
        data_path: Path to AD_NC folder containing train/ and test/ subfolders
        batch_size: Batch size for data loaders
        val_split: Fraction of training data to use for validation (default: 0.2)
        num_workers: Number of worker processes for data loading
        random_seed: Random seed for reproducible splits
    
    Returns:
        train_loader: Training data (80% of train folder)
        val_loader: Validation data (20% of train folder)
        test_loader: Test data (entire test folder, held out)
    """
    
    print("\n" + "="*80)
    print("DATA LOADING CONFIGURATION")
    print("="*80)
    
    # === STEP 1: Load full training dataset ===
    train_path = os.path.join(data_path, 'train')
    full_train_dataset = ADNIDataset(root_dir=train_path, transform=None)
    
    # === STEP 2: Stratified train/val split ===
    # Stratification ensures class balance is maintained in both splits
    train_indices, val_indices = train_test_split(
        range(len(full_train_dataset)),
        test_size=val_split,
        stratify=full_train_dataset.labels,
        random_state=random_seed
    )
    
    print(f"\nTrain/Val Split (stratified):")
    print(f"  Training samples: {len(train_indices)}")
    print(f"  Validation samples: {len(val_indices)}")
    
    # Check class balance in splits
    train_labels = [full_train_dataset.labels[i] for i in train_indices]
    val_labels = [full_train_dataset.labels[i] for i in val_indices]
    
    print(f"\nClass distribution in training split:")
    print(f"  NC: {train_labels.count(0)} ({train_labels.count(0)/len(train_labels)*100:.1f}%)")
    print(f"  AD: {train_labels.count(1)} ({train_labels.count(1)/len(train_labels)*100:.1f}%)")
    
    print(f"\nClass distribution in validation split:")
    print(f"  NC: {val_labels.count(0)} ({val_labels.count(0)/len(val_labels)*100:.1f}%)")
    print(f"  AD: {val_labels.count(1)} ({val_labels.count(1)/len(val_labels)*100:.1f}%)")
    
    # === STEP 3: Create datasets with appropriate transforms ===
    
    # Training subset with heavy augmentation
    train_dataset = Subset(full_train_dataset, train_indices)
    # Need to apply transform after subsetting
    class TransformedSubset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        
        def __len__(self):
            return len(self.subset)
        
        def __getitem__(self, idx):
            image, label = self.subset[idx]
            if self.transform:
                image = self.transform(image)
            return image, label
    
    # Apply transforms
    train_dataset_transformed = TransformedSubset(train_dataset, get_train_transform())
    val_dataset_transformed = TransformedSubset(
        Subset(full_train_dataset, val_indices),
        get_test_transform()
    )
    
    # === STEP 4: Load test dataset (completely separate) ===
    test_path = os.path.join(data_path, 'test')
    test_dataset = ADNIDataset(root_dir=test_path, transform=get_test_transform())
    
    print(f"\nTest set (held out):")
    test_labels = test_dataset.labels
    print(f"  NC: {test_labels.count(0)} ({test_labels.count(0)/len(test_labels)*100:.1f}%)")
    print(f"  AD: {test_labels.count(1)} ({test_labels.count(1)/len(test_labels)*100:.1f}%)")
    
    # === STEP 5: Create data loaders ===
    train_loader = DataLoader(
        train_dataset_transformed,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset_transformed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nData loaders created successfully!")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of workers: {num_workers}")
    print("="*80 + "\n")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    """
    Test the data loader
    """
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    
    print("Testing data loader...")
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_PATH, 
        batch_size=16,
        val_split=0.2
    )
    
    # Test loading a batch from each loader
    print("\nTesting batch loading...")
    
    train_images, train_labels = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  Images shape: {train_images.shape}")
    print(f"  Labels shape: {train_labels.shape}")
    print(f"  Image range: [{train_images.min():.3f}, {train_images.max():.3f}]")
    print(f"  Labels: {train_labels.tolist()}")
    
    val_images, val_labels = next(iter(val_loader))
    print(f"\nValidation batch:")
    print(f"  Images shape: {val_images.shape}")
    print(f"  Image range: [{val_images.min():.3f}, {val_images.max():.3f}]")
    
    test_images, test_labels = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {test_images.shape}")
    print(f"  Image range: [{test_images.min():.3f}, {test_images.max():.3f}]")
    
    print("\n✓ Data loader test successful!")
