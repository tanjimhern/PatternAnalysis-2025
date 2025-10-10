# dataset.py
# Data loader for ADNI Alzheimer's classification
# Updated with data-specific normalization (not ImageNet!)

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np


class ADNIDataset(Dataset):
    """
    Dataset class for ADNI brain MRI images
    Loads JPEG images from AD and NC folders
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to train or test folder containing AD/NC subfolders
            transform: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load AD images (label = 1)
        ad_path = os.path.join(root_dir, 'AD')
        for img_name in os.listdir(ad_path):
            if img_name.endswith('.jpeg') or img_name.endswith('.jpg'):
                self.images.append(os.path.join(ad_path, img_name))
                self.labels.append(1)
        
        # Load NC images (label = 0)
        nc_path = os.path.join(root_dir, 'NC')
        for img_name in os.listdir(nc_path):
            if img_name.endswith('.jpeg') or img_name.endswith('.jpg'):
                self.images.append(os.path.join(nc_path, img_name))
                self.labels.append(0)
        
        print(f"Loaded {len(self.images)} images from {root_dir}")
        print(f"AD samples: {sum(self.labels)}, NC samples: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Transformed image tensor
            label: 0 for NC, 1 for AD
        """
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # Convert grayscale to RGB
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_loaders(data_path, batch_size=32, num_workers=2, use_val_split=True, val_split=0.2, random_seed=42):
    """
    Create train, validation, and test data loaders with appropriate transforms
    
    Args:
        data_path: Path to AD_NC folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_val_split: If True, split train into train/val. If False, use test as val.
        val_split: Proportion of train data to use for validation (if use_val_split=True)
        random_seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # CRITICAL: Use normalization computed from YOUR ADNI data, not ImageNet!
    # These values were computed from the actual ADNI train set:
    # Mean: [0.115, 0.115, 0.115]
    # Std:  [0.225, 0.225, 0.225]
    ADNI_MEAN = [0.115, 0.115, 0.115]
    ADNI_STD = [0.225, 0.225, 0.225]
    
    print("\n" + "="*80)
    print("NORMALIZATION INFO:")
    print("="*80)
    print(f"Using ADNI-specific normalization (NOT ImageNet):")
    print(f"  Mean: {ADNI_MEAN}")
    print(f"  Std:  {ADNI_STD}")
    print("="*80 + "\n")
    
    # Data augmentation for training
    # Note: Reduced augmentation intensity for medical images
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Reduced from 20
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),  # Less aggressive
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Reduced
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),  # BEFORE ToTensor!
        transforms.ToTensor(),
        transforms.Normalize(mean=ADNI_MEAN, std=ADNI_STD)  # ADNI-specific!
    ])
    
    # No augmentation for validation/testing
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=ADNI_MEAN, std=ADNI_STD)  # ADNI-specific!
    ])
    
    # Create train dataset
    train_dataset_full = ADNIDataset(
        root_dir=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    # Create test dataset
    test_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'test'),
        transform=eval_transform
    )
    
    # Handle validation split
    if use_val_split:
        # Split train into train + val
        print(f"\nSplitting train set: {100*(1-val_split):.0f}% train, {100*val_split:.0f}% validation")
        
        # Set random seed for reproducibility
        torch.manual_seed(random_seed)
        
        val_size = int(val_split * len(train_dataset_full))
        train_size = len(train_dataset_full) - val_size
        
        train_dataset, val_dataset = random_split(
            train_dataset_full, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed)
        )
        
        print(f"Train samples: {train_size}")
        print(f"Validation samples: {val_size}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
    else:
        # Use test as validation (your original approach)
        print("\nWARNING: Using test set as validation!")
        print("This is not recommended for final evaluation.")
        
        train_dataset = train_dataset_full
        val_dataset = test_dataset
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Test loader (always from test set)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Backward compatibility - if someone calls the old function name
def get_data_loaders_old(data_path, batch_size=32, num_workers=2):
    """
    Old interface - returns only train and test loaders
    Kept for backward compatibility
    """
    train_loader, _, test_loader = get_data_loaders(
        data_path, 
        batch_size, 
        num_workers, 
        use_val_split=False
    )
    return train_loader, test_loader


if __name__ == "__main__":
    """
    Test the data loader
    """
    DATA_PATH = '/home/groups/comp3710/ADNI/AD_NC'
    
    print("Testing data loader with validation split...")
    train_loader, val_loader, test_loader = get_data_loaders(
        DATA_PATH, 
        batch_size=16,
        use_val_split=True,
        val_split=0.2
    )
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
    
    print("\nData loader test successful!")
