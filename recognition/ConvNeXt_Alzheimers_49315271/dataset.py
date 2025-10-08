# dataset.py
# Data loader for ADNI Alzheimer's classification

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

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


def get_data_loaders(data_path, batch_size=32, num_workers=2):
    """
    Create train and test data loaders with appropriate transforms
    
    Args:
        data_path: Path to AD_NC folder
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),  # Increase from 10
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Add random crop
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),  # Stronger
        transforms.RandomGrayscale(p=0.1),  # Occasionally convert to grayscale
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Add blur
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    test_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'test'),
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_data_loaders_proper_split(data_path, batch_size=32, val_split=0.2, random_seed=42, num_workers=2):
    """
    Create train, validation, and test loaders with proper splits
    
    Args:
        data_path: Path to AD_NC folder
        batch_size: Batch size for training
        val_split: Fraction of training data to use for validation (0.2 = 20%)
        random_seed: Random seed for reproducible splits
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import random_split
    import numpy as np
    
    # Set random seed for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation and testing
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full training dataset with augmentation
    full_train_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'train'),
        transform=train_transform
    )
    
    # Split training data into train and validation
    total_size = len(full_train_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset_with_aug = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    print(f"\nData split:")
    print(f"Training: {train_size} images")
    print(f"Validation: {val_size} images")
    
    # Create validation dataset WITHOUT augmentation
    # We need to reload the validation indices with eval_transform
    val_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'train'),
        transform=eval_transform
    )
    
    # Get the same validation indices
    val_indices = val_dataset_with_aug.indices
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Load test dataset (completely untouched during training)
    test_dataset = ADNIDataset(
        root_dir=os.path.join(data_path, 'test'),
        transform=eval_transform
    )
    print(f"Test: {len(test_dataset)} images (held out)\n")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
