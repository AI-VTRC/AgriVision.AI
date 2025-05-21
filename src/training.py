import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import re
from sklearn.model_selection import train_test_split

class LeafDataset(Dataset):
    def __init__(self, root_dir, plant_name, split='train', transform=None, train_ratio=0.7, val_ratio=0.15, random_state=42):
        """
        Initialize the dataset for leaf classification.
        
        Args:
            root_dir (str): Root directory containing the dataset
            plant_name (str): Name of the plant (e.g., 'Apple')
            split (str): Dataset split ('train', 'val', or 'test')
            transform (callable, optional): Optional transform to be applied to the images
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            random_state (int): Random seed for reproducibility
        """
        self.root_dir = root_dir
        self.plant_name = plant_name
        self.split = split
        self.transform = transform
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.images = []
        self.labels = []
        self.data = []
        
        # Dictionary to map from category to class index
        self.category_to_idx = {
            f"{plant_name}-Healthy-Real-Real": 0,
            f"{plant_name}-Unhealthy-Real-Real": 1,
            f"{plant_name}-Healthy-Diffusion-SDXL": 2,
            f"{plant_name}-Unhealthy-Diffusion-SDXL": 3,
            f"{plant_name}-Healthy-Diffusion-SPx2Px": 4,
            f"{plant_name}-Unhealthy-Diffusion-SPx2Px": 5,
            f"{plant_name}-Healthy-GAN-DCGAN": 6,
            f"{plant_name}-Unhealthy-GAN-DCGAN": 7,
            f"{plant_name}-Healthy-GAN-StyleGAN2": 8,
            f"{plant_name}-Unhealthy-GAN-StyleGAN2": 9
        }
        
        # Load images and labels
        self._load_data()
    
    def _parse_folder_name(self, folder_name):
        """
        Parse a folder name to extract plant, health status, and generation method.
        
        Args:
            folder_name (str): Name of the folder to parse
            
        Returns:
            tuple: (plant_name, health_status, generation_type, generation_method)
        """
        parts = folder_name.split('-')
        
        if len(parts) < 3:
            return None, None, None, None
        
        plant_name = parts[0]
        health_status = parts[1]  # Healthy or Unhealthy
        
        # Handle different generation methods
        if len(parts) == 3:
            # Real images have only 3 parts, e.g., "Apple-Healthy-Real"
            generation_type = parts[2]
            generation_method = parts[2]
        else:
            # Generated images have 4 parts, e.g., "Apple-Healthy-Diffusion-SDXL"
            generation_type = parts[2]
            generation_method = parts[3]
            
        return plant_name, health_status, generation_type, generation_method
    
    def _load_data(self):
        """Load images and their corresponding labels from the directory structure."""
        dataset_path = os.path.join(self.root_dir, self.plant_name)
        
        if not os.path.exists(dataset_path):
            raise ValueError(f"Dataset path {dataset_path} does not exist.")
        
        # All image paths and labels
        all_data = []
        
        # Walk through the directory structure
        for category_folder in os.listdir(dataset_path):
            category_path = os.path.join(dataset_path, category_folder)
            
            if not os.path.isdir(category_path):
                continue
                
            if category_folder in self.category_to_idx:
                class_idx = self.category_to_idx[category_folder]
                
                # Collect all image files in this category
                for img_file in os.listdir(category_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(category_path, img_file)
                        all_data.append((img_path, class_idx))
        
        # Split the dataset indices instead of moving files
        all_indices = np.arange(len(all_data))
        all_labels = [label for _, label in all_data]
        
        # Split indices for train/val/test
        train_indices, test_val_indices = train_test_split(
            all_indices, 
            train_size=self.train_ratio, 
            random_state=self.random_state,
            stratify=all_labels
        )
        
        # Calculate validation ratio out of the remaining data
        val_ratio_adjusted = self.val_ratio / (1 - self.train_ratio)
        
        # Get the labels for the test_val set
        test_val_labels = [all_labels[i] for i in test_val_indices]
        
        # Split the remaining indices into validation and test sets
        val_indices, test_indices = train_test_split(
            test_val_indices,
            train_size=val_ratio_adjusted,
            random_state=self.random_state,
            stratify=test_val_labels
        )
        
        # Select the appropriate split
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = val_indices
        elif self.split == 'test':
            split_indices = test_indices
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', or 'test'.")
        
        # Store the data for this split
        self.data = [all_data[i] for i in split_indices]
        
        print(f"Loaded {len(self.data)} images for {self.split} split")
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            tuple: (image, label) where label is the class index
        """
        img_path, label = self.data[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, criterion, optimizer, scheduler=None):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            device: The device to train on (CPU or GPU)
            train_loader: DataLoader for the training set
            val_loader: DataLoader for the validation set
            criterion: Loss function
            optimizer: Optimizer for the model parameters
            scheduler: Learning rate scheduler (optional)
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar for training
        train_pbar = tqdm(self.train_loader, desc="Training")
        
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            train_pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
            
        if self.scheduler is not None:
            self.scheduler.step()
            
        train_loss = running_loss / total
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def validate(self):
        """Validate the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(self.val_loader, desc="Validation")
            
            for inputs, targets in val_pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                val_pbar.set_postfix({
                    'loss': running_loss/total,
                    'acc': 100.*correct/total
                })
                
        val_loss = running_loss / total
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs, save_path=None):
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs (int): Number of epochs to train
            save_path (str, optional): Path to save the model checkpoints
            
        Returns:
            dict: Dictionary containing training and validation metrics
        """
        best_val_acc = 0.0
        metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            
            val_loss, val_acc = self.validate()
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save the best model
            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
                
        return metrics


def get_transforms(split='train', model_type=None):
    """
    Get transforms for data augmentation and normalization.
    
    Args:
        split (str): Dataset split ('train', 'val', or 'test')
        model_type (str, optional): Type of model ('clip' or None for standard models)
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    if model_type == 'clip':
        import clip
        _, preprocess = clip.load("ViT-B/32", device='cpu')
        
        # For CLIP models
        if split == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.ToTensor(),  # Convert PIL image to tensor
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            ])
        else:
            # For validation/testing, use standard CLIP preprocessing
            return transforms.Compose([
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),  # Convert PIL image to tensor
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)
                )
            ])
        
    # Standard transforms for other models (ImageNet normalization)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])


def create_dataloaders(root_dir, plant_name, batch_size=32, num_workers=4, train_ratio=0.7, val_ratio=0.15, random_state=42, model_type=None):
    """
    Create dataloaders for training, validation, and testing.
    
    Args:
        root_dir (str): Root directory containing the dataset
        plant_name (str): Name of the plant (e.g., 'Apple')
        batch_size (int): Batch size for the dataloaders
        num_workers (int): Number of workers for data loading
        train_ratio (float): Ratio of training data
        val_ratio (float): Ratio of validation data
        random_state (int): Random seed for reproducibility
        model_type (str, optional): Type of model ('clip' or None for standard models)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    train_transform = get_transforms('train', model_type)
    val_transform = get_transforms('val', model_type)
    test_transform = get_transforms('test', model_type)
    
    train_dataset = LeafDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        split='train',
        transform=train_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
    val_dataset = LeafDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        split='val',
        transform=val_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
    test_dataset = LeafDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        split='test',
        transform=test_transform,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=random_state
    )
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader 