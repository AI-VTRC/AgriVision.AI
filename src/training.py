import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import logging
from typing import Tuple, List, Optional
from glob import glob
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PlantDiseaseDataset(Dataset):
    """Dataset class for plant disease classification."""
    
    def __init__(self, root_dir: str, plant_name: str, classification_type: str = 'binary', transform=None, model_type: Optional[str] = None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            plant_name: Name of the plant (e.g., 'Apple', 'Maize', 'Tomato')
            classification_type: Type of classification ('binary', 'generation', 'detailed')
            transform: Torchvision transforms to apply
            model_type: Type of model being used (e.g., 'clip')
        """
        self.root_dir = root_dir
        self.plant_name = plant_name
        self.classification_type = classification_type
        self.transform = transform
        self.model_type = model_type
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # Load dataset
        self._load_dataset()
        
    def _load_dataset(self):
        """Load dataset by scanning directories."""
        # Debug logging
        logger.info(f"Looking for plant: {self.plant_name} in directory: {self.root_dir}")
        logger.info(f"Root directory exists: {os.path.exists(self.root_dir)}")
        
        # Get all subdirectories for this plant - improved Windows compatibility
        # First, get all directories in root and filter manually for better cross-platform compatibility
        try:
            all_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
            logger.info(f"All directories in root: {all_dirs}")
            
            # Filter directories that start with plant name
            plant_dirs = [d for d in all_dirs if d.startswith(f"{self.plant_name}-")]
            logger.info(f"Found directories for {self.plant_name}: {plant_dirs}")
            
            if not plant_dirs:
                # Try case-insensitive search
                plant_dirs = [d for d in all_dirs if d.lower().startswith(f"{self.plant_name.lower()}-")]
                logger.info(f"Case-insensitive match: {plant_dirs}")
                
                if not plant_dirs:
                    raise ValueError(f"No directories found for plant: {self.plant_name}. Available directories: {all_dirs}")
            
            # Convert to full paths
            plant_dirs = [os.path.join(self.root_dir, d) for d in plant_dirs]
            logger.info(f"Full paths: {plant_dirs}")
            
        except Exception as e:
            logger.error(f"Error scanning directory {self.root_dir}: {e}")
            raise ValueError(f"Could not scan directory {self.root_dir}: {e}")
        
        # Sort for consistent ordering
        plant_dirs.sort()
        
        # Build class mappings
        for idx, dir_path in enumerate(plant_dirs):
            class_name = os.path.basename(dir_path)
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        logger.info(f"Found {len(plant_dirs)} classes for {self.plant_name}")
        logger.info(f"Classes: {list(self.class_to_idx.keys())}")
        
        # Load all image files and filter out classes with no images
        valid_classes = {}
        new_class_idx = 0
        
        for class_name, original_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            
            image_files = []
            for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]:
                image_files.extend(glob(os.path.join(class_dir, ext)))
            
            logger.info(f"  {class_name}: {len(image_files)} images")
            
            if image_files:
                valid_classes[class_name] = new_class_idx
                for img_path in image_files:
                    self.samples.append((img_path, new_class_idx))
                new_class_idx += 1
            else:
                logger.warning(f"  Skipping empty class directory: {class_name}")

        self.class_to_idx = valid_classes
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"Filtered to {len(self.class_to_idx)} non-empty classes.")
        logger.info(f"Total samples across valid classes: {len(self.samples)}")
        
        if self.classification_type == 'binary':
            logger.info("Converting to binary classification (Healthy vs Unhealthy)")
            self._convert_to_binary()
        elif self.classification_type == 'generation':
            logger.info("Converting to generation-based classification (Real vs GAN vs Diffusion)")
            self._convert_to_generation()
        elif self.classification_type == 'detailed':
            logger.info("Using detailed classification (all subdirectories as separate classes)")
        else:
            raise ValueError(f"Unknown classification type: {self.classification_type}")
    
    def _convert_to_generation(self):
        """Convert multi-class labels to generation-based (Real=0, GAN=1, Diffusion=2)."""
        new_samples = []
        for img_path, _ in self.samples:
            dir_name = os.path.basename(os.path.dirname(img_path))
            
            if "Real" in dir_name:
                label = 0  # Real
            elif "GAN" in dir_name:
                label = 1  # GAN
            elif "Diffusion" in dir_name:
                label = 2  # Diffusion
            else:
                logger.warning(f"Could not determine generation type for directory: {dir_name}")
                label = 0  # Default to Real
            
            new_samples.append((img_path, label))
        
        self.samples = new_samples
        self.class_to_idx = {"Real": 0, "GAN": 1, "Diffusion": 2}
        self.idx_to_class = {0: "Real", 1: "GAN", 2: "Diffusion"}
        
        real_count = sum(1 for _, label in self.samples if label == 0)
        gan_count = sum(1 for _, label in self.samples if label == 1)
        diffusion_count = sum(1 for _, label in self.samples if label == 2)
        logger.info(f"Generation classification: Real={real_count}, GAN={gan_count}, Diffusion={diffusion_count}")
    
    def _check_binary_classification(self) -> bool:
        """Check if this is a binary classification task."""
        has_healthy = any("Healthy" in class_name for class_name in self.class_to_idx.keys())
        has_unhealthy = any("Unhealthy" in class_name for class_name in self.class_to_idx.keys())
        return has_healthy and has_unhealthy
    
    def _convert_to_binary(self):
        """Convert multi-class labels to binary (healthy=0, unhealthy=1)."""
        new_samples = []
        for img_path, _ in self.samples:
            # Determine binary label based on directory name
            dir_name = os.path.basename(os.path.dirname(img_path))
            
            # Check for Healthy vs Unhealthy in directory name
            # Handle cases like "Tomato-Unhealthy--Real-Real" with double dashes
            if "Healthy" in dir_name and "Unhealthy" not in dir_name:
                label = 0  # Healthy
            elif "Unhealthy" in dir_name:
                label = 1  # Unhealthy
            else:
                # Fallback: if neither is explicitly found, log warning
                logger.warning(f"Could not determine health status for directory: {dir_name}")
                label = 1  # Default to unhealthy
            
            new_samples.append((img_path, label))
        
        self.samples = new_samples
        self.class_to_idx = {"Healthy": 0, "Unhealthy": 1}
        self.idx_to_class = {0: "Healthy", 1: "Unhealthy"}
        
        # Count samples per class
        healthy_count = sum(1 for _, label in self.samples if label == 0)
        unhealthy_count = len(self.samples) - healthy_count
        logger.info(f"Binary classification: Healthy={healthy_count}, Unhealthy={unhealthy_count}")
        
        # Log class distribution for verification
        if healthy_count == 0:
            logger.warning("No healthy samples found!")
        if unhealthy_count == 0:
            logger.warning("No unhealthy samples found!")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {e}")
            # Return a blank image in case of error
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def create_dataloaders(
    root_dir: str,
    plant_name: str,
    classification_type: str = 'binary',
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
    model_type: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        root_dir: Root directory containing the dataset
        plant_name: Name of the plant
        classification_type: Type of classification ('binary', 'generation', 'detailed')
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_state: Random seed for reproducibility
        model_type: Type of model being used
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    if model_type == 'clip':
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                               std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        # Standard ImageNet normalization for other models
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    # Create full dataset
    full_dataset = PlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        classification_type=classification_type,
        transform=train_transform,
        model_type=model_type
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f"Dataset split: train={train_size}, val={val_size}, test={test_size}")
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Update transforms for validation and test sets
    # Create new datasets with appropriate transforms
    val_dataset_copy = PlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        classification_type=classification_type,
        transform=val_transform,
        model_type=model_type
    )
    val_dataset_copy.samples = [full_dataset.samples[i] for i in val_dataset.indices]
    
    test_dataset_copy = PlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        classification_type=classification_type,
        transform=val_transform,
        model_type=model_type
    )
    test_dataset_copy.samples = [full_dataset.samples[i] for i in test_dataset.indices]
    
    # Device optimization settings (CUDA or MPS)
    pin_memory = torch.cuda.is_available() or torch.backends.mps.is_available()
    persistent_workers = num_workers > 0 and (torch.cuda.is_available() or torch.backends.mps.is_available())
    
    # Create data loaders with CUDA optimizations
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    val_loader = DataLoader(
        val_dataset_copy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    test_loader = DataLoader(
        test_dataset_copy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if num_workers > 0 else 2
    )
    
    return train_loader, val_loader, test_loader


class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        progress_interval: int = 10,
        plant_name: str = "Plant",
        classification_type: str = "binary",
        model_name: str = "efficientnet_b0",
        checkpoint: Optional[dict] = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            progress_interval: Interval for progress updates
            plant_name: Name of the plant
            classification_type: Type of classification
            model_name: Name of the model
            checkpoint: Checkpoint to resume from
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.progress_interval = progress_interval
        self.plant_name = plant_name
        self.classification_type = classification_type
        self.model_name = model_name
        self.start_epoch = 1
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0

        if checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.metrics = checkpoint.get('metrics', self.metrics)
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            logger.info(f"Resumed from epoch {checkpoint['epoch']} with best val_acc: {self.best_val_acc:.2f}%")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} - Training')
        
        for batch_idx, (inputs, labels) in enumerate(pbar):
            # Move data to device with non_blocking for CUDA optimization
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Clear cache periodically for GPU memory optimization (CUDA or MPS)
            if batch_idx % 50 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            
            # Update progress bar
            if batch_idx % self.progress_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100. * correct / total
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} - Validation')
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                # Move data to device with non_blocking for CUDA optimization
                inputs = inputs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Clear cache periodically for GPU memory optimization (CUDA or MPS)
                if batch_idx % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    elif torch.backends.mps.is_available():
                        torch.mps.empty_cache()
        
        # Calculate validation metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, num_epochs: int, save_path: str) -> dict:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save the best model
        
        Returns:
            Dictionary containing training metrics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            self.metrics['train_loss'].append(train_loss)
            self.metrics['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(epoch)
            self.metrics['val_loss'].append(val_loss)
            self.metrics['val_acc'].append(val_acc)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                
                logger.info(f"New best model! Saving to {save_path}")
                
                # Save checkpoint
                # Try to get class names from the dataset
                class_names = None
                try:
                    if hasattr(self.train_loader.dataset, 'dataset'):
                        original_dataset = self.train_loader.dataset.dataset  # type: ignore
                        if hasattr(original_dataset, 'class_to_idx'):
                            class_names = list(original_dataset.class_to_idx.keys())  # type: ignore
                    elif hasattr(self.train_loader.dataset, 'class_to_idx'):
                        class_names = list(self.train_loader.dataset.class_to_idx.keys())  # type: ignore
                except Exception as e:
                    logger.warning(f"Could not extract class names: {e}")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'best_val_acc': self.best_val_acc,
                    'metrics': self.metrics,
                    'plant_name': self.plant_name,
                    'classification_type': self.classification_type,
                    'model_name': self.model_name,
                    'class_names': class_names
                }
                
                # Robust saving: save to temp file, then rename
                save_dir = os.path.dirname(save_path)
                os.makedirs(save_dir, exist_ok=True)
                
                temp_file_path = f"{save_path}.tmp"
                try:
                    torch.save(checkpoint, temp_file_path)
                    os.rename(temp_file_path, save_path)
                except Exception as e:
                    logger.error(f"Failed to save checkpoint to {save_path}: {e}")
                    # Clean up temp file if it exists
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
                    raise   # Re-raise the exception to stop the process if saving fails
        
        logger.info(f"\nTraining completed!")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}% at epoch {self.best_epoch}")
        
        return self.metrics