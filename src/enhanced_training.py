"""
Enhanced training module that extends the existing training functionality with support for
multiple classification schemes and weighted loss functions.

This module provides:
- Enhanced dataset class supporting multiple classification schemes
- Integration with weighted loss utilities
- Backward compatibility with existing training code
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from typing import Tuple, List, Optional, Dict
import logging
from glob import glob

from training import PlantDiseaseDataset, Trainer
from classification_schemes import ClassificationScheme, ClassificationSchemeManager
from weighted_loss_utils import (
    WeightedCrossEntropyLoss,
    WeightingStrategy,
    analyze_class_distribution,
    create_weighted_sampler,
    get_recommended_loss_params
)

logger = logging.getLogger(__name__)


class EnhancedPlantDiseaseDataset(PlantDiseaseDataset):
    """
    Enhanced dataset class that supports multiple classification schemes.
    Extends the existing PlantDiseaseDataset with scheme-based label mapping.
    """
    
    def __init__(
        self,
        root_dir: str,
        plant_name: str,
        transform=None,
        model_type: Optional[str] = None,
        classification_scheme: ClassificationScheme = ClassificationScheme.FULL
    ):
        """
        Initialize the enhanced dataset.
        
        Args:
            root_dir: Root directory containing the dataset
            plant_name: Name of the plant (e.g., 'Apple', 'Maize', 'Tomato')
            transform: Torchvision transforms to apply
            model_type: Type of model being used (e.g., 'clip')
            classification_scheme: Classification scheme to use
        """
        self.classification_scheme = classification_scheme
        self.scheme_manager = ClassificationSchemeManager(plant_name)
        
        # Initialize parent class (will call _load_dataset)
        super().__init__(root_dir, plant_name, transform, model_type)
        
        # Apply classification scheme after loading
        if classification_scheme != ClassificationScheme.FULL:
            self._apply_classification_scheme()
    
    def _load_dataset(self):
        """Override to prevent automatic binary conversion."""
        # Call grandparent's _load_dataset logic without binary conversion
        # Get all subdirectories for this plant
        plant_dirs = glob(os.path.join(self.root_dir, f"{self.plant_name}-*"))
        plant_dirs = [d for d in plant_dirs if os.path.isdir(d)]
        
        if not plant_dirs:
            raise ValueError(f"No directories found for plant: {self.plant_name}")
        
        # Sort for consistent ordering
        plant_dirs.sort()
        
        # Store original class names for scheme mapping
        self.original_classes = [os.path.basename(d) for d in plant_dirs]
        
        # Build initial class mappings (will be updated by scheme)
        for idx, dir_path in enumerate(plant_dirs):
            class_name = os.path.basename(dir_path)
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
        
        logger.info(f"Found {len(plant_dirs)} original classes for {self.plant_name}")
        
        # Load all image files
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(self.root_dir, class_name)
            
            # Support multiple image formats
            image_patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
            image_files = []
            
            for pattern in image_patterns:
                image_files.extend(glob(os.path.join(class_dir, pattern)))
            
            for img_path in image_files:
                self.samples.append((img_path, class_idx))
            
            logger.info(f"  {class_name}: {len(image_files)} images")
        
        logger.info(f"Total samples: {len(self.samples)}")
    
    def _apply_classification_scheme(self):
        """Apply the selected classification scheme to remap labels."""
        logger.info(f"Applying {self.classification_scheme.value} classification scheme")
        
        # Get scheme information
        scheme_info = self.scheme_manager.get_scheme_info(self.classification_scheme)
        logger.info(f"Scheme: {scheme_info['description']}")
        
        # Create new label mapping
        new_samples = []
        label_mapping = {}
        
        for img_path, _ in self.samples:
            # Get original class from directory
            dir_name = os.path.basename(os.path.dirname(img_path))
            
            # Map to new label using scheme
            mapped_class, mapped_idx = self.scheme_manager.get_mapped_label(
                dir_name, self.classification_scheme
            )
            
            # Build consistent label mapping
            if mapped_class not in label_mapping:
                if mapped_idx == -1:  # Full scheme case
                    label_mapping[mapped_class] = len(label_mapping)
                else:
                    label_mapping[mapped_class] = mapped_idx
            
            new_samples.append((img_path, label_mapping[mapped_class]))
        
        # Update dataset properties
        self.samples = new_samples
        
        # Create proper class mappings
        if self.classification_scheme == ClassificationScheme.FULL:
            # For full scheme, use the dynamically created mapping
            self.class_to_idx = label_mapping
            self.idx_to_class = {v: k for k, v in label_mapping.items()}
        else:
            # For predefined schemes, use the scheme's class names
            self.class_to_idx = self.scheme_manager.get_label_mapping(
                self.classification_scheme, self.original_classes
            )
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Log class distribution
        from collections import Counter
        label_counts = Counter([label for _, label in self.samples])
        logger.info(f"Class distribution after scheme application:")
        for class_name, class_idx in sorted(self.class_to_idx.items(), key=lambda x: x[1]):
            count = label_counts.get(class_idx, 0)
            logger.info(f"  {class_name} (idx={class_idx}): {count} samples")
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        from collections import Counter
        labels = [label for _, label in self.samples]
        return dict(Counter(labels))


class EnhancedTrainer(Trainer):
    """
    Enhanced trainer that supports weighted loss functions and multiple classification schemes.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        progress_interval: int = 10,
        plant_name: str = "Plant",
        classification_scheme: ClassificationScheme = ClassificationScheme.FULL
    ):
        """
        Initialize the enhanced trainer.
        
        Args:
            model: Model to train
            device: Device to use for training
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function (can be WeightedCrossEntropyLoss)
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            progress_interval: Interval for progress updates
            plant_name: Name of the plant
            classification_scheme: Classification scheme being used
        """
        super().__init__(
            model, device, train_loader, val_loader,
            criterion, optimizer, scheduler, progress_interval, plant_name
        )
        
        self.classification_scheme = classification_scheme
        
        # Store additional metrics for weighted training
        self.class_metrics = {
            'per_class_accuracy': [],
            'per_class_loss': []
        }
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Extended validation that also computes per-class metrics.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class tracking
        class_correct = {}
        class_total = {}
        
        # Get number of classes from the dataset
        if hasattr(self.val_loader.dataset, 'dataset'):
            # When using random_split
            num_classes = len(self.val_loader.dataset.dataset.class_to_idx)
            class_names = self.val_loader.dataset.dataset.idx_to_class
        else:
            # Direct dataset
            num_classes = len(self.val_loader.dataset.class_to_idx)
            class_names = self.val_loader.dataset.idx_to_class
        
        for i in range(num_classes):
            class_correct[i] = 0
            class_total[i] = 0
        
        with torch.no_grad():
            from tqdm import tqdm
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} - Validation')
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Per-class statistics
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i].item() == label:
                        class_correct[label] += 1
        
        # Calculate overall metrics
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for class_idx in range(num_classes):
            if class_total[class_idx] > 0:
                per_class_acc[class_idx] = 100. * class_correct[class_idx] / class_total[class_idx]
            else:
                per_class_acc[class_idx] = 0.0
        
        # Log per-class metrics
        logger.info("\nPer-class validation accuracy:")
        for class_idx, acc in sorted(per_class_acc.items()):
            class_name = class_names.get(class_idx, f"Class_{class_idx}")
            logger.info(f"  {class_name}: {acc:.2f}% ({class_correct[class_idx]}/{class_total[class_idx]})")
        
        # Store per-class metrics
        self.class_metrics['per_class_accuracy'].append(per_class_acc)
        
        return val_loss, val_acc
    
    def save_checkpoint(self, save_path: str, epoch: int):
        """
        Extended checkpoint saving that includes scheme information.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'metrics': self.metrics,
            'class_metrics': self.class_metrics,
            'plant_name': self.plant_name,
            'classification_scheme': self.classification_scheme.value,
            'class_names': self._get_class_names()
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(checkpoint, save_path)
    
    def _get_class_names(self) -> Optional[List[str]]:
        """Get class names from the dataset."""
        if hasattr(self.train_loader.dataset, 'dataset'):
            # When using random_split
            return list(self.train_loader.dataset.dataset.class_to_idx.keys())
        elif hasattr(self.train_loader.dataset, 'class_to_idx'):
            # Direct dataset
            return list(self.train_loader.dataset.class_to_idx.keys())
        return None


def create_enhanced_dataloaders(
    root_dir: str,
    plant_name: str,
    classification_scheme: ClassificationScheme = ClassificationScheme.FULL,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    random_state: int = 42,
    model_type: Optional[str] = None,
    use_weighted_sampler: bool = False,
    weighting_strategy: str = WeightingStrategy.BALANCED
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create enhanced data loaders with support for classification schemes and weighted sampling.
    
    Args:
        root_dir: Root directory containing the dataset
        plant_name: Name of the plant
        classification_scheme: Classification scheme to use
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        random_state: Random seed for reproducibility
        model_type: Type of model being used
        use_weighted_sampler: Whether to use weighted sampling
        weighting_strategy: Strategy for calculating weights
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, dataset_info)
    """
    # Set random seed for reproducibility
    torch.manual_seed(random_state)
    
    # Define transforms based on model type
    if model_type == 'clip':
        from torchvision.transforms import Normalize
        
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                     std=[0.26862954, 0.26130258, 0.27577711])
        ])
    else:
        # Standard ImageNet normalization
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
    full_dataset = EnhancedPlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        transform=train_transform,
        model_type=model_type,
        classification_scheme=classification_scheme
    )
    
    # Get dataset information
    dataset_info = {
        'num_classes': len(full_dataset.class_to_idx),
        'class_names': list(full_dataset.class_to_idx.keys()),
        'classification_scheme': classification_scheme.value,
        'total_samples': len(full_dataset),
        'class_distribution': full_dataset.get_class_distribution()
    }
    
    # Analyze class distribution
    distribution_analysis = analyze_class_distribution(
        [label for _, label in full_dataset.samples],
        num_classes=dataset_info['num_classes']
    )
    dataset_info['distribution_analysis'] = distribution_analysis
    
    # Get recommended loss parameters
    loss_params = get_recommended_loss_params(
        dataset_info['class_distribution'],
        dataset_info['total_samples']
    )
    dataset_info['recommended_loss_params'] = loss_params
    
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
    
    # Create datasets with appropriate transforms
    val_dataset_copy = EnhancedPlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        transform=val_transform,
        model_type=model_type,
        classification_scheme=classification_scheme
    )
    val_dataset_copy.samples = [full_dataset.samples[i] for i in val_dataset.indices]
    
    test_dataset_copy = EnhancedPlantDiseaseDataset(
        root_dir=root_dir,
        plant_name=plant_name,
        transform=val_transform,
        model_type=model_type,
        classification_scheme=classification_scheme
    )
    test_dataset_copy.samples = [full_dataset.samples[i] for i in test_dataset.indices]
    
    # Create samplers if needed
    train_sampler = None
    if use_weighted_sampler:
        # Get labels for training subset
        train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
        from weighted_loss_utils import calculate_sample_weights
        
        sample_weights = calculate_sample_weights(train_labels, strategy=weighting_strategy)
        train_sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        logger.info("Using weighted sampler for training")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),  # Only shuffle if not using sampler
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset_copy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset_copy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader, dataset_info


def create_weighted_criterion(
    dataset_info: Dict,
    weighting_strategy: str = WeightingStrategy.BALANCED,
    device: Optional[torch.device] = None
) -> WeightedCrossEntropyLoss:
    """
    Create a weighted loss criterion based on dataset information.
    
    Args:
        dataset_info: Dictionary containing class distribution information
        weighting_strategy: Strategy for calculating weights
        device: Device to place weights on
        
    Returns:
        WeightedCrossEntropyLoss instance
    """
    class_distribution = dataset_info['class_distribution']
    
    # Create weighted loss
    criterion = WeightedCrossEntropyLoss(
        class_counts=class_distribution,
        strategy=weighting_strategy,
        device=device
    )
    
    logger.info(f"Created weighted loss with {weighting_strategy} strategy")
    
    return criterion