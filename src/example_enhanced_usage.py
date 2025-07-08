"""
Example script demonstrating how to use the enhanced training infrastructure
with different classification schemes and weighted loss functions.
"""

import torch
import torch.nn as nn
from torchvision import models

from classification_schemes import ClassificationScheme, ClassificationSchemeManager
from weighted_loss_utils import WeightedCrossEntropyLoss, WeightingStrategy
from enhanced_training import (
    create_enhanced_dataloaders,
    create_weighted_criterion,
    EnhancedTrainer
)


def demo_classification_schemes():
    """Demonstrate different classification schemes."""
    print("=== Classification Scheme Demo ===\n")
    
    # Example class directories from the dataset
    example_classes = [
        "Apple-Healthy-Real-Real",
        "Apple-Healthy-Diffusion-DS8",
        "Apple-Healthy-GAN-StyleGAN2",
        "Apple-Unhealthy-Real-Real",
        "Apple-Unhealthy-Diffusion-SPx2Px",
        "Apple-Unhealthy-GAN-Stylegan3"
    ]
    
    manager = ClassificationSchemeManager("Apple")
    
    # Test each scheme
    for scheme in [ClassificationScheme.BINARY, ClassificationScheme.FOUR_WAY, ClassificationScheme.TEN_WAY]:
        print(f"\n{scheme.value.upper()} Classification:")
        scheme_info = manager.get_scheme_info(scheme)
        print(f"Description: {scheme_info['description']}")
        print(f"Number of classes: {scheme_info['num_classes']}")
        
        print("\nLabel mappings:")
        for class_dir in example_classes:
            mapped_class, label_idx = manager.get_mapped_label(class_dir, scheme)
            print(f"  {class_dir} -> {mapped_class} (label={label_idx})")


def demo_weighted_loss():
    """Demonstrate weighted loss calculation."""
    print("\n\n=== Weighted Loss Demo ===\n")
    
    # Simulate class distribution (imbalanced)
    class_counts = {
        0: 1000,  # Healthy
        1: 200    # Unhealthy (5:1 imbalance)
    }
    
    print("Class distribution:")
    for cls, count in class_counts.items():
        print(f"  Class {cls}: {count} samples")
    
    # Test different weighting strategies
    for strategy in [WeightingStrategy.INVERSE_FREQUENCY, 
                     WeightingStrategy.BALANCED, 
                     WeightingStrategy.EFFECTIVE_NUMBER]:
        print(f"\n{strategy} weights:")
        
        criterion = WeightedCrossEntropyLoss(
            class_counts=class_counts,
            strategy=strategy
        )
        
        if criterion.weights is not None:
            weights = criterion.weights.cpu().numpy()
            for i, w in enumerate(weights):
                print(f"  Class {i}: weight={w:.4f}")


def demo_enhanced_training_setup():
    """Demonstrate setting up enhanced training with different schemes."""
    print("\n\n=== Enhanced Training Setup Demo ===\n")
    
    # Configuration
    config = {
        'root_dir': './Dataset',
        'plant_name': 'Apple',
        'batch_size': 32,
        'num_workers': 4,
        'classification_scheme': ClassificationScheme.FOUR_WAY,
        'use_weighted_sampler': True,
        'weighting_strategy': WeightingStrategy.BALANCED
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nNote: To actually run training, you would:")
    print("1. Create dataloaders using create_enhanced_dataloaders()")
    print("2. Create a model appropriate for the number of classes")
    print("3. Create weighted loss using create_weighted_criterion()")
    print("4. Initialize EnhancedTrainer with all components")
    print("5. Call trainer.train() to start training")
    
    # Example code structure (not executable without actual data)
    print("\nExample code structure:")
    print("""
    # Create dataloaders
    train_loader, val_loader, test_loader, dataset_info = create_enhanced_dataloaders(
        root_dir=config['root_dir'],
        plant_name=config['plant_name'],
        classification_scheme=config['classification_scheme'],
        batch_size=config['batch_size'],
        use_weighted_sampler=config['use_weighted_sampler'],
        weighting_strategy=config['weighting_strategy']
    )
    
    # Create model
    num_classes = dataset_info['num_classes']
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Create weighted loss
    criterion = create_weighted_criterion(
        dataset_info=dataset_info,
        weighting_strategy=config['weighting_strategy']
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create trainer
    trainer = EnhancedTrainer(
        model=model,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        plant_name=config['plant_name'],
        classification_scheme=config['classification_scheme']
    )
    
    # Train model
    trainer.train(num_epochs=50, save_path='models/best_model.pth')
    """)


if __name__ == "__main__":
    # Run demonstrations
    demo_classification_schemes()
    demo_weighted_loss()
    demo_enhanced_training_setup()
    
    print("\n\n=== Infrastructure Components Summary ===")
    print("\n1. classification_schemes.py:")
    print("   - ClassificationScheme enum for different schemes")
    print("   - ClassificationSchemeManager for label mapping")
    print("   - Support for Binary, 4-way, 10-way, and Full classifications")
    
    print("\n2. weighted_loss_utils.py:")
    print("   - WeightedCrossEntropyLoss with automatic weight calculation")
    print("   - Multiple weighting strategies (inverse frequency, balanced, effective number)")
    print("   - Class distribution analysis utilities")
    print("   - Support for weighted sampling")
    
    print("\n3. enhanced_training.py:")
    print("   - EnhancedPlantDiseaseDataset supporting classification schemes")
    print("   - EnhancedTrainer with per-class metrics tracking")
    print("   - create_enhanced_dataloaders() for easy setup")
    print("   - Backward compatible with existing training code")