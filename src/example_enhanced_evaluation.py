"""
Example usage of the enhanced evaluation system.

This script demonstrates how to:
1. Evaluate models trained with different classification schemes
2. Use percentage-based confusion matrices
3. Compare performance across schemes
4. Export results for academic papers
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from torchvision import transforms

# Import from existing modules
from model import get_model
from training import PlantDiseaseDataset
from classification_schemes import ClassificationScheme, ClassificationSchemeManager
from enhanced_validation import EnhancedValidator
from comparison_visualizer import ComparisonVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_single_model(
    model_path: str,
    dataset_root: str,
    plant_name: str,
    scheme: ClassificationScheme,
    batch_size: int = 32,
    use_weighted_metrics: bool = False,
    output_dir: Optional[str] = None
):
    """
    Evaluate a single model with enhanced metrics.
    
    Args:
        model_path: Path to the model checkpoint
        dataset_root: Root directory of the dataset
        plant_name: Name of the plant
        scheme: Classification scheme used
        batch_size: Batch size for evaluation
        use_weighted_metrics: Whether to use weighted metrics
        output_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation results
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"outputs/{plant_name}_enhanced_evaluation_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset with the appropriate scheme
    logger.info(f"Loading dataset for {plant_name} with {scheme.value} classification")
    
    # Create test dataset (using existing PlantDiseaseDataset)
    # For now, we'll use the regular dataset and handle scheme mapping separately
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = PlantDiseaseDataset(
        root_dir=dataset_root,
        plant_name=plant_name,
        transform=test_transform,
        model_type=None
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Get class information based on scheme
    scheme_manager = ClassificationSchemeManager(plant_name)
    
    # Map classes based on scheme
    if scheme == ClassificationScheme.BINARY:
        class_names = ["Healthy", "Unhealthy"]
        num_classes = 2
    elif scheme == ClassificationScheme.FOUR_WAY:
        class_names = ["Real", "Diffusion", "GAN", "Mixed"]
        num_classes = 4
    elif scheme == ClassificationScheme.TEN_WAY:
        # Get from scheme manager
        scheme_info = scheme_manager.get_scheme_info(scheme)
        class_names = scheme_info['class_names'][:10]  # Limit to 10
        num_classes = 10
    else:  # FULL
        class_names = list(test_dataset.class_to_idx.keys())
        num_classes = len(class_names)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {class_names}")
    
    # Create model
    model = get_model(
        model_name='resnet50',
        num_classes=num_classes,
        pretrained=False
    )
    model = model.to(device)
    
    # Load model weights
    if os.path.exists(model_path):
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        logger.info("Model loaded successfully")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create loss function
    if use_weighted_metrics:
        # Calculate class weights from test distribution (in practice, use training dataset)
        # Get class distribution
        class_counts = {}
        for _, label in test_dataset:
            label_int = label.item() if torch.is_tensor(label) else label
            class_counts[label_int] = class_counts.get(label_int, 0) + 1
        
        # Calculate weights
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        class_weights = torch.zeros(num_classes)
        
        for class_idx, count in class_counts.items():
            class_weights[class_idx] = total_samples / (num_classes * count)
        
        class_weights = class_weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        class_weights = None
    
    # Create enhanced validator
    validator = EnhancedValidator(plant_name=plant_name, device=device)
    
    # Evaluate model
    results = validator.evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        class_names=class_names,
        save_dir=output_dir,
        scheme=scheme,
        use_weighted_metrics=use_weighted_metrics,
        class_weights=class_weights
    )
    
    logger.info(f"Evaluation completed for {scheme.value} scheme")
    return results


def evaluate_multiple_schemes(
    model_dir: str,
    dataset_root: str,
    plant_name: str,
    schemes: Optional[list] = None,
    batch_size: int = 32,
    output_base_dir: Optional[str] = None
):
    """
    Evaluate models trained with different classification schemes and compare results.
    
    Args:
        model_dir: Directory containing model checkpoints
        dataset_root: Root directory of the dataset
        plant_name: Name of the plant
        schemes: List of classification schemes to evaluate
        batch_size: Batch size for evaluation
        output_base_dir: Base directory for outputs
    
    Returns:
        Dictionary mapping scheme names to their results
    """
    if schemes is None:
        schemes = [
            ClassificationScheme.BINARY,
            ClassificationScheme.FOUR_WAY,
            ClassificationScheme.TEN_WAY
        ]
    
    # Create output directory
    if output_base_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_base_dir = f"outputs/{plant_name}_comparison_{timestamp}"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    # Evaluate each scheme
    for scheme in schemes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating {scheme.value} classification scheme")
        logger.info(f"{'='*60}")
        
        # Find model file for this scheme
        model_filename = f"{plant_name}_{scheme.value}_best_model.pth"
        model_path = os.path.join(model_dir, model_filename)
        
        # Alternative naming patterns
        if not os.path.exists(model_path):
            # Try different naming conventions
            alt_names = [
                f"{plant_name.lower()}_{scheme.value}_model.pth",
                f"best_model_{scheme.value}.pth",
                f"{scheme.value}_model.pth"
            ]
            
            for alt_name in alt_names:
                alt_path = os.path.join(model_dir, alt_name)
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
        
        if not os.path.exists(model_path):
            logger.warning(f"Model not found for {scheme.value} scheme at {model_path}")
            continue
        
        # Create scheme-specific output directory
        scheme_output_dir = os.path.join(output_base_dir, scheme.value)
        
        try:
            # Evaluate model
            results = evaluate_single_model(
                model_path=model_path,
                dataset_root=dataset_root,
                plant_name=plant_name,
                scheme=scheme,
                batch_size=batch_size,
                use_weighted_metrics=True,
                output_dir=scheme_output_dir
            )
            
            all_results[scheme.value] = results
            
        except Exception as e:
            logger.error(f"Error evaluating {scheme.value} scheme: {str(e)}")
            continue
    
    # Create comparison visualizations
    if len(all_results) > 1:
        logger.info("\n" + "="*60)
        logger.info("Generating comparison visualizations")
        logger.info("="*60)
        
        comparison_dir = os.path.join(output_base_dir, 'comparisons')
        visualizer = ComparisonVisualizer(output_dir=comparison_dir)
        
        # Generate all comparisons
        visualizer.generate_all_comparisons(
            results_dict=all_results,
            dataset_info={'plant_name': plant_name}
        )
        
        logger.info("Comparison analysis completed!")
    
    return all_results


def main():
    """Main function demonstrating the enhanced evaluation system."""
    
    # Configuration
    config = {
        'dataset_root': '../datasets',
        'model_dir': 'outputs/models',  # Directory containing trained models
        'plant_names': ['Apple', 'Maize', 'Tomato'],
        'batch_size': 32,
        'schemes': [
            ClassificationScheme.BINARY,
            ClassificationScheme.FOUR_WAY,
            ClassificationScheme.TEN_WAY
        ]
    }
    
    # Example 1: Evaluate a single model
    logger.info("\n" + "="*80)
    logger.info("Example 1: Single Model Evaluation")
    logger.info("="*80)
    
    # Assuming you have a trained model
    single_model_path = "outputs/Apple/Apple_training_20250707_193203/models/best_model.pth"
    
    if os.path.exists(single_model_path):
        results = evaluate_single_model(
            model_path=single_model_path,
            dataset_root=config['dataset_root'],
            plant_name='Apple',
            scheme=ClassificationScheme.TEN_WAY,
            batch_size=config['batch_size'],
            use_weighted_metrics=True
        )
        
        logger.info(f"Single model evaluation completed!")
        logger.info(f"Accuracy: {results['overall_metrics']['accuracy']*100:.2f}%")
        logger.info(f"F1-Score: {results['overall_metrics']['f1_score']*100:.2f}%")
    
    # Example 2: Compare multiple schemes
    logger.info("\n" + "="*80)
    logger.info("Example 2: Multi-Scheme Comparison")
    logger.info("="*80)
    
    for plant_name in config['plant_names']:
        logger.info(f"\nEvaluating {plant_name} models...")
        
        # Adjust model directory based on your structure
        plant_model_dir = f"outputs/{plant_name}/models"
        
        if os.path.exists(plant_model_dir):
            all_results = evaluate_multiple_schemes(
                model_dir=plant_model_dir,
                dataset_root=config['dataset_root'],
                plant_name=plant_name,
                schemes=config['schemes'],
                batch_size=config['batch_size']
            )
            
            # Print summary
            logger.info(f"\nSummary for {plant_name}:")
            for scheme, results in all_results.items():
                metrics = results['overall_metrics']
                logger.info(f"  {scheme}: Acc={metrics['accuracy']*100:.1f}%, "
                          f"F1={metrics['f1_score']*100:.1f}%, "
                          f"Kappa={metrics['cohen_kappa']:.3f}")
    
    # Example 3: Custom evaluation with specific requirements
    logger.info("\n" + "="*80)
    logger.info("Example 3: Custom Evaluation for Paper")
    logger.info("="*80)
    
    # Create a high-quality evaluation for academic paper
    paper_output_dir = "outputs/paper_results"
    os.makedirs(paper_output_dir, exist_ok=True)
    
    # You can customize the evaluation here
    # For example, evaluate only specific schemes or use different metrics
    
    logger.info("\nEnhanced evaluation examples completed!")
    logger.info("Check the outputs directory for detailed results and visualizations.")


if __name__ == "__main__":
    main()