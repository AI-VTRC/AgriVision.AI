import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

from model import get_model
from enhanced_training import EnhancedTrainer, create_enhanced_dataloaders
from enhanced_validation import EnhancedValidator
from classification_schemes import ClassificationScheme, ClassificationSchemeManager
from weighted_loss_utils import (
    WeightedCrossEntropyLoss,
    calculate_sample_weights,
    analyze_class_distribution,
    get_recommended_loss_params
)
from comparison_visualizer import ComparisonVisualizer


class WeightedLossCalculator:
    """Simple wrapper for weighted loss calculation utilities."""
    
    @staticmethod
    def calculate_weights(class_counts: List[int], strategy: str = 'inverse') -> np.ndarray:
        """Calculate class weights based on strategy."""
        counts = np.array(class_counts)
        counts = np.maximum(counts, 1)  # Avoid division by zero
        
        if strategy == 'inverse':
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(weights)
        elif strategy == 'sqrt_inverse':
            weights = 1.0 / np.sqrt(counts)
            weights = weights / weights.sum() * len(weights)
        else:  # 'none'
            weights = np.ones(len(counts))
        
        return weights / weights.mean()
    
    @staticmethod
    def calculate_effective_samples_weights(class_counts: List[int], beta: float = 0.9999) -> np.ndarray:
        """Calculate weights using effective number of samples."""
        counts = np.array(class_counts)
        counts = np.maximum(counts, 1)
        
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)
        
        return weights / weights.mean()


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None,
                 weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss
        
        return focal_loss.mean()

# Configure logging
def setup_logging(output_dir: str, plant_name: str, scheme: str) -> str:
    """Set up logging configuration with file and console output."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(output_dir, f"{plant_name}_{scheme}_training_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'training.log')
    
    # Create formatters and handlers
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    # Clear existing handlers
    root_logger.handlers = []
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_dir

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhanced Leaf Classification Training')
    
    # Basic parameters
    parser.add_argument('--plant', type=str, default='Maize', 
                        help='Plant name (e.g., Apple, Maize, Tomato)')
    parser.add_argument('--data_dir', type=str, default='./Dataset', 
                        help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                        help='Output directory')
    parser.add_argument('--model_name', type=str, default='clip',
                        help='Base model name (efficientnet_b0, resnet50, clip)')
    
    # Classification scheme selection
    parser.add_argument('--scheme', type=str, default='all',
                        choices=['binary', '4-way', '10-way', 'all'],
                        help='Classification scheme to use')
    parser.add_argument('--schemes', type=str, nargs='+',
                        help='Multiple classification schemes to train in sequence')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of workers for data loading')
    parser.add_argument('--train_ratio', type=float, default=0.7, 
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, 
                        help='Ratio of validation data')
    parser.add_argument('--random_seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    # Enhanced features
    parser.add_argument('--weighting_strategy', type=str, default='effective_samples',
                        choices=['none', 'inverse', 'sqrt_inverse', 'effective_samples', 'custom'],
                        help='Class weighting strategy for imbalanced data')
    parser.add_argument('--beta', type=float, default=0.9999,
                        help='Beta parameter for effective samples weighting')
    parser.add_argument('--focal_loss', action='store_true',
                        help='Use focal loss instead of cross-entropy')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    parser.add_argument('--focal_alpha', type=float, default=0.25,
                        help='Alpha parameter for focal loss')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler type')
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for plateau scheduler')
    
    # Additional features
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--progress_interval', type=int, default=10,
                        help='Number of batches between progress reports')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    # Evaluation options
    parser.add_argument('--eval', action='store_true', 
                        help='Evaluate the model instead of training')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to model checkpoint')
    parser.add_argument('--pretrained', type=int, default=1, 
                        help='Use pretrained weights (1=True, 0=False)')
    parser.add_argument('--percentage_matrix', action='store_true',
                        help='Generate percentage-based confusion matrix')
    
    # Comparative analysis
    parser.add_argument('--compare', action='store_true',
                        help='Compare results across different schemes')
    parser.add_argument('--comparison_dir', type=str, default=None,
                        help='Directory containing results to compare')
    
    return parser.parse_args()


def get_scheme_enum(scheme_str: str) -> ClassificationScheme:
    """Convert scheme string to ClassificationScheme enum."""
    scheme_map = {
        'binary': ClassificationScheme.BINARY,
        '4-way': ClassificationScheme.FOUR_WAY,
        '10-way': ClassificationScheme.TEN_WAY,
        'full': ClassificationScheme.FULL
    }
    return scheme_map.get(scheme_str, ClassificationScheme.FULL)


def train_single_scheme(args, scheme: ClassificationScheme, output_base_dir: str) -> Dict:
    """Train a model for a single classification scheme."""
    
    # Set up logging for this scheme
    log_dir = setup_logging(output_base_dir, args.plant, scheme.value)
    
    logger.info("="*80)
    logger.info(f"Starting training for {args.plant} with {scheme.value} classification")
    logger.info(f"Configuration: {vars(args)}")
    logger.info("="*80)
    
    # Set random seeds
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directories
    model_dir = os.path.join(log_dir, 'models')
    figures_dir = os.path.join(log_dir, 'figures')
    results_dir = os.path.join(log_dir, 'results')
    
    for dir_path in [model_dir, figures_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize classification scheme manager
    scheme_manager = ClassificationSchemeManager(args.plant)
    scheme_info = scheme_manager.get_scheme_info(scheme)
    
    logger.info(f"Classification scheme: {scheme_info['description']}")
    logger.info(f"Number of classes: {scheme_info['num_classes']}")
    
    # Prepare model type
    model_type = 'clip' if args.model_name == 'clip' or args.model_name.startswith('clip-') else None
    
    # Create enhanced dataloaders
    train_loader, val_loader, test_loader, dataset_info = create_enhanced_dataloaders(
        root_dir=args.data_dir,
        plant_name=args.plant,
        classification_scheme=scheme,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_seed,
        model_type=model_type
    )
    
    # Log dataset information
    logger.info(f"Dataset statistics: {dataset_info}")
    
    # Get actual number of classes from dataset
    num_classes = dataset_info['num_classes']
    class_names = dataset_info['class_names']
    
    logger.info(f"Actual number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Initialize model
    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=bool(args.pretrained)
    )
    model = model.to(device)
    
    # Calculate class weights if using weighted loss
    class_weights = None
    if args.weighting_strategy != 'none':
        weight_calculator = WeightedLossCalculator()
        class_counts = list(dataset_info['class_distribution'].values())
        
        if args.weighting_strategy == 'effective_samples':
            class_weights = weight_calculator.calculate_effective_samples_weights(
                class_counts, beta=args.beta
            )
        else:
            class_weights = weight_calculator.calculate_weights(
                class_counts, strategy=args.weighting_strategy
            )
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        logger.info(f"Using class weights: {class_weights.tolist()}")
    
    # Initialize loss function
    if args.focal_loss:
        criterion = FocalLoss(
            gamma=args.focal_gamma,
            alpha=args.focal_alpha,
            weight=class_weights
        )
        logger.info(f"Using Focal Loss with gamma={args.focal_gamma}, alpha={args.focal_alpha}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info("Using Cross Entropy Loss")
    
    # Initialize optimizer
    if model_type == 'clip':
        lr = args.lr * 0.1  # Smaller learning rate for fine-tuning
    else:
        lr = args.lr
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=args.patience
        )
    
    # Initialize enhanced trainer
    trainer = EnhancedTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        progress_interval=args.progress_interval,
        plant_name=args.plant,
        classification_scheme=scheme
    )
    
    logger.info("Starting model training...")
    
    # Train the model
    best_model_path = os.path.join(model_dir, f"{args.plant}_{scheme.value}_best_model.pth")
    metrics = trainer.train(
        num_epochs=args.epochs,
        save_path=best_model_path
    )
    
    # Save training metrics
    metrics_path = os.path.join(results_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Evaluate model with enhanced validator
    logger.info("\nEvaluating model after training...")
    validator = EnhancedValidator(plant_name=args.plant, device=device)
    
    eval_dir = os.path.join(results_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    eval_results = validator.evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        class_names=class_names,
        save_dir=eval_dir,
        scheme=scheme,
        use_weighted_metrics=(args.weighting_strategy != 'none'),
        class_weights=class_weights
    )
    
    # Save summary results
    summary = {
        'plant': args.plant,
        'scheme': scheme.value,
        'model': args.model_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'final_metrics': {
            'train_loss': metrics['train_loss'][-1] if metrics['train_loss'] else None,
            'val_loss': metrics['val_loss'][-1] if metrics['val_loss'] else None,
            'train_acc': metrics['train_acc'][-1] if metrics['train_acc'] else None,
            'val_acc': metrics['val_acc'][-1] if metrics['val_acc'] else None,
            'test_acc': eval_results['overall_metrics']['accuracy'] * 100,
            'test_balanced_acc': eval_results['overall_metrics'].get('balanced_accuracy',
                                                                     eval_results['overall_metrics']['accuracy']) * 100
        },
        'class_distribution': dataset_info['class_distribution'],
        'training_completed': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(results_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info("="*80)
    logger.info(f"Training completed for {scheme.value} classification!")
    logger.info(f"Test Accuracy: {eval_results['overall_metrics']['accuracy']*100:.2f}%")
    logger.info(f"Balanced Accuracy: {eval_results['overall_metrics'].get('balanced_accuracy', eval_results['overall_metrics']['accuracy'])*100:.2f}%")
    logger.info(f"All results saved to: {log_dir}")
    logger.info("="*80)
    
    return summary


def main():
    """Main function."""
    try:
        args = parse_args()
        
        # Determine which schemes to train
        schemes_to_train = []
        
        if args.schemes:
            # Multiple schemes specified
            schemes_to_train = [get_scheme_enum(s) for s in args.schemes]
        elif args.scheme == 'all':
            # Train all schemes
            schemes_to_train = [
                ClassificationScheme.BINARY,
                ClassificationScheme.FOUR_WAY,
                ClassificationScheme.TEN_WAY
            ]
        else:
            # Single scheme
            schemes_to_train = [get_scheme_enum(args.scheme)]
        
        # Base output directory
        output_base_dir = os.path.join(args.output_dir, args.plant)
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Store results for all schemes
        all_results = {}
        
        # Train each scheme
        for scheme in schemes_to_train:
            logger.info(f"\n{'='*80}")
            logger.info(f"Training {scheme.value} classification scheme")
            logger.info(f"{'='*80}\n")
            
            try:
                results = train_single_scheme(args, scheme, output_base_dir)
                all_results[scheme.value] = results
            except Exception as e:
                logger.error(f"Failed to train {scheme.value} scheme: {str(e)}", exc_info=True)
                all_results[scheme.value] = {'error': str(e)}
        
        # Save combined results
        if len(schemes_to_train) > 1:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            combined_results_path = os.path.join(
                output_base_dir, 
                f'combined_results_{timestamp}.json'
            )
            with open(combined_results_path, 'w') as f:
                json.dump(all_results, f, indent=4)
            
            logger.info(f"\nCombined results saved to: {combined_results_path}")
            
            # Generate comparison visualization if requested
            if args.compare or len(schemes_to_train) > 1:
                logger.info("\nGenerating comparison visualizations...")
                
                comparison_dir = os.path.join(output_base_dir, f'comparison_{timestamp}')
                os.makedirs(comparison_dir, exist_ok=True)
                
                # Create comparison plots
                visualizer = ComparisonVisualizer(output_dir=comparison_dir)
                
                # Generate all comparison visualizations
                visualizer.generate_all_comparisons(all_results)
                
                logger.info(f"Comparison visualizations saved to: {comparison_dir}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()