#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

This script provides standalone evaluation functionality for existing model checkpoints
with the enhanced validation infrastructure. It can:
- Evaluate models across different classification schemes
- Generate comprehensive comparison reports
- Work with both old and new model formats
- Support batch evaluation of multiple checkpoints
"""

import os
import sys
import argparse
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import get_model
from src.training import PlantDiseaseDataset
from src.classification_schemes import ClassificationScheme, ClassificationSchemeManager
from src.enhanced_validation import EnhancedValidator
from src.comparison_visualizer import ComparisonVisualizer
from torch.utils.data import DataLoader


class ComprehensiveEvaluator:
    """Manages comprehensive evaluation of trained models"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[torch.nn.Module, Dict[str, Any]]:
        """
        Load model checkpoint and extract metadata.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            
        Returns:
            Tuple of (model, metadata)
        """
        self.logger.info(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'best_val_acc': checkpoint.get('best_val_acc', checkpoint.get('val_acc', 0)),
            'plant': checkpoint.get('plant', 'unknown'),
            'model_type': checkpoint.get('model_type', 'resnet18'),
            'num_classes': checkpoint.get('num_classes', None),
            'classification_scheme': checkpoint.get('classification_scheme', 'unknown')
        }
        
        # Handle old vs new checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Old format - entire checkpoint is state dict
            state_dict = checkpoint
            # Try to infer metadata from filename
            filename = Path(checkpoint_path).stem
            if 'corn' in filename.lower():
                metadata['plant'] = 'corn'
            elif 'pepper' in filename.lower():
                metadata['plant'] = 'pepper'
            elif 'potato' in filename.lower():
                metadata['plant'] = 'potato'
            elif 'strawberry' in filename.lower():
                metadata['plant'] = 'strawberry'
            elif 'tomato' in filename.lower():
                metadata['plant'] = 'tomato'
                
        # Infer num_classes from state dict if not provided
        if metadata['num_classes'] is None:
            # Look for final layer
            for key in state_dict.keys():
                if 'fc.weight' in key or 'classifier' in key and 'weight' in key:
                    metadata['num_classes'] = state_dict[key].shape[0]
                    break
                    
        # Create model
        model = get_model(
            model_name=metadata['model_type'],
            num_classes=metadata['num_classes'] or 2  # Default to binary if unknown
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        
        return model, metadata
        
    def create_dataloader(self, plant: str, scheme: ClassificationScheme,
                         batch_size: int = 32) -> DataLoader:
        """
        Create dataloader for evaluation.
        
        Args:
            plant: Plant type
            scheme: Classification scheme
            batch_size: Batch size for evaluation
            
        Returns:
            DataLoader for validation set
        """
        # Create dataset
        # Using PlantDiseaseDataset with transforms
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = PlantDiseaseDataset(
            root_dir=f"data/{plant}/val",
            transform=transform,
            plant_name=plant
        )
        
        # Create dataloader
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return val_loader
        
    def evaluate_single_checkpoint(
        self,
        checkpoint_path: str,
        scheme: Optional[ClassificationScheme] = None,
        plant: Optional[str] = None,
        save_results: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate a single checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            scheme: Classification scheme (if None, tries to infer from checkpoint)
            plant: Plant type (if None, tries to infer from checkpoint)
            save_results: Whether to save evaluation results
            
        Returns:
            Evaluation results dictionary
        """
        # Load checkpoint
        model, metadata = self.load_checkpoint(checkpoint_path)
        
        # Use provided plant or infer from metadata
        if plant is None:
            plant = metadata['plant']
            if plant == 'unknown':
                raise ValueError("Could not determine plant type. Please specify --plant")
                
        # Determine classification scheme
        if scheme is None:
            # Try to infer from metadata or num_classes
            if metadata['classification_scheme'] != 'unknown':
                scheme = ClassificationScheme[metadata['classification_scheme'].upper()]
            elif metadata['num_classes'] == 2:
                scheme = ClassificationScheme.BINARY
            elif metadata['num_classes'] == 4:
                scheme = ClassificationScheme.FOUR_WAY
            elif metadata['num_classes'] == 10:
                scheme = ClassificationScheme.TEN_WAY
            else:
                scheme = ClassificationScheme.FULL
                
        self.logger.info(f"Evaluating {plant} model with {scheme.value if scheme else 'unknown'} classification")
        
        # Validate plant is not None
        if plant is None:
            raise ValueError("Plant type must be specified")
            
        # Create dataloader
        val_loader = self.create_dataloader(plant, scheme)
        
        # Create evaluator
        evaluator = EnhancedValidator(
            plant_name=plant,
            device=self.device
        )
        
        # Get class names based on scheme
        scheme_manager = ClassificationSchemeManager(plant)
        scheme_info = scheme_manager.get_scheme_info(scheme)
        class_names = scheme_info['class_names']
        
        # Create output directory for this evaluation
        eval_output_dir = self.output_dir / f"{plant}_{scheme.value}"
        eval_output_dir.mkdir(exist_ok=True)
        
        # Run evaluation
        results = evaluator.evaluate_model(
            model=model,
            test_loader=val_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            class_names=class_names,
            save_dir=str(eval_output_dir),
            scheme=scheme,
            use_weighted_metrics=False
        )
        
        # Add metadata to results
        results['checkpoint_info'] = {
            'path': checkpoint_path,
            'metadata': metadata
        }
        results['classification_scheme'] = scheme.value
        results['num_classes'] = len(class_names)
        
        # Save results if requested
        if save_results:
            output_path = self.output_dir / f"{plant}_{scheme.value if scheme else 'unknown'}_evaluation.json"
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            self.logger.info(f"Results saved to: {output_path}")
            
        return results
        
    def evaluate_multiple_checkpoints(
        self,
        checkpoint_paths: List[str],
        schemes: Optional[List[ClassificationScheme]] = None,
        plants: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate multiple checkpoints and generate comparisons.
        
        Args:
            checkpoint_paths: List of checkpoint paths
            schemes: List of classification schemes to evaluate (if None, infers from checkpoints)
            plants: List of plant types (if None, infers from checkpoints)
            
        Returns:
            Dictionary mapping checkpoint names to results
        """
        all_results = {}
        
        for i, checkpoint_path in enumerate(checkpoint_paths):
            self.logger.info(f"\nEvaluating checkpoint {i+1}/{len(checkpoint_paths)}")
            
            # Determine scheme and plant for this checkpoint
            scheme = schemes[i] if schemes and i < len(schemes) else None
            plant = plants[i] if plants and i < len(plants) else None
            
            try:
                results = self.evaluate_single_checkpoint(
                    checkpoint_path,
                    scheme=scheme,
                    plant=plant
                )
                
                # Create a unique key for this checkpoint
                checkpoint_key = Path(checkpoint_path).stem
                all_results[checkpoint_key] = results
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
                
        return all_results
        
    def evaluate_plant_schemes(
        self,
        plant: str,
        checkpoint_dir: str,
        schemes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all schemes for a specific plant.
        
        Args:
            plant: Plant type
            checkpoint_dir: Directory containing checkpoints
            schemes: List of schemes to evaluate (default: all schemes)
            
        Returns:
            Dictionary mapping schemes to evaluation results
        """
        if schemes is None:
            schemes = ['binary', 'generation', 'combined']
            
        results_by_scheme = {}
        checkpoint_dir_path = Path(checkpoint_dir)
        
        for scheme_name in schemes:
            # Look for checkpoint files
            possible_patterns = [
                f"{plant}_{scheme_name}_best.pth",
                f"{plant}_{scheme_name}_final.pth",
                f"best_{plant}_{scheme_name}.pth",
                f"{scheme_name}/best_model.pth",
                f"{scheme_name}/final_model.pth"
            ]
            
            checkpoint_found = False
            for pattern in possible_patterns:
                checkpoint_path = checkpoint_dir_path / pattern
                if checkpoint_path.exists():
                    self.logger.info(f"Found checkpoint: {checkpoint_path}")
                    
                    try:
                        # Map scheme names to correct enum values
                        scheme_map = {
                            'binary': 'BINARY',
                            'generation': 'FOUR_WAY',
                            'combined': 'TEN_WAY',
                            'full': 'FULL'
                        }
                        enum_name = scheme_map.get(scheme_name, scheme_name.upper())
                        scheme_enum = ClassificationScheme[enum_name]
                        results = self.evaluate_single_checkpoint(
                            str(checkpoint_path),
                            scheme=scheme_enum,
                            plant=plant
                        )
                        results_by_scheme[scheme_name] = results
                        checkpoint_found = True
                        break
                    except Exception as e:
                        self.logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
                        
            if not checkpoint_found:
                self.logger.warning(f"No checkpoint found for {plant} - {scheme_name}")
                
        return results_by_scheme
        
    def compare_schemes(
        self,
        results_by_scheme: Dict[str, Dict[str, Any]],
        plant: str
    ):
        """
        Generate comparison visualizations for different schemes.
        
        Args:
            results_by_scheme: Dictionary mapping schemes to evaluation results
            plant: Plant type for labeling
        """
        if len(results_by_scheme) < 2:
            self.logger.warning("Need at least 2 schemes for comparison")
            return
            
        # Create comparison output directory
        comparison_dir = self.output_dir / f"{plant}_comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        # Initialize visualizer
        visualizer = ComparisonVisualizer(str(comparison_dir))
        
        # Generate all comparisons
        visualizer.generate_all_comparisons(
            results_by_scheme,
            dataset_info={'plant': plant}
        )
        
        self.logger.info(f"Comparison visualizations saved to: {comparison_dir}")
        
    def batch_evaluate_directory(
        self,
        directory: str,
        recursive: bool = True,
        pattern: str = "*.pth"
    ) -> Dict[str, Any]:
        """
        Evaluate all checkpoints in a directory.
        
        Args:
            directory: Directory to search for checkpoints
            recursive: Whether to search recursively
            pattern: File pattern to match
            
        Returns:
            Summary of evaluation results
        """
        directory_path = Path(directory)
        
        # Find all checkpoint files
        if recursive:
            checkpoint_files = list(directory_path.rglob(pattern))
        else:
            checkpoint_files = list(directory_path.glob(pattern))
            
        self.logger.info(f"Found {len(checkpoint_files)} checkpoint files")
        
        # Group by plant
        plant_checkpoints = {}
        for checkpoint_file in checkpoint_files:
            # Try to determine plant from path
            path_str = str(checkpoint_file).lower()
            plant = None
            for p in ['corn', 'pepper', 'potato', 'strawberry', 'tomato']:
                if p in path_str:
                    plant = p
                    break
                    
            if plant:
                if plant not in plant_checkpoints:
                    plant_checkpoints[plant] = []
                plant_checkpoints[plant].append(checkpoint_file)
                
        # Evaluate each plant's checkpoints
        all_results = {}
        for plant, checkpoints in plant_checkpoints.items():
            self.logger.info(f"\nEvaluating {plant} checkpoints ({len(checkpoints)} files)")
            
            results = self.evaluate_multiple_checkpoints(
                [str(cp) for cp in checkpoints],
                plants=[plant] * len(checkpoints)
            )
            
            all_results[plant] = results
            
        # Generate summary report
        self.generate_batch_summary(all_results)
        
        return all_results
        
    def generate_batch_summary(self, all_results: Dict[str, Any]):
        """Generate summary report for batch evaluation"""
        summary_data = []
        
        for plant, plant_results in all_results.items():
            for checkpoint_name, results in plant_results.items():
                summary_data.append({
                    'Plant': plant,
                    'Checkpoint': checkpoint_name,
                    'Scheme': results['classification_scheme'],
                    'Accuracy': results['overall_metrics']['accuracy'] * 100,
                    'F1-Score': results['overall_metrics']['f1_score'] * 100,
                    'Cohen\'s Kappa': results['overall_metrics']['cohen_kappa'],
                    'Num Classes': results['num_classes'],
                    'Num Samples': results['overall_metrics']['num_samples']
                })
                
        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / "batch_evaluation_summary.csv"
        df.to_csv(summary_path, index=False)
        
        self.logger.info(f"Batch summary saved to: {summary_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("BATCH EVALUATION SUMMARY")
        print("="*60)
        print(f"Total checkpoints evaluated: {len(summary_data)}")
        print(f"Plants covered: {df['Plant'].unique().tolist()}")
        print(f"Average accuracy: {df['Accuracy'].mean():.2f}%")
        print(f"Best checkpoint: {df.loc[df['Accuracy'].idxmax(), 'Checkpoint']} "
              f"({df['Accuracy'].max():.2f}%)")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation script for trained models"
    )
    
    # Evaluation modes
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "multiple", "plant", "batch"],
        default="single",
        help="Evaluation mode"
    )
    
    # Single checkpoint evaluation
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (for single mode)"
    )
    
    # Multiple checkpoints
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="List of checkpoint paths (for multiple mode)"
    )
    
    # Plant-specific evaluation
    parser.add_argument(
        "--plant",
        type=str,
        choices=["corn", "pepper", "potato", "strawberry", "tomato"],
        help="Plant type"
    )
    
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing checkpoints"
    )
    
    # Scheme specification
    parser.add_argument(
        "--scheme",
        type=str,
        choices=["binary", "four_way", "ten_way", "full"],
        help="Classification scheme"
    )
    
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=["binary", "four_way", "ten_way", "full"],
        help="List of classification schemes"
    )
    
    # Batch evaluation
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory for batch evaluation"
    )
    
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively for checkpoints"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.pth",
        help="File pattern for checkpoints"
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Generate comparison visualizations"
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(args.output_dir)
    
    # Execute based on mode
    if args.mode == "single":
        if not args.checkpoint:
            parser.error("--checkpoint required for single mode")
            
        if args.scheme:
            # Map scheme names to correct enum values
            scheme_map = {
                'binary': 'BINARY',
                'four_way': 'FOUR_WAY',
                'ten_way': 'TEN_WAY',
                'full': 'FULL'
            }
            enum_name = scheme_map.get(args.scheme, args.scheme.upper())
            if enum_name:
                scheme = ClassificationScheme[enum_name]
            else:
                scheme = None
        else:
            scheme = None
        results = evaluator.evaluate_single_checkpoint(
            args.checkpoint,
            scheme=scheme,
            plant=args.plant
        )
        
        print(f"\nEvaluation complete!")
        print(f"Accuracy: {results['overall_metrics']['accuracy']*100:.2f}%")
        print(f"F1-Score: {results['overall_metrics']['f1_score']*100:.2f}%")
        
    elif args.mode == "multiple":
        if not args.checkpoints:
            parser.error("--checkpoints required for multiple mode")
            
        if args.schemes:
            schemes = []
            scheme_map = {
                'binary': 'BINARY',
                'four_way': 'FOUR_WAY',
                'ten_way': 'TEN_WAY',
                'full': 'FULL'
            }
            for s in args.schemes:
                enum_name = scheme_map.get(s, s.upper())
                if enum_name:
                    schemes.append(ClassificationScheme[enum_name])
        else:
            schemes = None
        plants = [args.plant] * len(args.checkpoints) if args.plant else None
        
        results = evaluator.evaluate_multiple_checkpoints(
            args.checkpoints,
            schemes=schemes,
            plants=plants
        )
        
        if args.compare and len(results) > 1:
            # Group by plant for comparison
            plant_results = {}
            for key, result in results.items():
                plant = result['checkpoint_info']['metadata']['plant']
                if plant not in plant_results:
                    plant_results[plant] = {}
                plant_results[plant][result['classification_scheme']] = result
                
            for plant, schemes_results in plant_results.items():
                evaluator.compare_schemes(schemes_results, plant)
                
    elif args.mode == "plant":
        if not args.plant or not args.checkpoint_dir:
            parser.error("--plant and --checkpoint_dir required for plant mode")
            
        results = evaluator.evaluate_plant_schemes(
            args.plant,
            args.checkpoint_dir,
            args.schemes
        )
        
        if args.compare and len(results) > 1:
            evaluator.compare_schemes(results, args.plant)
            
    elif args.mode == "batch":
        if not args.directory:
            parser.error("--directory required for batch mode")
            
        results = evaluator.batch_evaluate_directory(
            args.directory,
            recursive=args.recursive,
            pattern=args.pattern
        )
        
    print(f"\nAll results saved to: {evaluator.output_dir}")


if __name__ == "__main__":
    main()