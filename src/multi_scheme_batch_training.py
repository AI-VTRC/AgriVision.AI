#!/usr/bin/env python3
"""
Multi-Scheme Batch Training Script

This script trains models for all classification schemes (binary, 4-way, 10-way)
for each plant species, using the enhanced infrastructure with weighted loss functions.
It generates comprehensive comparison reports automatically.
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification_schemes import ClassificationScheme
from src.comparison_visualizer import ComparisonVisualizer


class MultiSchemeTrainer:
    """Manages training across multiple classification schemes"""
    
    def __init__(self, base_results_dir: str = "multi_scheme_results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging configuration"""
        log_file = self.base_results_dir / f"batch_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_training_command(self, plant: str, scheme: str, output_dir: Path,
                           loss_type: str = "weighted", balance_strategy: str = "inverse") -> List[str]:
        """Generate training command for a specific configuration"""
        cmd = [
            sys.executable, "src/enhanced_main.py",
            "--plant", plant,
            "--scheme", scheme,
            "--weighting_strategy", balance_strategy,
            "--batch_size", "32",
            "--epochs", "50",
            "--lr", "0.001",
            "--scheduler", "cosine",
            "--output_dir", str(output_dir)
        ]
        
        # Add focal loss parameters if using focal loss
        if loss_type == "focal":
            cmd.extend(["--focal_alpha", "0.25", "--focal_gamma", "2.0"])
            
        return cmd
        
    def train_single_configuration(self, plant: str, scheme: str, 
                                 loss_type: str = "weighted") -> Dict:
        """Train a single plant-scheme configuration"""
        # Create output directory
        output_dir = self.base_results_dir / plant / scheme / loss_type
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Training {plant} - {scheme} - {loss_type}")
        
        # Get training command
        cmd = self.get_training_command(plant, scheme, output_dir, loss_type)
        
        # Run training
        start_time = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Parse training results
            results = self.parse_training_output(result.stdout, output_dir)
            results['duration'] = duration
            results['status'] = 'success'
            
            self.logger.info(f"Completed {plant} - {scheme} - {loss_type} in {duration:.1f}s")
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed {plant} - {scheme} - {loss_type}: {e}")
            results = {
                'status': 'failed',
                'error': str(e),
                'stderr': e.stderr
            }
            
        # Save configuration results
        results_file = output_dir / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    def parse_training_output(self, output: str, output_dir: Path) -> Dict:
        """Parse training output to extract key metrics"""
        results = {}
        
        # Extract final metrics from output
        lines = output.split('\n')
        for line in lines:
            if "Best validation accuracy:" in line:
                results['best_val_acc'] = float(line.split(':')[-1].strip().rstrip('%'))
            elif "Final validation accuracy:" in line:
                results['final_val_acc'] = float(line.split(':')[-1].strip().rstrip('%'))
            elif "Best F1 score:" in line:
                results['best_f1'] = float(line.split(':')[-1].strip())
                
        # Load metrics history if available
        metrics_file = output_dir / "training_metrics.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                results['metrics_history'] = metrics
                
        # Load confusion matrix if available
        cm_file = output_dir / "confusion_matrix.npy"
        if cm_file.exists():
            results['has_confusion_matrix'] = True
            
        return results
        
    def train_all_configurations(self, plants: List[str], schemes: List[str],
                               loss_types: List[str] = ["weighted", "focal"]):
        """Train all combinations of plants, schemes, and loss types"""
        total_configs = len(plants) * len(schemes) * len(loss_types)
        self.logger.info(f"Starting batch training for {total_configs} configurations")
        
        results_summary = []
        
        for i, plant in enumerate(plants):
            for j, scheme in enumerate(schemes):
                for k, loss_type in enumerate(loss_types):
                    config_num = i * len(schemes) * len(loss_types) + j * len(loss_types) + k + 1
                    self.logger.info(f"\nConfiguration {config_num}/{total_configs}")
                    
                    # Train configuration
                    results = self.train_single_configuration(plant, scheme, loss_type)
                    
                    # Add configuration info
                    results['plant'] = plant
                    results['scheme'] = scheme
                    results['loss_type'] = loss_type
                    
                    results_summary.append(results)
                    
                    # Save intermediate summary
                    self.save_results_summary(results_summary)
                    
        # Generate final comparison reports
        self.generate_comparison_reports(results_summary)
        
        return results_summary
        
    def save_results_summary(self, results: List[Dict]):
        """Save summary of all training results"""
        summary_file = self.base_results_dir / "training_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Create CSV summary for easy viewing
        df_data = []
        for r in results:
            if r['status'] == 'success':
                df_data.append({
                    'Plant': r['plant'],
                    'Scheme': r['scheme'],
                    'Loss Type': r['loss_type'],
                    'Best Val Acc': r.get('best_val_acc', 'N/A'),
                    'Final Val Acc': r.get('final_val_acc', 'N/A'),
                    'Best F1': r.get('best_f1', 'N/A'),
                    'Duration (s)': r.get('duration', 'N/A')
                })
                
        if df_data:
            df = pd.DataFrame(df_data)
            df.to_csv(self.base_results_dir / "training_summary.csv", index=False)
            
    def generate_comparison_reports(self, results: List[Dict]):
        """Generate comprehensive comparison reports"""
        self.logger.info("Generating comparison reports...")
        
        # Group results by plant
        plants = set(r['plant'] for r in results if r['status'] == 'success')
        
        for plant in plants:
            plant_results = [r for r in results if r['plant'] == plant and r['status'] == 'success']
            
            if not plant_results:
                continue
                
            # Create plant-specific comparison
            self.create_plant_comparison(plant, plant_results)
            
        # Create overall comparison
        self.create_overall_comparison(results)
        
    def create_plant_comparison(self, plant: str, results: List[Dict]):
        """Create comparison visualizations for a specific plant"""
        output_dir = self.base_results_dir / plant / "comparisons"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for visualization
        scheme_results = {}
        for r in results:
            scheme = r['scheme']
            if scheme not in scheme_results:
                scheme_results[scheme] = {}
                
            # Load detailed results if available
            result_dir = self.base_results_dir / plant / scheme / r['loss_type']
            
            # Try to load validation results
            val_results_file = result_dir / "validation_results.json"
            if val_results_file.exists():
                with open(val_results_file, 'r') as f:
                    val_results = json.load(f)
                    scheme_results[scheme] = val_results
                    
        # Generate comparison visualizations
        if len(scheme_results) > 1:
            try:
                # Create a comparison visualizer for this plant
                plant_visualizer = ComparisonVisualizer(str(output_dir))
                plant_visualizer.generate_all_comparisons(
                    scheme_results,
                    dataset_info={'plant': plant}
                )
            except Exception as e:
                self.logger.error(f"Failed to create comparison dashboard for {plant}: {e}")
                
    def create_overall_comparison(self, results: List[Dict]):
        """Create overall comparison across all plants and schemes"""
        output_dir = self.base_results_dir / "overall_comparisons"
        output_dir.mkdir(exist_ok=True)
        
        # Create accuracy comparison heatmap
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            # Prepare data for heatmap
            plants = sorted(set(r['plant'] for r in successful_results))
            schemes = sorted(set(r['scheme'] for r in successful_results))
            
            # Create accuracy matrix
            acc_matrix = pd.DataFrame(index=plants, columns=schemes)
            
            for r in successful_results:
                if r['loss_type'] == 'weighted':  # Use weighted as default for comparison
                    acc_matrix.loc[r['plant'], r['scheme']] = r.get('best_val_acc', 0)
                    
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(acc_matrix.astype(float), annot=True, fmt='.1f', 
                       cmap='YlOrRd', cbar_kws={'label': 'Validation Accuracy (%)'})
            plt.title('Multi-Scheme Performance Comparison Across Plants')
            plt.xlabel('Classification Scheme')
            plt.ylabel('Plant Species')
            plt.tight_layout()
            plt.savefig(output_dir / "overall_accuracy_heatmap.png", dpi=150)
            plt.close()
            
        self.logger.info("Comparison reports generated successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Train models for all classification schemes across multiple plants"
    )
    
    parser.add_argument(
        "--plants",
        nargs="+",
        default=["Apple", "Maize", "Tomato"],
        help="List of plants to train"
    )
    
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["binary", "4-way", "10-way"],
        help="Classification schemes to use"
    )
    
    parser.add_argument(
        "--loss_types",
        nargs="+",
        default=["weighted"],
        choices=["weighted", "focal"],
        help="Loss types to use"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="multi_scheme_results",
        help="Base directory for results"
    )
    
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip configurations that have already been trained"
    )
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MultiSchemeTrainer(args.output_dir)
    
    # Train all configurations
    results = trainer.train_all_configurations(
        plants=args.plants,
        schemes=args.schemes,
        loss_types=args.loss_types
    )
    
    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"\nTraining completed: {successful}/{len(results)} successful")
    
    if successful < len(results):
        print("\nFailed configurations:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['plant']} / {r['scheme']} / {r['loss_type']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()