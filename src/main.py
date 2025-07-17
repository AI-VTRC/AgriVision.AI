import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import json
from typing import Dict, List, Tuple, Any, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import get_model
from training import create_dataloaders, Trainer
from validation import evaluate_model

def setup_logging(output_dir, experiment_name):
    """Set up logging configuration with file and console output."""
    log_dir = os.path.join(output_dir, f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'experiment.log')
    
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_dir

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive Plant Disease Classification Experiment')
    
    parser.add_argument('--data_dir', type=str, default='./datasets', help='Data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint to resume training from')
    
    return parser.parse_args()


def get_class_weights(dataset, device):
    """Calculate class weights for imbalanced datasets."""
    labels = [sample[1] for sample in dataset.samples]
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=labels)
    return torch.FloatTensor(class_weights).to(device)


def train_model(
    plant_name: str,
    classification_type: str,
    model_name: str,
    args: Any,
    device: torch.device,
    output_dir: str,
    resume_path: Optional[str] = None
) -> Dict[str, Any]:
    """Train a single model configuration."""
    
    logger.info(f"Training {model_name} for {plant_name} - {classification_type} classification")

    checkpoint = None
    if resume_path and os.path.exists(resume_path):
        logger.info(f"Loading checkpoint for resumption: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

    model_type = 'clip' if model_name == 'clip' or model_name.startswith('clip-') else None
    
    train_loader, val_loader, test_loader = create_dataloaders(
        root_dir=args.data_dir,
        plant_name=plant_name,
        classification_type=classification_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_seed,
        model_type=model_type
    )
    
    num_classes = len(train_loader.dataset.dataset.class_to_idx)  # type: ignore
    class_names = list(train_loader.dataset.dataset.class_to_idx.keys())  # type: ignore
    
    model = get_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=not checkpoint,
        device=device
    )
    
    class_weights = get_class_weights(train_loader.dataset.dataset, device)  # type: ignore
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if model_type == 'clip':
        lr = args.lr * 0.1
    else:
        lr = args.lr

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        plant_name=plant_name,
        classification_type=classification_type,
        model_name=model_name,
        checkpoint=checkpoint
    )
    
    model_save_path = os.path.join(
        output_dir, 
        f"{plant_name}_{classification_type}_{model_name}_best_model.pth"
    )
    
    training_metrics = trainer.train(
        num_epochs=args.epochs,
        save_path=model_save_path
    )
    
    eval_dir = os.path.join(output_dir, f"{plant_name}_{classification_type}_{model_name}_evaluation")
    evaluation_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
        criterion=criterion,
        save_dir=eval_dir
    )
    
    return {
        'plant': plant_name,
        'classification_type': classification_type,
        'model': model_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'training_metrics': training_metrics,
        'evaluation_metrics': evaluation_metrics,
        'model_path': model_save_path,
        'eval_dir': eval_dir
    }


def create_comparison_table(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive comparison table of all results."""
    
    data = []
    for result in results:
        data.append({
            'Plant': result['plant'],
            'Classification': result.get('classification_display_name', result['classification_type']),
            'Classification_Type': result['classification_type'],
            'Model': result['model'],
            'Num Classes': result['num_classes'],
            'Test Accuracy': result['evaluation_metrics']['accuracy'],
            'Test Precision': result['evaluation_metrics']['precision'],
            'Test Recall': result['evaluation_metrics']['recall'],
            'Test F1-Score': result['evaluation_metrics']['f1_score'],
            'Best Val Accuracy': max(result['training_metrics']['val_acc']) / 100.0,
            'Final Train Loss': result['training_metrics']['train_loss'][-1],
            'Final Val Loss': result['training_metrics']['val_loss'][-1],
        })
    
    df = pd.DataFrame(data)
    
    csv_path = os.path.join(output_dir, 'comprehensive_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Results table saved to: {csv_path}")
    
    summary_stats = df.groupby(['Plant', 'Classification']).agg({
        'Test Accuracy': ['mean', 'std', 'max'],
        'Test F1-Score': ['mean', 'std', 'max'],
        'Best Val Accuracy': ['mean', 'std', 'max']
    }).round(4)
    
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_stats.to_csv(summary_path)
    logger.info(f"Summary statistics saved to: {summary_path}")
    
    return df


def create_visualizations(results: List[Dict[str, Any]], df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations."""
    
    fig_dir = os.path.join(output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    accuracy_pivot = df.pivot_table(
        values='Test Accuracy', 
        index='Plant', 
        columns='Model', 
        aggfunc='mean'
    )
    sns.heatmap(accuracy_pivot, annot=True, fmt='.3f', cmap='viridis', ax=axes[0,0])
    axes[0,0].set_title('Test Accuracy by Plant and Model')
    
    f1_pivot = df.pivot_table(
        values='Test F1-Score', 
        index='Plant', 
        columns='Model', 
        aggfunc='mean'
    )
    sns.heatmap(f1_pivot, annot=True, fmt='.3f', cmap='plasma', ax=axes[0,1])
    axes[0,1].set_title('Test F1-Score by Plant and Model')
    
    class_pivot = df.pivot_table(
        values='Test Accuracy', 
        index='Classification', 
        columns='Model', 
        aggfunc='mean'
    )
    sns.heatmap(class_pivot, annot=True, fmt='.3f', cmap='coolwarm', ax=axes[0,2])
    axes[0,2].set_title('Test Accuracy by Classification Type and Model')
    
    df_melted = df.melt(
        id_vars=['Plant', 'Model', 'Classification'],
        value_vars=['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score'],
        var_name='Metric',
        value_name='Score'
    )
    
    sns.boxplot(data=df_melted, x='Model', y='Score', hue='Metric', ax=axes[1,0])
    axes[1,0].set_title('Performance Distribution by Model')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    sns.boxplot(data=df_melted, x='Plant', y='Score', hue='Metric', ax=axes[1,1])
    axes[1,1].set_title('Performance Distribution by Plant')
    
    sns.boxplot(data=df_melted, x='Classification', y='Score', hue='Metric', ax=axes[1,2])
    axes[1,2].set_title('Performance Distribution by Classification Type')
    axes[1,2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    overview_path = os.path.join(fig_dir, 'performance_overview.png')
    plt.savefig(overview_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, plant in enumerate(df['Plant'].unique()):
        plant_data = df[df['Plant'] == plant]
        
        accuracy_data = []
        labels = []
        for _, row in plant_data.iterrows():
            accuracy_data.append(row['Test Accuracy'])
            labels.append(f"{row['Classification']}\n{row['Model']}")
        
        axes[i].bar(range(len(accuracy_data)), accuracy_data, alpha=0.7)
        axes[i].set_xticks(range(len(labels)))
        axes[i].set_xticklabels(labels, rotation=45, ha='right')
        axes[i].set_ylabel('Test Accuracy')
        axes[i].set_title(f'{plant} - Test Accuracy Comparison')
        axes[i].grid(axis='y', alpha=0.3)
        
        for j, acc in enumerate(accuracy_data):
            axes[i].text(j, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plant_comparison_path = os.path.join(fig_dir, 'plant_comparison.png')
    plt.savefig(plant_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    best_results = df.loc[df.groupby(['Plant', 'Classification'])['Test Accuracy'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    classification_types = best_results['Classification'].unique()
    plants = best_results['Plant'].unique()
    
    x = np.arange(len(plants))
    width = 0.25
    
    for i, class_type in enumerate(classification_types):
        class_data = best_results[best_results['Classification'] == class_type]
        accuracies = []
        for plant in plants:
            plant_acc = class_data[class_data['Plant'] == plant]['Test Accuracy']
            accuracies.append(plant_acc.iloc[0] if len(plant_acc) > 0 else 0)
        
        ax.bar(x + i*width, accuracies, width, label=class_type, alpha=0.8)
        
        for j, acc in enumerate(accuracies):
            if acc > 0:
                ax.text(x[j] + i*width, acc + 0.01, f'{acc:.3f}', 
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Plant')
    ax.set_ylabel('Best Test Accuracy')
    ax.set_title('Best Performance Comparison by Plant and Classification Type')
    ax.set_xticks(x + width)
    ax.set_xticklabels(plants)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    best_performance_path = os.path.join(fig_dir, 'best_performance_comparison.png')
    plt.savefig(best_performance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Visualizations saved to: {fig_dir}")


def save_experiment_summary(results: List[Dict[str, Any]], output_dir: str):
    """Save comprehensive experiment summary."""
    
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'plants': list(set(r['plant'] for r in results)),
            'classification_types': list(set(r['classification_type'] for r in results)),
            'models': list(set(r['model'] for r in results))
        },
        'results': results
    }
    
    summary_path = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Experiment summary saved to: {summary_path}")


def main():
    try:
        args = parse_args()
        
        experiment_name = "comprehensive_plant_classification"
        log_dir = setup_logging(args.output_dir, experiment_name)
        
        logger.info("="*80)
        logger.info("Starting Comprehensive Plant Disease Classification Experiment")
        logger.info(f"Configuration: {vars(args)}")
        logger.info("="*80)
        
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
        
        # Enhanced device detection with MPS support for Apple Silicon
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_type = 'CUDA'
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
            device_type = 'MPS (Apple Silicon)'
        else:
            device = torch.device('cpu')
            device_type = 'CPU'
        
        logger.info(f"Using device: {device} ({device_type})")
        
        # Device-specific optimizations
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            logger.info("CUDA optimizations enabled:")
            logger.info("  - CuDNN benchmark mode enabled")
            logger.info("  - Memory cache cleared")
            
        elif torch.backends.mps.is_available():
            # Apple Silicon MPS optimizations
            logger.info("MPS (Metal Performance Shaders) acceleration enabled for Apple Silicon")
            logger.info("Optimizations:")
            logger.info("  - GPU acceleration via Metal")
            logger.info("  - Optimized for Apple M-series chips")
            
        else:
            logger.info("Using CPU - consider using a device with GPU acceleration")
        
        experiments_to_run = [
            {'plant': 'Maize', 'classification_type': 'detailed', 'model': 'efficientnet_b0'},
            {'plant': 'Maize', 'classification_type': 'detailed', 'model': 'resnet50'},
            {'plant': 'Maize', 'classification_type': 'detailed', 'model': 'clip'},
            {'plant': 'Tomato', 'classification_type': 'binary', 'model': 'clip'},
        ]

        # Mapping for classification display names
        classification_names_map = {
            'detailed': '10-way',
            'generation': '3-way',
            'binary': '2-way'
        }

        all_results = []
        total_experiments = len(experiments_to_run)
        current_experiment = 0
        
        for experiment in experiments_to_run:
            plant = experiment['plant']
            classification_type = experiment['classification_type']
            model = experiment['model']
            classification_name = classification_names_map.get(classification_type, classification_type)

            current_experiment += 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Experiment {current_experiment}/{total_experiments}")
            logger.info(f"Plant: {plant}, Classification: {classification_name} ({classification_type}), Model: {model}")
            logger.info(f"{'='*60}")

            try:
                result = train_model(
                    plant_name=plant,
                    classification_type=classification_type,
                    model_name=model,
                    args=args,
                    device=device,
                    output_dir=log_dir,
                    resume_path=None  # Not resuming for these specific runs
                )
                result['classification_display_name'] = classification_name
                all_results.append(result)
                
                logger.info(f"✓ Completed: {plant} - {classification_name} - {model}")
                logger.info(f"  Test Accuracy: {result['evaluation_metrics']['accuracy']:.4f}")
                logger.info(f"  Test F1-Score: {result['evaluation_metrics']['f1_score']:.4f}")
                
            except Exception as e:
                logger.error(f"X Failed: {plant} - {classification_name} - {model}")
                logger.error(f"  Error: {str(e)}")
                continue
        
        if all_results:
            logger.info("\n" + "="*80)
            logger.info("GENERATING COMPREHENSIVE ANALYSIS")
            logger.info("="*80)
            
            df = create_comparison_table(all_results, log_dir)
            create_visualizations(all_results, df, log_dir)
            save_experiment_summary(all_results, log_dir)
            
            logger.info("\n" + "="*80)
            logger.info("EXPERIMENT SUMMARY")
            logger.info("="*80)
            
            best_overall = df.loc[df['Test Accuracy'].idxmax()]
            logger.info(f"Best Overall Performance:")
            logger.info(f"  Plant: {best_overall['Plant']}")
            logger.info(f"  Classification: {best_overall['Classification']}")
            logger.info(f"  Model: {best_overall['Model']}")
            logger.info(f"  Test Accuracy: {best_overall['Test Accuracy']:.4f}")
            
            unique_plants = df['Plant'].unique()
            for plant in unique_plants:
                plant_df = df[df['Plant'] == plant]
                if not plant_df.empty:
                    plant_best = plant_df.loc[plant_df['Test Accuracy'].idxmax()]
                    logger.info(f"\nBest for {plant}:")
                    logger.info(f"  Classification: {plant_best['Classification']}")
                    logger.info(f"  Model: {plant_best['Model']}")
                    logger.info(f"  Test Accuracy: {plant_best['Test Accuracy']:.4f}")
                else:
                    logger.info(f"\nNo successful results for {plant}")

            unique_class_names = df['Classification'].unique()
            for class_name in unique_class_names:
                class_data = df[df['Classification'] == class_name]
                if len(class_data) > 0:
                    class_best = class_data.loc[class_data['Test Accuracy'].idxmax()]
                    logger.info(f"\nBest for {class_name} classification:")
                    logger.info(f"  Plant: {class_best['Plant']}")
                    logger.info(f"  Model: {class_best['Model']}")
                    logger.info(f"  Test Accuracy: {class_best['Test Accuracy']:.4f}")
                else:
                    logger.info(f"\nNo successful results for {class_name} classification")
            
            logger.info(f"\n✓ All results saved to: {log_dir}")
            logger.info("="*80)
            
        else:
            logger.error("No experiments completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {str(e)}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
    
