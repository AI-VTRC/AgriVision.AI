import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
from typing import List, Optional, Dict, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    criterion: nn.Module,
    save_dir: str
) -> Dict[str, float]:
    """
    Evaluate the model on test data and generate comprehensive metrics.
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
        class_names: List of class names
        criterion: Loss function
        save_dir: Directory to save evaluation results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info("Starting model evaluation...")
    logger.info(f"Number of test batches: {len(test_loader)}")
    logger.info(f"Batch size: {test_loader.batch_size}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and labels
    all_predictions = []
    all_labels = []
    all_probs = []
    running_loss = 0.0
    
    # Disable gradient computation
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            running_loss += loss.item()
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    avg_loss = running_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted', zero_division=0
    )
    
    # Log overall metrics
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Loss: {avg_loss:.4f}")
    logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1_score:.4f}")
    
    # Get per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, zero_division=0
    )
    
    # Adjust class names if binary classification
    actual_num_classes = len(np.unique(all_labels))
    if actual_num_classes == 2 and len(class_names) > 2:
        # Binary classification case
        class_names_to_use = ["Healthy", "Unhealthy"]
    else:
        class_names_to_use = class_names[:actual_num_classes]
    
    # Generate classification report
    try:
        report_str = classification_report(
            all_labels, all_predictions,
            target_names=class_names_to_use,
            digits=4,
            zero_division=0
        )
        logger.info(f"\nClassification Report:\n{report_str}")
        
        report_path = os.path.join(save_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Evaluation Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Overall Metrics:\n")
            f.write(f"  Loss: {avg_loss:.4f}\n")
            f.write(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write(f"  Precision: {precision:.4f}\n")
            f.write(f"  Recall: {recall:.4f}\n")
            f.write(f"  F1-Score: {f1_score:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(str(report_str))
        logger.info(f"Classification report saved to: {report_path}")
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
    
    cm = None
    try:
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names_to_use,
            yticklabels=class_names_to_use
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        # Save confusion matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        # Save normalized confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_normalized * 100, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=class_names_to_use,
            yticklabels=class_names_to_use,
            cbar_kws={'label': 'Percentage (%)'}
        )
        plt.title('Confusion Matrix (Percentages)', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        
        cm_norm_path = os.path.join(save_dir, 'confusion_matrix_normalized.png')
        plt.savefig(cm_norm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Normalized confusion matrix saved to: {cm_norm_path}")
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {e}")
    
    # Save per-class metrics
    per_class_metrics = []
    if isinstance(precision_per_class, np.ndarray) and len(precision_per_class) > 0:
        for i, class_name in enumerate(class_names_to_use):
            if i < len(precision_per_class):
                per_class_metrics.append({
                    'class': class_name,
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]) if isinstance(recall_per_class, np.ndarray) else 0.0,
                    'f1_score': float(f1_per_class[i]) if isinstance(f1_per_class, np.ndarray) else 0.0,
                    'support': int(support_per_class[i]) if support_per_class is not None and i < len(support_per_class) else 0
                })
    
    # Plot per-class metrics
    if per_class_metrics:
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Precision and Recall
            class_labels = [m['class'] for m in per_class_metrics]
            precisions = [m['precision'] for m in per_class_metrics]
            recalls = [m['recall'] for m in per_class_metrics]
            
            x = np.arange(len(class_labels))
            width = 0.35
            
            ax1.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
            ax1.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Score')
            ax1.set_title('Precision and Recall per Class')
            ax1.set_xticks(x)
            ax1.set_xticklabels(class_labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
            
            # F1-Score
            f1_scores = [m['f1_score'] for m in per_class_metrics]
            ax2.bar(x, f1_scores, alpha=0.8, color='green')
            ax2.set_xlabel('Class')
            ax2.set_ylabel('F1-Score')
            ax2.set_title('F1-Score per Class')
            ax2.set_xticks(x)
            ax2.set_xticklabels(class_labels, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            metrics_plot_path = os.path.join(save_dir, 'per_class_metrics.png')
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Per-class metrics plot saved to: {metrics_plot_path}")
        except Exception as e:
            logger.error(f"Error generating per-class metrics plot: {e}")
    
    # Save all metrics to JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'num_samples': len(all_labels)
        },
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': cm.tolist() if cm is not None else None,
        'class_names': class_names_to_use
    }
    
    results_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    logger.info(f"Evaluation results saved to: {results_path}")
    
    # Plot confidence distribution
    try:
        plt.figure(figsize=(10, 6))
        
        # Get the confidence of the predicted class
        predicted_probs = []
        for i, pred in enumerate(all_predictions):
            predicted_probs.append(all_probs[i, pred])
        
        plt.hist(predicted_probs, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Distribution of Prediction Confidence')
        plt.grid(axis='y', alpha=0.3)
        
        conf_dist_path = os.path.join(save_dir, 'confidence_distribution.png')
        plt.savefig(conf_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Confidence distribution plot saved to: {conf_dist_path}")
    except Exception as e:
        logger.error(f"Error generating confidence distribution plot: {e}")
    
    # Create a summary report
    summary_path = os.path.join(save_dir, 'evaluation_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Model Evaluation Summary\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("Test Set Information:\n")
        f.write(f"  Total samples: {len(all_labels)}\n")
        f.write(f"  Number of classes: {actual_num_classes}\n")
        f.write(f"  Classes: {', '.join(class_names_to_use)}\n\n")
        
        f.write("Overall Performance:\n")
        f.write(f"  Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"  Loss: {avg_loss:.4f}\n")
        f.write(f"  Precision (weighted): {precision:.4f}\n")
        f.write(f"  Recall (weighted): {recall:.4f}\n")
        f.write(f"  F1-Score (weighted): {f1_score:.4f}\n\n")
        
        if per_class_metrics:
            f.write("Per-Class Performance:\n")
            for metrics in per_class_metrics:
                f.write(f"\n  {metrics['class']}:\n")
                f.write(f"    Precision: {metrics['precision']:.4f}\n")
                f.write(f"    Recall: {metrics['recall']:.4f}\n")
                f.write(f"    F1-Score: {metrics['f1_score']:.4f}\n")
                f.write(f"    Support: {metrics['support']}\n")
    
    logger.info(f"Evaluation summary saved to: {summary_path}")
    logger.info("Model evaluation completed successfully!")
    
    return results['overall_metrics']