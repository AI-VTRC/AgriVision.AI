"""
Enhanced validation module with percentage-based confusion matrices and multi-classification scheme support.

This module extends the existing validation functionality to provide:
- Percentage-based (row-normalized) confusion matrices
- Multi-classification scheme evaluation
- Per-class metrics with percentage display
- Support for weighted metrics with class balancing
- Comprehensive comparison capabilities
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import pandas as pd
from typing import List, Optional, Dict, Tuple, Any
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from classification_schemes import ClassificationScheme, ClassificationSchemeManager

logger = logging.getLogger(__name__)


class EnhancedValidator:
    """Enhanced validator for comprehensive model evaluation across classification schemes."""
    
    def __init__(self, plant_name: str, device: torch.device):
        """
        Initialize the enhanced validator.
        
        Args:
            plant_name: Name of the plant (e.g., 'Apple', 'Maize', 'Tomato')
            device: Device to run evaluation on
        """
        self.plant_name = plant_name
        self.device = device
        self.scheme_manager = ClassificationSchemeManager(plant_name)
        self.results_cache = {}
    
    def evaluate_model(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
        class_names: List[str],
        save_dir: str,
        scheme: ClassificationScheme,
        use_weighted_metrics: bool = False,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model with enhanced metrics and visualizations.
        
        Args:
            model: The model to evaluate
            test_loader: Test data loader
            criterion: Loss function
            class_names: List of class names
            save_dir: Directory to save evaluation results
            scheme: Classification scheme being evaluated
            use_weighted_metrics: Whether to use weighted metrics
            class_weights: Optional class weights for weighted metrics
            
        Returns:
            Dictionary containing comprehensive evaluation results
        """
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Starting enhanced evaluation for scheme: {scheme.value}")
        logger.info(f"Number of test batches: {len(test_loader)}")
        logger.info(f"Using weighted metrics: {use_weighted_metrics}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Initialize tracking variables
        all_predictions = []
        all_labels = []
        all_probs = []
        running_loss = 0.0
        
        # Disable gradient computation
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(test_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
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
        
        # Calculate basic metrics
        avg_loss = running_loss / len(test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate weighted or macro metrics based on configuration
        average_type = 'weighted' if use_weighted_metrics else 'macro'
        precision, recall, f1_score, support = precision_recall_fscore_support(
            all_labels, all_predictions, average=average_type, zero_division=0
        )
        
        # Calculate Cohen's Kappa
        kappa = cohen_kappa_score(all_labels, all_predictions)
        
        # Log overall metrics
        logger.info(f"\nOverall Metrics ({average_type} average):")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1_score:.4f}")
        logger.info(f"  Cohen's Kappa: {kappa:.4f}")
        
        # Get per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(
            all_labels, all_predictions, all_probs, class_names
        )
        
        # Generate enhanced confusion matrices
        cm_results = self._generate_enhanced_confusion_matrices(
            all_labels, all_predictions, class_names, save_dir
        )
        
        # Generate classification report
        report_results = self._generate_enhanced_classification_report(
            all_labels, all_predictions, class_names, save_dir, scheme
        )
        
        # Generate per-class visualizations
        self._generate_per_class_visualizations(
            per_class_metrics, save_dir, use_weighted_metrics
        )
        
        # Generate confidence analysis
        self._analyze_prediction_confidence(
            all_predictions, all_probs, all_labels, class_names, save_dir
        )
        
        # Calculate statistical metrics
        statistical_metrics = self._calculate_statistical_metrics(
            all_labels, all_predictions, all_probs
        )
        
        # Compile comprehensive results
        results = {
            'timestamp': datetime.now().isoformat(),
            'scheme': scheme.value,
            'overall_metrics': {
                'loss': float(avg_loss),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'cohen_kappa': float(kappa),
                'num_samples': len(all_labels),
                'average_type': average_type
            },
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': cm_results,
            'statistical_metrics': statistical_metrics,
            'class_names': class_names,
            'use_weighted_metrics': use_weighted_metrics
        }
        
        # Save comprehensive results
        results_path = os.path.join(save_dir, 'enhanced_evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Enhanced evaluation results saved to: {results_path}")
        
        # Generate LaTeX tables
        self._generate_latex_tables(results, save_dir)
        
        # Cache results for comparison
        self.results_cache[scheme.value] = results
        
        return results
    
    def _calculate_per_class_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Calculate detailed per-class metrics including percentages."""
        # Get per-class metrics (always returns arrays when average=None)
        metrics_result = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        # Unpack results
        precision_per_class = metrics_result[0]
        recall_per_class = metrics_result[1]
        f1_per_class = metrics_result[2]
        support_per_class = metrics_result[3]
        
        # Calculate class-wise accuracy
        class_accuracies = []
        for i in range(len(class_names)):
            class_mask = labels == i
            if np.any(class_mask):
                class_acc = accuracy_score(labels[class_mask], predictions[class_mask])
                class_accuracies.append(class_acc)
            else:
                class_accuracies.append(0.0)
        
        # Calculate AUC for each class (one-vs-rest)
        class_aucs = []
        if len(class_names) > 2:
            for i in range(len(class_names)):
                try:
                    class_probs = probabilities[:, i]
                    class_labels = (labels == i).astype(int)
                    auc = roc_auc_score(class_labels, class_probs)
                    class_aucs.append(auc)
                except:
                    class_aucs.append(0.0)
        else:
            # Binary classification
            try:
                auc = roc_auc_score(labels, probabilities[:, 1])
                class_aucs = [auc, auc]
            except:
                class_aucs = [0.0, 0.0]
        
        # Compile per-class metrics
        per_class_metrics = []
        total_support = np.sum(support_per_class) if support_per_class is not None else 0
        
        # Ensure we have arrays
        if isinstance(precision_per_class, np.ndarray):
            num_classes = len(precision_per_class)
        else:
            num_classes = 0
            
        for i, class_name in enumerate(class_names):
            if i < num_classes:
                metrics = {
                    'class': class_name,
                    'precision': float(precision_per_class[i]) if isinstance(precision_per_class, np.ndarray) else 0.0,
                    'precision_pct': float(precision_per_class[i] * 100) if isinstance(precision_per_class, np.ndarray) else 0.0,
                    'recall': float(recall_per_class[i]) if isinstance(recall_per_class, np.ndarray) else 0.0,
                    'recall_pct': float(recall_per_class[i] * 100) if isinstance(recall_per_class, np.ndarray) else 0.0,
                    'f1_score': float(f1_per_class[i]) if isinstance(f1_per_class, np.ndarray) else 0.0,
                    'f1_score_pct': float(f1_per_class[i] * 100) if isinstance(f1_per_class, np.ndarray) else 0.0,
                    'accuracy': float(class_accuracies[i]) if i < len(class_accuracies) else 0.0,
                    'accuracy_pct': float(class_accuracies[i] * 100) if i < len(class_accuracies) else 0.0,
                    'support': int(support_per_class[i]) if isinstance(support_per_class, np.ndarray) else 0,
                    'support_pct': float(support_per_class[i] / total_support * 100) if isinstance(support_per_class, np.ndarray) and total_support > 0 else 0.0,
                    'auc': float(class_aucs[i]) if i < len(class_aucs) else 0.0
                }
                per_class_metrics.append(metrics)
        
        return per_class_metrics
    
    def _generate_enhanced_confusion_matrices(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str],
        save_dir: str
    ) -> Dict[str, Any]:
        """Generate enhanced confusion matrices with percentage displays."""
        cm = confusion_matrix(labels, predictions)
        
        # Create figure with subplots for different normalizations
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        
        # 1. Raw counts
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 0], cbar_kws={'label': 'Count'}
        )
        axes[0, 0].set_title('Confusion Matrix - Raw Counts', fontsize=14)
        axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
        axes[0, 0].set_ylabel('True Label', fontsize=12)
        
        # 2. Row-normalized (percentage by true class)
        cm_row_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_row_norm, annot=True, fmt='.1%', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0, 1], cbar_kws={'label': 'Percentage'}
        )
        axes[0, 1].set_title('Confusion Matrix - Row Normalized (%)', fontsize=14)
        axes[0, 1].set_xlabel('Predicted Label', fontsize=12)
        axes[0, 1].set_ylabel('True Label', fontsize=12)
        
        # 3. Column-normalized (percentage by predicted class)
        cm_col_norm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        sns.heatmap(
            cm_col_norm, annot=True, fmt='.1%', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1, 0], cbar_kws={'label': 'Percentage'}
        )
        axes[1, 0].set_title('Confusion Matrix - Column Normalized (%)', fontsize=14)
        axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
        axes[1, 0].set_ylabel('True Label', fontsize=12)
        
        # 4. Total-normalized (percentage of all predictions)
        cm_total_norm = cm.astype('float') / cm.sum()
        sns.heatmap(
            cm_total_norm, annot=True, fmt='.2%', cmap='Oranges',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1, 1], cbar_kws={'label': 'Percentage'}
        )
        axes[1, 1].set_title('Confusion Matrix - Total Normalized (%)', fontsize=14)
        axes[1, 1].set_xlabel('Predicted Label', fontsize=12)
        axes[1, 1].set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        cm_path = os.path.join(save_dir, 'enhanced_confusion_matrices.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.savefig(cm_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced confusion matrices saved to: {cm_path}")
        
        # Return confusion matrix data
        return {
            'raw': cm.tolist(),
            'row_normalized': cm_row_norm.tolist(),
            'column_normalized': cm_col_norm.tolist(),
            'total_normalized': cm_total_norm.tolist()
        }
    
    def _generate_enhanced_classification_report(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        class_names: List[str],
        save_dir: str,
        scheme: ClassificationScheme
    ) -> Dict[str, Any]:
        """Generate enhanced classification report with additional metrics."""
        # Generate detailed classification report as dictionary
        report_dict = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=True,
            digits=4,
            zero_division=0
        )
        
        # Generate text report separately
        report_text = classification_report(
            labels, predictions,
            target_names=class_names,
            output_dict=False,  # Explicitly request text format
            digits=4,
            zero_division=0
        )
        
        # Save text report
        report_path = os.path.join(save_dir, 'enhanced_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Enhanced Classification Report\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Classification Scheme: {scheme.value}\n")
            f.write("="*80 + "\n\n")
            f.write(str(report_text))  # Ensure it's converted to string
            f.write("\n\n" + "="*80 + "\n")
            f.write("Additional Metrics:\n")
            f.write(f"  Cohen's Kappa: {cohen_kappa_score(labels, predictions):.4f}\n")
            
            # Add per-class percentage metrics from the dictionary
            f.write("\nPer-Class Performance (Percentage):\n")
            for class_name in class_names:
                if isinstance(report_dict, dict) and class_name in report_dict:
                    metrics = report_dict[class_name]
                    if isinstance(metrics, dict):  # Ensure metrics is a dictionary
                        f.write(f"\n  {class_name}:\n")
                        f.write(f"    Precision: {metrics.get('precision', 0)*100:.2f}%\n")
                        f.write(f"    Recall: {metrics.get('recall', 0)*100:.2f}%\n")
                        f.write(f"    F1-Score: {metrics.get('f1-score', 0)*100:.2f}%\n")
                        f.write(f"    Support: {int(metrics.get('support', 0))}\n")
        
        logger.info(f"Enhanced classification report saved to: {report_path}")
        
        # Ensure we return a dictionary
        if isinstance(report_dict, dict):
            return report_dict
        else:
            return {}
    
    def _generate_per_class_visualizations(
        self,
        per_class_metrics: List[Dict[str, Any]],
        save_dir: str,
        use_weighted: bool
    ):
        """Generate comprehensive per-class metric visualizations."""
        if not per_class_metrics:
            return
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(per_class_metrics)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # 1. Precision, Recall, F1 comparison
        metrics = ['precision_pct', 'recall_pct', 'f1_score_pct']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            axes[0, 0].bar(
                x + i * width, df[metric], width,
                label=metric.replace('_pct', '').replace('_', ' ').title(),
                alpha=0.8
            )
        
        axes[0, 0].set_xlabel('Class')
        axes[0, 0].set_ylabel('Performance (%)')
        axes[0, 0].set_title('Precision, Recall, and F1-Score by Class')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(df['class'], rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_ylim(0, 105)
        
        # 2. Class accuracy
        axes[0, 1].bar(x, df['accuracy_pct'], alpha=0.8, color='green')
        axes[0, 1].set_xlabel('Class')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].set_title('Per-Class Accuracy')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(df['class'], rotation=45, ha='right')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].set_ylim(0, 105)
        
        # 3. Support distribution
        # Use matplotlib's built-in color cycle
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(df)))
        axes[1, 0].pie(
            df['support'], labels=df['class'], autopct='%1.1f%%',
            startangle=90, colors=colors
        )
        axes[1, 0].set_title('Class Distribution in Test Set')
        
        # 4. AUC scores (if available)
        if 'auc' in df.columns and df['auc'].sum() > 0:
            axes[1, 1].bar(x, df['auc'], alpha=0.8, color='purple')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('AUC Score')
            axes[1, 1].set_title('Per-Class AUC Scores')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(df['class'], rotation=45, ha='right')
            axes[1, 1].grid(axis='y', alpha=0.3)
            axes[1, 1].set_ylim(0, 1.05)
        else:
            axes[1, 1].text(
                0.5, 0.5, 'AUC not applicable\nfor this classification scheme',
                ha='center', va='center', transform=axes[1, 1].transAxes
            )
            axes[1, 1].axis('off')
        
        plt.suptitle(
            f'Per-Class Performance Metrics {"(Weighted)" if use_weighted else "(Macro)"}',
            fontsize=16
        )
        plt.tight_layout()
        
        metrics_path = os.path.join(save_dir, 'per_class_metrics_enhanced.png')
        plt.savefig(metrics_path, dpi=300, bbox_inches='tight')
        plt.savefig(metrics_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced per-class metrics saved to: {metrics_path}")
    
    def _analyze_prediction_confidence(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        class_names: List[str],
        save_dir: str
    ):
        """Analyze and visualize prediction confidence distributions."""
        # Get confidence of predicted class
        predicted_probs = []
        for i, pred in enumerate(predictions):
            predicted_probs.append(probabilities[i, pred])
        predicted_probs = np.array(predicted_probs)
        
        # Separate correct and incorrect predictions
        correct_mask = predictions == labels
        correct_probs = predicted_probs[correct_mask]
        incorrect_probs = predicted_probs[~correct_mask]
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Overall confidence distribution
        axes[0, 0].hist(predicted_probs, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(predicted_probs.mean(), color='red', linestyle='--', 
                          label=f'Mean: {predicted_probs.mean():.3f}')
        axes[0, 0].set_xlabel('Prediction Confidence')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Distribution of Prediction Confidence')
        axes[0, 0].legend()
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Confidence comparison: correct vs incorrect
        axes[0, 1].hist(correct_probs, bins=30, alpha=0.5, label='Correct', 
                       color='green', density=True)
        axes[0, 1].hist(incorrect_probs, bins=30, alpha=0.5, label='Incorrect', 
                       color='red', density=True)
        axes[0, 1].set_xlabel('Prediction Confidence')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0, 1].legend()
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Per-class average confidence
        class_confidences = {}
        for i, class_name in enumerate(class_names):
            class_mask = predictions == i
            if np.any(class_mask):
                class_conf = predicted_probs[class_mask].mean()
                class_confidences[class_name] = class_conf
        
        if class_confidences:
            classes = list(class_confidences.keys())
            confidences = list(class_confidences.values())
            x = np.arange(len(classes))
            
            axes[1, 0].bar(x, confidences, alpha=0.8, color='blue')
            axes[1, 0].set_xlabel('Class')
            axes[1, 0].set_ylabel('Average Confidence')
            axes[1, 0].set_title('Average Prediction Confidence by Class')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(classes, rotation=45, ha='right')
            axes[1, 0].grid(axis='y', alpha=0.3)
            axes[1, 0].set_ylim(0, 1.05)
        
        # 4. Confidence vs accuracy relationship
        confidence_bins = np.linspace(0, 1, 11)
        bin_accuracies = []
        bin_counts = []
        
        for i in range(len(confidence_bins) - 1):
            bin_mask = (predicted_probs >= confidence_bins[i]) & \
                      (predicted_probs < confidence_bins[i + 1])
            if np.any(bin_mask):
                bin_acc = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_mask.sum())
            else:
                bin_accuracies.append(0)
                bin_counts.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', markersize=8, linewidth=2)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('Calibration Plot: Confidence vs Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        conf_path = os.path.join(save_dir, 'confidence_analysis.png')
        plt.savefig(conf_path, dpi=300, bbox_inches='tight')
        plt.savefig(conf_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confidence analysis saved to: {conf_path}")
    
    def _calculate_statistical_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, float]:
        """Calculate additional statistical metrics for evaluation."""
        # Matthews Correlation Coefficient
        from sklearn.metrics import matthews_corrcoef
        mcc = matthews_corrcoef(labels, predictions)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(predictions, probabilities, labels)
        
        # Maximum Calibration Error (MCE)
        mce = self._calculate_mce(predictions, probabilities, labels)
        
        # Brier Score
        from sklearn.metrics import brier_score_loss
        # For multi-class, calculate average Brier score
        brier_scores = []
        for i in range(probabilities.shape[1]):
            class_labels = (labels == i).astype(int)
            class_probs = probabilities[:, i]
            if len(np.unique(class_labels)) > 1:  # Skip if only one class
                brier = brier_score_loss(class_labels, class_probs)
                brier_scores.append(brier)
        
        avg_brier = np.mean(brier_scores) if brier_scores else 0.0
        
        return {
            'matthews_correlation_coefficient': float(mcc),
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
            'brier_score': float(avg_brier)
        }
    
    def _calculate_ece(self, predictions, probabilities, labels, n_bins=10):
        """Calculate Expected Calibration Error."""
        predicted_probs = []
        for i, pred in enumerate(predictions):
            predicted_probs.append(probabilities[i, pred])
        predicted_probs = np.array(predicted_probs)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            bin_mask = (predicted_probs > bin_boundaries[i]) & \
                      (predicted_probs <= bin_boundaries[i + 1])
            
            if np.any(bin_mask):
                bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_confidence = predicted_probs[bin_mask].mean()
                bin_size = bin_mask.sum()
                
                ece += (bin_size / len(predictions)) * \
                       abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def _calculate_mce(self, predictions, probabilities, labels, n_bins=10):
        """Calculate Maximum Calibration Error."""
        predicted_probs = []
        for i, pred in enumerate(predictions):
            predicted_probs.append(probabilities[i, pred])
        predicted_probs = np.array(predicted_probs)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        max_error = 0
        
        for i in range(n_bins):
            bin_mask = (predicted_probs > bin_boundaries[i]) & \
                      (predicted_probs <= bin_boundaries[i + 1])
            
            if np.any(bin_mask):
                bin_accuracy = (predictions[bin_mask] == labels[bin_mask]).mean()
                bin_confidence = predicted_probs[bin_mask].mean()
                
                error = abs(bin_accuracy - bin_confidence)
                max_error = max(max_error, error)
        
        return max_error
    
    def _generate_latex_tables(self, results: Dict[str, Any], save_dir: str):
        """Generate LaTeX tables for academic papers."""
        # Overall metrics table
        latex_path = os.path.join(save_dir, 'latex_tables.tex')
        
        with open(latex_path, 'w') as f:
            f.write("% Enhanced Evaluation Results - LaTeX Tables\n")
            f.write(f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"% Classification Scheme: {results['scheme']}\n\n")
            
            # Overall metrics table
            f.write("% Table 1: Overall Performance Metrics\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Overall Performance Metrics}\n")
            f.write("\\begin{tabular}{lc}\n")
            f.write("\\hline\n")
            f.write("Metric & Value \\\\\n")
            f.write("\\hline\n")
            
            metrics = results['overall_metrics']
            f.write(f"Accuracy & {metrics['accuracy']*100:.2f}\\% \\\\\n")
            f.write(f"Precision & {metrics['precision']*100:.2f}\\% \\\\\n")
            f.write(f"Recall & {metrics['recall']*100:.2f}\\% \\\\\n")
            f.write(f"F1-Score & {metrics['f1_score']*100:.2f}\\% \\\\\n")
            f.write(f"Cohen's Kappa & {metrics['cohen_kappa']:.4f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Per-class metrics table
            if results['per_class_metrics']:
                f.write("% Table 2: Per-Class Performance Metrics\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Per-Class Performance Metrics}\n")
                f.write("\\begin{tabular}{lcccc}\n")
                f.write("\\hline\n")
                f.write("Class & Precision (\\%) & Recall (\\%) & F1-Score (\\%) & Support \\\\\n")
                f.write("\\hline\n")
                
                for metric in results['per_class_metrics']:
                    f.write(f"{metric['class']} & ")
                    f.write(f"{metric['precision_pct']:.2f} & ")
                    f.write(f"{metric['recall_pct']:.2f} & ")
                    f.write(f"{metric['f1_score_pct']:.2f} & ")
                    f.write(f"{metric['support']} \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n\n")
            
            # Confusion matrix table (percentage)
            if 'row_normalized' in results['confusion_matrix']:
                f.write("% Table 3: Confusion Matrix (Row-Normalized Percentages)\n")
                f.write("\\begin{table}[h]\n")
                f.write("\\centering\n")
                f.write("\\caption{Confusion Matrix (\\%)}\n")
                f.write("\\begin{tabular}{l|" + "c" * len(results['class_names']) + "}\n")
                f.write("\\hline\n")
                f.write("True\\textbackslash Pred & " + " & ".join(results['class_names']) + " \\\\\n")
                f.write("\\hline\n")
                
                cm = np.array(results['confusion_matrix']['row_normalized'])
                for i, class_name in enumerate(results['class_names']):
                    f.write(f"{class_name} & ")
                    row_values = [f"{val*100:.1f}" for val in cm[i]]
                    f.write(" & ".join(row_values) + " \\\\\n")
                
                f.write("\\hline\n")
                f.write("\\end{tabular}\n")
                f.write("\\end{table}\n")
        
        logger.info(f"LaTeX tables saved to: {latex_path}")
    
    def compare_schemes(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        save_dir: str
    ) -> Dict[str, Any]:
        """
        Compare evaluation results across different classification schemes.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            save_dir: Directory to save comparison results
            
        Returns:
            Dictionary containing comparison results
        """
        comparison_results = {}
        
        # Extract metrics for comparison
        schemes = list(results_dict.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
        
        # Compile metrics
        for metric in metrics_names:
            comparison_results[metric] = {
                scheme: results_dict[scheme]['overall_metrics'][metric] 
                for scheme in schemes
            }
        
        # Perform statistical tests
        if len(schemes) > 1:
            comparison_results['statistical_tests'] = self._perform_statistical_tests(
                results_dict
            )
        
        # Save comparison results
        comparison_path = os.path.join(save_dir, 'scheme_comparison_results.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=4)
        
        logger.info(f"Scheme comparison results saved to: {comparison_path}")
        
        return comparison_results
    
    def _perform_statistical_tests(
        self,
        results_dict: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests between schemes."""
        # This is a placeholder for actual statistical tests
        # In practice, you would need the raw predictions to perform proper tests
        tests = {
            'note': 'Statistical significance tests require raw predictions',
            'recommendations': [
                'McNemar test for pairwise comparisons',
                'Friedman test for multiple comparisons',
                'Post-hoc analysis with Bonferroni correction'
            ]
        }
        
        return tests