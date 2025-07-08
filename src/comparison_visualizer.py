"""
Comparison visualizer for multi-classification scheme evaluation.

This module provides visualization tools for:
- Side-by-side confusion matrix comparisons
- Performance comparison charts across classification schemes
- Class distribution visualizations
- Export functionality for academic papers
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ComparisonVisualizer:
    """Visualizer for comparing evaluation results across classification schemes."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the comparison visualizer.
        
        Args:
            output_dir: Directory to save comparison visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style for academic papers
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
        
        # Define color schemes for different classification types
        self.scheme_colors = {
            'binary': '#1f77b4',
            'four_way': '#ff7f0e', 
            '4-way': '#ff7f0e',
            'ten_way': '#2ca02c',
            '10-way': '#2ca02c',
            'full': '#d62728'
        }
    
    def compare_confusion_matrices(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        normalize: str = 'row'
    ):
        """
        Create side-by-side confusion matrix comparisons.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            normalize: Type of normalization ('row', 'column', 'all', 'none')
        """
        schemes = list(results_dict.keys())
        n_schemes = len(schemes)
        
        if n_schemes == 0:
            logger.warning("No results to compare")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(7 * n_schemes, 6))
        gs = GridSpec(1, n_schemes, figure=fig, wspace=0.3)
        
        for idx, scheme in enumerate(schemes):
            ax = fig.add_subplot(gs[0, idx])
            
            # Get confusion matrix data
            cm_data = results_dict[scheme].get('confusion_matrix', {})
            
            # Select appropriate matrix based on normalization
            if normalize == 'row' and 'row_normalized' in cm_data:
                cm = np.array(cm_data['row_normalized'])
                fmt = '.1%'
                cbar_label = 'Percentage'
            elif normalize == 'column' and 'column_normalized' in cm_data:
                cm = np.array(cm_data['column_normalized'])
                fmt = '.1%'
                cbar_label = 'Percentage'
            elif normalize == 'all' and 'total_normalized' in cm_data:
                cm = np.array(cm_data['total_normalized'])
                fmt = '.2%'
                cbar_label = 'Percentage'
            else:
                cm = np.array(cm_data.get('raw', []))
                fmt = 'd'
                cbar_label = 'Count'
            
            # Get class names
            class_names = results_dict[scheme].get('class_names', [])
            
            # Create heatmap
            sns.heatmap(
                cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, cbar_kws={'label': cbar_label},
                square=True
            )
            
            # Set title and labels
            scheme_title = self._format_scheme_name(scheme)
            ax.set_title(f'{scheme_title} Classification\n'
                        f'Accuracy: {results_dict[scheme]["overall_metrics"]["accuracy"]*100:.2f}%',
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            
            # Rotate labels if needed
            if len(class_names) > 4:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        # Add main title
        normalize_text = {
            'row': 'Row-Normalized',
            'column': 'Column-Normalized',
            'all': 'Total-Normalized',
            'none': 'Raw Counts'
        }.get(normalize, 'Row-Normalized')
        
        fig.suptitle(f'Confusion Matrix Comparison ({normalize_text})', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        # Save in multiple formats
        base_path = os.path.join(self.output_dir, f'confusion_matrix_comparison_{normalize}')
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix comparison saved to: {base_path}.png/pdf")
    
    def compare_performance_metrics(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ):
        """
        Create performance comparison charts across classification schemes.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            metrics: List of metrics to compare (default: accuracy, precision, recall, f1_score)
        """
        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
        
        schemes = list(results_dict.keys())
        
        # Extract metric values
        metric_data = {metric: [] for metric in metrics}
        
        for scheme in schemes:
            overall_metrics = results_dict[scheme].get('overall_metrics', {})
            for metric in metrics:
                value = overall_metrics.get(metric, 0)
                metric_data[metric].append(value)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 1. Bar chart comparison
        ax = axes[0]
        x = np.arange(len(schemes))
        width = 0.15
        
        for i, metric in enumerate(metrics[:4]):  # Show first 4 metrics
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, [v * 100 for v in metric_data[metric]], 
                          width, label=self._format_metric_name(metric),
                          alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Classification Scheme', fontsize=12)
        ax.set_ylabel('Performance (%)', fontsize=12)
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self._format_scheme_name(s) for s in schemes])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)
        
        # 2. Radar chart for multi-metric comparison
        ax = axes[1]
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax = plt.subplot(2, 2, 2, projection='polar')
        
        for scheme in schemes:
            values = [results_dict[scheme]['overall_metrics'].get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            color = self.scheme_colors.get(scheme, '#333333')
            ax.plot(angles, values, 'o-', linewidth=2, label=self._format_scheme_name(scheme),
                   color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self._format_metric_name(m) for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Multi-Metric Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # 3. Heatmap of all metrics
        ax = axes[2]
        
        # Create matrix of metric values
        metric_matrix = []
        for scheme in schemes:
            row = []
            for metric in metrics:
                value = results_dict[scheme]['overall_metrics'].get(metric, 0)
                row.append(value)
            metric_matrix.append(row)
        
        metric_matrix = np.array(metric_matrix)
        
        # Create heatmap
        im = ax.imshow(metric_matrix.T, aspect='auto', cmap='YlOrRd')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(schemes)))
        ax.set_yticks(np.arange(len(metrics)))
        ax.set_xticklabels([self._format_scheme_name(s) for s in schemes])
        ax.set_yticklabels([self._format_metric_name(m) for m in metrics])
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(schemes)):
                text = ax.text(j, i, f'{metric_matrix[j, i]:.3f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Score')
        
        # 4. Statistical summary
        ax = axes[3]
        ax.axis('off')
        
        # Create summary text
        summary_text = "Statistical Summary\n" + "="*40 + "\n\n"
        
        for scheme in schemes:
            metrics_dict = results_dict[scheme]['overall_metrics']
            summary_text += f"{self._format_scheme_name(scheme)}:\n"
            summary_text += f"  Accuracy: {metrics_dict['accuracy']*100:.2f}%\n"
            summary_text += f"  F1-Score: {metrics_dict['f1_score']*100:.2f}%\n"
            summary_text += f"  Cohen's Kappa: {metrics_dict.get('cohen_kappa', 0):.4f}\n"
            summary_text += f"  Samples: {metrics_dict['num_samples']}\n\n"
        
        # Find best performing scheme
        best_scheme = max(schemes, key=lambda s: results_dict[s]['overall_metrics']['accuracy'])
        summary_text += f"Best Performing: {self._format_scheme_name(best_scheme)}\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        fig.suptitle('Comprehensive Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        base_path = os.path.join(self.output_dir, 'performance_comparison')
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance comparison saved to: {base_path}.png/pdf")
    
    def visualize_class_distributions(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        dataset_info: Optional[Dict[str, Any]] = None
    ):
        """
        Visualize class distributions across different schemes.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            dataset_info: Optional dataset information including class counts
        """
        schemes = list(results_dict.keys())
        n_schemes = len(schemes)
        
        # Create figure
        fig, axes = plt.subplots(2, (n_schemes + 1) // 2, figsize=(15, 10))
        if n_schemes <= 2:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, scheme in enumerate(schemes):
            ax = axes[idx]
            
            # Get per-class metrics
            per_class_metrics = results_dict[scheme].get('per_class_metrics', [])
            
            if per_class_metrics:
                # Create DataFrame
                df = pd.DataFrame(per_class_metrics)
                
                # Create grouped bar chart
                x = np.arange(len(df))
                width = 0.25
                
                # Support distribution
                bars1 = ax.bar(x - width, df['support'], width, label='Support', alpha=0.8)
                
                # Add percentage labels
                for i, bar in enumerate(bars1):
                    height = bar.get_height()
                    pct = df.iloc[i]['support_pct']
                    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                           f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
                
                # Performance metrics (scaled for visibility)
                max_support = df['support'].max()
                scale_factor = max_support / 100
                
                bars2 = ax.bar(x, df['f1_score_pct'] * scale_factor, width, 
                              label='F1-Score (scaled)', alpha=0.8)
                bars3 = ax.bar(x + width, df['accuracy_pct'] * scale_factor, width,
                              label='Accuracy (scaled)', alpha=0.8)
                
                ax.set_xlabel('Class', fontsize=12)
                ax.set_ylabel('Count / Scaled Performance', fontsize=12)
                ax.set_title(f'{self._format_scheme_name(scheme)} Class Distribution',
                           fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(df['class'], rotation=45, ha='right')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_schemes, len(axes)):
            axes[idx].axis('off')
        
        # Add dataset imbalance analysis if available
        if dataset_info and n_schemes < len(axes):
            ax = axes[-1]
            ax.axis('on')
            
            # Calculate imbalance metrics
            imbalance_text = "Dataset Imbalance Analysis\n" + "="*30 + "\n\n"
            
            for scheme in schemes:
                per_class = results_dict[scheme].get('per_class_metrics', [])
                if per_class:
                    supports = [m['support'] for m in per_class]
                    imbalance_ratio = max(supports) / min(supports) if min(supports) > 0 else np.inf
                    
                    imbalance_text += f"{self._format_scheme_name(scheme)}:\n"
                    imbalance_text += f"  Imbalance Ratio: {imbalance_ratio:.2f}\n"
                    imbalance_text += f"  Std Dev: {np.std(supports):.2f}\n\n"
            
            ax.text(0.1, 0.9, imbalance_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.axis('off')
        
        plt.suptitle('Class Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        base_path = os.path.join(self.output_dir, 'class_distribution_analysis')
        plt.savefig(f'{base_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{base_path}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Class distribution analysis saved to: {base_path}.png/pdf")
    
    def create_comparison_report(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        report_format: str = 'all'
    ):
        """
        Create comprehensive comparison report in multiple formats.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            report_format: Format(s) to export ('json', 'csv', 'latex', 'all')
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create comparison summary
        comparison_summary = {
            'timestamp': datetime.now().isoformat(),
            'num_schemes': len(results_dict),
            'schemes': list(results_dict.keys()),
            'best_scheme_by_metric': {}
        }
        
        # Find best scheme for each metric
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
        for metric in metrics:
            best_scheme = max(
                results_dict.keys(),
                key=lambda s: results_dict[s]['overall_metrics'].get(metric, 0)
            )
            best_value = results_dict[best_scheme]['overall_metrics'].get(metric, 0)
            comparison_summary['best_scheme_by_metric'][metric] = {
                'scheme': best_scheme,
                'value': best_value
            }
        
        # Export JSON
        if report_format in ['json', 'all']:
            json_path = os.path.join(self.output_dir, f'comparison_report_{timestamp}.json')
            with open(json_path, 'w') as f:
                json.dump({
                    'summary': comparison_summary,
                    'detailed_results': results_dict
                }, f, indent=4)
            logger.info(f"JSON report saved to: {json_path}")
        
        # Export CSV
        if report_format in ['csv', 'all']:
            # Overall metrics CSV
            overall_data = []
            for scheme, results in results_dict.items():
                row = {'scheme': scheme}
                row.update(results['overall_metrics'])
                overall_data.append(row)
            
            overall_df = pd.DataFrame(overall_data)
            csv_path = os.path.join(self.output_dir, f'comparison_overall_metrics_{timestamp}.csv')
            overall_df.to_csv(csv_path, index=False)
            logger.info(f"Overall metrics CSV saved to: {csv_path}")
            
            # Per-class metrics CSV
            per_class_data = []
            for scheme, results in results_dict.items():
                for class_metric in results.get('per_class_metrics', []):
                    row = {'scheme': scheme}
                    row.update(class_metric)
                    per_class_data.append(row)
            
            if per_class_data:
                per_class_df = pd.DataFrame(per_class_data)
                csv_path = os.path.join(self.output_dir, f'comparison_per_class_metrics_{timestamp}.csv')
                per_class_df.to_csv(csv_path, index=False)
                logger.info(f"Per-class metrics CSV saved to: {csv_path}")
        
        # Export LaTeX
        if report_format in ['latex', 'all']:
            latex_path = os.path.join(self.output_dir, f'comparison_tables_{timestamp}.tex')
            self._generate_latex_comparison_tables(results_dict, latex_path)
            logger.info(f"LaTeX tables saved to: {latex_path}")
        
        # Create comprehensive PDF report
        if report_format == 'all':
            self._create_pdf_summary(results_dict, comparison_summary)
    
    def _generate_latex_comparison_tables(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        output_path: str
    ):
        """Generate LaTeX tables for academic papers."""
        with open(output_path, 'w') as f:
            f.write("% Comparison Tables for Academic Papers\n")
            f.write(f"% Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Table 1: Overall Performance Comparison
            f.write("% Table 1: Overall Performance Comparison\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Performance Comparison Across Classification Schemes}\n")
            f.write("\\label{tab:performance_comparison}\n")
            
            # Determine columns
            metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cohen_kappa']
            metric_labels = ['Acc.', 'Prec.', 'Rec.', 'F1', 'Kappa']
            
            f.write("\\begin{tabular}{l" + "c" * len(metrics) + "}\n")
            f.write("\\toprule\n")
            f.write("Scheme & " + " & ".join(metric_labels) + " \\\\\n")
            f.write("\\midrule\n")
            
            # Add data rows
            for scheme in results_dict:
                f.write(self._format_scheme_name(scheme) + " & ")
                values = []
                for metric in metrics:
                    value = results_dict[scheme]['overall_metrics'].get(metric, 0)
                    if metric == 'cohen_kappa':
                        values.append(f"{value:.3f}")
                    else:
                        values.append(f"{value*100:.1f}\\%")
                f.write(" & ".join(values) + " \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Table 2: Best performing scheme per metric
            f.write("% Table 2: Best Performing Scheme by Metric\n")
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Best Performing Classification Scheme by Metric}\n")
            f.write("\\label{tab:best_by_metric}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\toprule\n")
            f.write("Metric & Best Scheme & Value \\\\\n")
            f.write("\\midrule\n")
            
            for metric, label in zip(metrics, metric_labels):
                best_scheme = max(
                    results_dict.keys(),
                    key=lambda s: results_dict[s]['overall_metrics'].get(metric, 0)
                )
                best_value = results_dict[best_scheme]['overall_metrics'].get(metric, 0)
                
                f.write(f"{label} & {self._format_scheme_name(best_scheme)} & ")
                if metric == 'cohen_kappa':
                    f.write(f"{best_value:.3f}")
                else:
                    f.write(f"{best_value*100:.1f}\\%")
                f.write(" \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n\n")
            
            # Additional notes
            f.write("% Note: Acc. = Accuracy, Prec. = Precision, Rec. = Recall\n")
            f.write("% All metrics except Cohen's Kappa are presented as percentages\n")
    
    def _create_pdf_summary(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        comparison_summary: Dict[str, Any]
    ):
        """Create a comprehensive PDF summary (placeholder for actual implementation)."""
        # This would require additional libraries like reportlab or matplotlib.backends.backend_pdf
        # For now, we'll create a comprehensive PNG summary
        
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.2)
        
        # Title
        fig.suptitle('Comprehensive Classification Scheme Comparison Report', 
                    fontsize=20, fontweight='bold')
        
        # Add timestamp
        fig.text(0.5, 0.97, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ha='center', fontsize=12)
        
        # Summary statistics
        ax = fig.add_subplot(gs[0, :])
        ax.axis('off')
        
        summary_text = "Executive Summary\n" + "="*50 + "\n\n"
        summary_text += f"Number of Classification Schemes: {comparison_summary['num_schemes']}\n"
        summary_text += f"Schemes Evaluated: {', '.join([self._format_scheme_name(s) for s in comparison_summary['schemes']])}\n\n"
        
        summary_text += "Best Performing Scheme by Metric:\n"
        for metric, info in comparison_summary['best_scheme_by_metric'].items():
            summary_text += f"  • {self._format_metric_name(metric)}: "
            summary_text += f"{self._format_scheme_name(info['scheme'])} "
            if metric == 'cohen_kappa':
                summary_text += f"({info['value']:.3f})\n"
            else:
                summary_text += f"({info['value']*100:.1f}%)\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
               fontsize=12, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Performance metrics bar chart
        ax = fig.add_subplot(gs[1, :])
        schemes = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        x = np.arange(len(schemes))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results_dict[s]['overall_metrics'].get(metric, 0) * 100 for s in schemes]
            offset = (i - 1.5) * width
            ax.bar(x + offset, values, width, label=self._format_metric_name(metric))
        
        ax.set_xlabel('Classification Scheme', fontsize=12)
        ax.set_ylabel('Performance (%)', fontsize=12)
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([self._format_scheme_name(s) for s in schemes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add remaining visualizations...
        # This is a placeholder - in practice, you would add all the other charts
        
        plt.tight_layout()
        pdf_path = os.path.join(self.output_dir, 'comprehensive_comparison_report.png')
        plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive report saved to: {pdf_path}")
    
    def _format_scheme_name(self, scheme: str) -> str:
        """Format scheme name for display."""
        name_map = {
            'binary': 'Binary',
            'four_way': '4-Way',
            '4-way': '4-Way',
            'ten_way': '10-Way',
            '10-way': '10-Way',
            'full': 'Full'
        }
        return name_map.get(scheme.lower(), scheme.title())
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        name_map = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1_score': 'F1-Score',
            'cohen_kappa': "Cohen's Kappa",
            'matthews_correlation_coefficient': 'MCC',
            'expected_calibration_error': 'ECE',
            'maximum_calibration_error': 'MCE',
            'brier_score': 'Brier Score'
        }
        return name_map.get(metric, metric.replace('_', ' ').title())
    
    def generate_all_comparisons(
        self,
        results_dict: Dict[str, Dict[str, Any]],
        dataset_info: Optional[Dict[str, Any]] = None
    ):
        """
        Generate all comparison visualizations and reports.
        
        Args:
            results_dict: Dictionary mapping scheme names to their evaluation results
            dataset_info: Optional dataset information
        """
        logger.info("Generating comprehensive comparison analysis...")
        
        # Generate confusion matrix comparisons
        for normalize in ['row', 'none']:
            self.compare_confusion_matrices(results_dict, normalize=normalize)
        
        # Generate performance comparisons
        self.compare_performance_metrics(results_dict)
        
        # Generate class distribution analysis
        self.visualize_class_distributions(results_dict, dataset_info)
        
        # Generate comparison reports in all formats
        self.create_comparison_report(results_dict, report_format='all')
        
        logger.info("All comparison visualizations and reports generated successfully!")