"""
Weighted loss functions and utilities for handling class imbalance in plant disease classification.

This module provides:
- Weighted CrossEntropyLoss with automatic weight calculation
- Different weighting strategies (inverse frequency, balanced, custom)
- Class distribution analysis utilities
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class WeightingStrategy:
    """Enumeration of available weighting strategies."""
    INVERSE_FREQUENCY = "inverse_frequency"
    BALANCED = "balanced"
    EFFECTIVE_NUMBER = "effective_number"
    CUSTOM = "custom"


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted CrossEntropyLoss with automatic weight calculation based on class distribution.
    """
    
    def __init__(
        self,
        class_counts: Optional[Dict[int, int]] = None,
        strategy: str = WeightingStrategy.BALANCED,
        beta: float = 0.999,
        custom_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the weighted loss function.
        
        Args:
            class_counts: Dictionary mapping class indices to their counts
            strategy: Weighting strategy to use
            beta: Beta parameter for effective number weighting (default: 0.999)
            custom_weights: Custom weight tensor (used when strategy='custom')
            device: Device to place weights on
        """
        super().__init__()
        self.strategy = strategy
        self.beta = beta
        self.device = device or torch.device('cpu')
        
        if strategy == WeightingStrategy.CUSTOM and custom_weights is not None:
            self.weights = custom_weights.to(self.device)
        elif class_counts is not None:
            self.weights = self._calculate_weights(class_counts)
        else:
            self.weights = None
        
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)
    
    def _calculate_weights(self, class_counts: Dict[int, int]) -> torch.Tensor:
        """
        Calculate class weights based on the specified strategy.
        
        Args:
            class_counts: Dictionary mapping class indices to their counts
            
        Returns:
            Tensor of class weights
        """
        # Ensure continuous indices starting from 0
        num_classes = max(class_counts.keys()) + 1
        counts = np.zeros(num_classes)
        
        for idx, count in class_counts.items():
            counts[idx] = count
        
        # Avoid division by zero
        counts = np.maximum(counts, 1)
        
        if self.strategy == WeightingStrategy.INVERSE_FREQUENCY:
            # Inverse frequency weighting
            weights = 1.0 / counts
            weights = weights / weights.sum() * len(weights)
            
        elif self.strategy == WeightingStrategy.BALANCED:
            # Balanced weighting (sklearn-style)
            total_samples = counts.sum()
            weights = total_samples / (len(counts) * counts)
            
        elif self.strategy == WeightingStrategy.EFFECTIVE_NUMBER:
            # Effective number weighting (from "Class-Balanced Loss" paper)
            effective_num = 1.0 - np.power(self.beta, counts)
            weights = (1.0 - self.beta) / effective_num
            weights = weights / weights.sum() * len(weights)
            
        else:
            # Default to uniform weights
            weights = np.ones(num_classes)
        
        # Normalize weights
        weights = weights / weights.mean()
        
        logger.info(f"Calculated class weights using {self.strategy} strategy:")
        for i, w in enumerate(weights):
            if counts[i] > 0:
                logger.info(f"  Class {i}: count={int(counts[i])}, weight={w:.4f}")
        
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of the loss function."""
        return self.criterion(input, target)
    
    def update_weights(self, class_counts: Dict[int, int]):
        """
        Update the class weights based on new class distribution.
        
        Args:
            class_counts: Updated class counts
        """
        self.weights = self._calculate_weights(class_counts)
        self.criterion = nn.CrossEntropyLoss(weight=self.weights)


def analyze_class_distribution(
    labels: List[int],
    num_classes: Optional[int] = None,
    return_percentages: bool = True
) -> Dict[str, Union[Dict[int, int], Dict[int, float]]]:
    """
    Analyze the distribution of class labels.
    
    Args:
        labels: List of class labels
        num_classes: Total number of classes (if None, inferred from labels)
        return_percentages: Whether to return percentages along with counts
        
    Returns:
        Dictionary containing class counts and optionally percentages
    """
    # Count occurrences
    counter = Counter(labels)
    
    # Determine number of classes
    if num_classes is None:
        num_classes = max(labels) + 1 if labels else 0
    
    # Create full distribution including zero counts
    counts = {}
    for i in range(num_classes):
        counts[i] = counter.get(i, 0)
    
    result: Dict[str, Any] = {"counts": counts}
    
    if return_percentages and labels:
        total = len(labels)
        percentages = {i: (count / total) * 100 for i, count in counts.items()}
        result["percentages"] = percentages
        
        # Log distribution
        logger.info("Class distribution analysis:")
        for i in range(num_classes):
            logger.info(f"  Class {i}: {counts[i]} samples ({percentages[i]:.2f}%)")
    
    # Calculate imbalance ratio
    if counts:
        non_zero_counts = [c for c in counts.values() if c > 0]
        if non_zero_counts:
            max_count = max(non_zero_counts)
            min_count = min(non_zero_counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            result["imbalance_ratio"] = imbalance_ratio
            logger.info(f"Class imbalance ratio: {imbalance_ratio:.2f}")
    
    return result


def calculate_sample_weights(
    labels: List[int],
    strategy: str = WeightingStrategy.BALANCED,
    beta: float = 0.999
) -> List[float]:
    """
    Calculate per-sample weights for weighted sampling.
    
    Args:
        labels: List of class labels
        strategy: Weighting strategy to use
        beta: Beta parameter for effective number weighting
        
    Returns:
        List of sample weights
    """
    # Get class distribution
    counter = Counter(labels)
    num_classes = max(labels) + 1 if labels else 0
    
    # Create weight calculator
    class_counts = {i: counter.get(i, 0) for i in range(num_classes)}
    weight_calc = WeightedCrossEntropyLoss(
        class_counts=class_counts,
        strategy=strategy,
        beta=beta
    )
    
    # Get class weights
    if weight_calc.weights is None:
        raise ValueError("Failed to calculate class weights")
    class_weights = weight_calc.weights.cpu().numpy()
    
    # Assign weight to each sample based on its class
    sample_weights = [class_weights[label] for label in labels]
    
    return sample_weights


def create_weighted_sampler(
    dataset,
    strategy: str = WeightingStrategy.BALANCED,
    beta: float = 0.999
) -> torch.utils.data.WeightedRandomSampler:
    """
    Create a WeightedRandomSampler for balanced batch sampling.
    
    Args:
        dataset: PyTorch dataset with samples attribute
        strategy: Weighting strategy to use
        beta: Beta parameter for effective number weighting
        
    Returns:
        WeightedRandomSampler instance
    """
    # Extract labels from dataset
    if hasattr(dataset, 'samples'):
        labels = [label for _, label in dataset.samples]
    elif hasattr(dataset, 'targets'):
        labels = dataset.targets
    else:
        raise ValueError("Dataset must have 'samples' or 'targets' attribute")
    
    # Calculate sample weights
    sample_weights = calculate_sample_weights(labels, strategy, beta)
    
    # Create sampler
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    logger.info(f"Created weighted sampler with {len(sample_weights)} samples")
    
    return sampler


def get_recommended_loss_params(
    class_distribution: Dict[int, int],
    dataset_size: int
) -> Dict[str, Any]:
    """
    Get recommended loss function parameters based on dataset characteristics.
    
    Args:
        class_distribution: Dictionary of class counts
        dataset_size: Total number of samples
        
    Returns:
        Dictionary of recommended parameters
    """
    # Calculate imbalance ratio
    counts = list(class_distribution.values())
    non_zero_counts = [c for c in counts if c > 0]
    
    if not non_zero_counts:
        return {"strategy": WeightingStrategy.BALANCED, "use_weighted_sampler": False}
    
    max_count = max(non_zero_counts)
    min_count = min(non_zero_counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Recommendations based on imbalance
    recommendations = {
        "imbalance_ratio": imbalance_ratio,
        "dataset_size": dataset_size,
        "num_classes": len(class_distribution)
    }
    
    if imbalance_ratio < 2:
        # Mild imbalance
        recommendations["strategy"] = WeightingStrategy.INVERSE_FREQUENCY
        recommendations["use_weighted_sampler"] = False
        recommendations["note"] = "Mild class imbalance detected. Using inverse frequency weighting."
        
    elif imbalance_ratio < 10:
        # Moderate imbalance
        recommendations["strategy"] = WeightingStrategy.BALANCED
        recommendations["use_weighted_sampler"] = True
        recommendations["note"] = "Moderate class imbalance detected. Using balanced weighting with weighted sampling."
        
    else:
        # Severe imbalance
        recommendations["strategy"] = WeightingStrategy.EFFECTIVE_NUMBER
        recommendations["use_weighted_sampler"] = True
        recommendations["beta"] = 0.999 if dataset_size > 10000 else 0.99
        recommendations["note"] = "Severe class imbalance detected. Using effective number weighting with weighted sampling."
    
    logger.info(f"Loss function recommendations: {recommendations['note']}")
    
    return recommendations