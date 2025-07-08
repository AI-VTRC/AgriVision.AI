"""
Classification scheme definitions and label mapping logic for plant disease classification.

This module provides different classification schemes for organizing the dataset:
- Binary: Healthy vs Unhealthy
- 4-way: Real vs Diffusion vs GAN variants
- 10-way: All original classes maintaining full granularity
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ClassificationScheme(Enum):
    """Enumeration of available classification schemes."""
    BINARY = "binary"  # Healthy vs Unhealthy
    FOUR_WAY = "four_way"  # Real vs Diffusion vs GAN variants  
    TEN_WAY = "ten_way"  # All original classes
    FULL = "full"  # All original classes (same as dataset structure)


class ClassificationSchemeManager:
    """Manages different classification schemes and label mappings."""
    
    def __init__(self, plant_name: str):
        """
        Initialize the classification scheme manager.
        
        Args:
            plant_name: Name of the plant (e.g., 'Apple', 'Maize', 'Tomato')
        """
        self.plant_name = plant_name
        self._scheme_mappings = {
            ClassificationScheme.BINARY: self._create_binary_mapping,
            ClassificationScheme.FOUR_WAY: self._create_four_way_mapping,
            ClassificationScheme.TEN_WAY: self._create_ten_way_mapping,
            ClassificationScheme.FULL: self._create_full_mapping
        }
    
    def get_label_mapping(self, scheme: ClassificationScheme, class_dirs: List[str]) -> Dict[str, int]:
        """
        Get the label mapping for a specific classification scheme.
        
        Args:
            scheme: The classification scheme to use
            class_dirs: List of class directory names
            
        Returns:
            Dictionary mapping class names to indices
        """
        if scheme not in self._scheme_mappings:
            raise ValueError(f"Unknown classification scheme: {scheme}")
        
        return self._scheme_mappings[scheme](class_dirs)
    
    def get_mapped_label(self, original_class: str, scheme: ClassificationScheme) -> Tuple[str, int]:
        """
        Map an original class name to its label under a specific scheme.
        
        Args:
            original_class: Original class directory name
            scheme: The classification scheme to use
            
        Returns:
            Tuple of (mapped_class_name, label_index)
        """
        # Parse the original class name
        parts = original_class.split('-')
        if len(parts) < 4:
            raise ValueError(f"Invalid class name format: {original_class}")
        
        plant, health, generation, specific = parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else ""
        
        if scheme == ClassificationScheme.BINARY:
            if health.lower() == "healthy":
                return "Healthy", 0
            else:
                return "Unhealthy", 1
        
        elif scheme == ClassificationScheme.FOUR_WAY:
            if generation.lower() == "real":
                return "Real", 0
            elif generation.lower() == "diffusion":
                return "Diffusion", 1
            elif generation.lower() == "gan":
                # Group StyleGAN2 and StyleGAN3 together
                return "GAN", 2
            else:
                # Default to Real if unknown
                logger.warning(f"Unknown generation method: {generation}, defaulting to Real")
                return "Real", 0
        
        elif scheme == ClassificationScheme.TEN_WAY:
            # For 10-way, we need to create a simplified version
            # Combine health and generation method
            simplified = f"{health}-{generation}"
            
            # Define the 10 core classes
            ten_way_classes = {
                "Healthy-Real": 0,
                "Healthy-Diffusion": 1,
                "Healthy-GAN": 2,
                "Unhealthy-Real": 3,
                "Unhealthy-Diffusion": 4,
                "Unhealthy-GAN": 5,
                # Additional classes for specific variants if needed
                "Healthy-DS8": 6,
                "Healthy-SPx2Px": 7,
                "Unhealthy-DS8": 8,
                "Unhealthy-SPx2Px": 9
            }
            
            # Try to match with specific variant first
            if specific:
                specific_key = f"{health}-{specific}"
                if specific_key in ten_way_classes:
                    return specific_key, ten_way_classes[specific_key]
            
            # Fall back to general category
            if simplified in ten_way_classes:
                return simplified, ten_way_classes[simplified]
            else:
                # Default mapping for unknown combinations
                logger.warning(f"Unknown class combination for 10-way: {simplified}")
                return simplified, 0
        
        else:  # FULL scheme
            # Return original class name and generate index based on sorted order
            return original_class, -1  # Index will be assigned by the full mapping function
    
    def _create_binary_mapping(self, class_dirs: List[str]) -> Dict[str, int]:
        """Create binary classification mapping (Healthy vs Unhealthy)."""
        return {
            "Healthy": 0,
            "Unhealthy": 1
        }
    
    def _create_four_way_mapping(self, class_dirs: List[str]) -> Dict[str, int]:
        """Create 4-way classification mapping (Real, Diffusion, GAN variants)."""
        return {
            "Real": 0,
            "Diffusion": 1,
            "GAN": 2,
            "Mixed": 3  # Reserved for potential mixed/unknown cases
        }
    
    def _create_ten_way_mapping(self, class_dirs: List[str]) -> Dict[str, int]:
        """Create 10-way classification mapping."""
        # Extract unique combinations from the provided class directories
        unique_classes = set()
        
        for class_dir in class_dirs:
            parts = class_dir.split('-')
            if len(parts) >= 3:
                health = parts[1]
                generation = parts[2]
                
                # Simplify GAN variants
                if generation.lower() == "gan":
                    generation = "GAN"
                
                unique_classes.add(f"{health}-{generation}")
        
        # Sort for consistent ordering
        sorted_classes = sorted(unique_classes)
        
        # Create mapping
        mapping = {class_name: idx for idx, class_name in enumerate(sorted_classes)}
        
        logger.info(f"10-way classification mapping: {mapping}")
        return mapping
    
    def _create_full_mapping(self, class_dirs: List[str]) -> Dict[str, int]:
        """Create full classification mapping (all original classes)."""
        # Sort directories for consistent ordering
        sorted_dirs = sorted(class_dirs)
        return {class_name: idx for idx, class_name in enumerate(sorted_dirs)}
    
    def get_scheme_info(self, scheme: ClassificationScheme) -> Dict[str, Any]:
        """
        Get information about a classification scheme.
        
        Args:
            scheme: The classification scheme
            
        Returns:
            Dictionary with scheme information
        """
        info = {
            "scheme": scheme.value,
            "description": "",
            "num_classes": 0,
            "class_names": []
        }
        
        if scheme == ClassificationScheme.BINARY:
            info["description"] = "Binary classification: Healthy vs Unhealthy"
            info["num_classes"] = 2
            info["class_names"] = ["Healthy", "Unhealthy"]
        
        elif scheme == ClassificationScheme.FOUR_WAY:
            info["description"] = "4-way classification by generation method"
            info["num_classes"] = 4
            info["class_names"] = ["Real", "Diffusion", "GAN", "Mixed"]
        
        elif scheme == ClassificationScheme.TEN_WAY:
            info["description"] = "10-way classification with health status and generation method"
            info["num_classes"] = 10
            info["class_names"] = [
                "Healthy-Real", "Healthy-Diffusion", "Healthy-GAN",
                "Unhealthy-Real", "Unhealthy-Diffusion", "Unhealthy-GAN",
                "Healthy-DS8", "Healthy-SPx2Px", 
                "Unhealthy-DS8", "Unhealthy-SPx2Px"
            ]
        
        elif scheme == ClassificationScheme.FULL:
            info["description"] = "Full classification with all original classes"
            info["num_classes"] = -1  # Will be determined by actual data
            info["class_names"] = []  # Will be populated from data
        
        return info


def analyze_class_distribution(class_dirs: List[str], scheme: ClassificationScheme) -> Dict[str, int]:
    """
    Analyze the distribution of classes under a specific scheme.
    
    Args:
        class_dirs: List of class directory names
        scheme: The classification scheme to analyze
        
    Returns:
        Dictionary mapping class names to counts
    """
    manager = ClassificationSchemeManager("")  # Plant name not needed for analysis
    distribution = {}
    
    for class_dir in class_dirs:
        try:
            mapped_class, _ = manager.get_mapped_label(class_dir, scheme)
            distribution[mapped_class] = distribution.get(mapped_class, 0) + 1
        except ValueError as e:
            logger.warning(f"Error processing class {class_dir}: {e}")
    
    return distribution


def validate_dataset_structure(root_dir: str, plant_names: List[str]) -> Dict[str, List[str]]:
    """
    Validate the dataset structure and return available classes per plant.
    
    Args:
        root_dir: Root directory of the dataset
        plant_names: List of plant names to validate
        
    Returns:
        Dictionary mapping plant names to their available class directories
    """
    plant_classes = {}
    
    for plant in plant_names:
        classes = []
        
        # Find all directories for this plant
        for item in os.listdir(root_dir):
            if os.path.isdir(os.path.join(root_dir, item)) and item.startswith(f"{plant}-"):
                classes.append(item)
        
        if classes:
            plant_classes[plant] = sorted(classes)
            logger.info(f"Found {len(classes)} classes for {plant}")
        else:
            logger.warning(f"No classes found for {plant}")
    
    return plant_classes