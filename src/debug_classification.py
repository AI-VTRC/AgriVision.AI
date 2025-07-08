"""Debug script to test classification scheme mapping"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classification_schemes import ClassificationScheme, ClassificationSchemeManager

# Test the classification mapping
plant_name = "Apple"
manager = ClassificationSchemeManager(plant_name)

# Test folders from your dataset
test_folders = [
    "Apple-Healthy-Diffusion-DS8",
    "Apple-Healthy-Diffusion-SPx2Px",
    "Apple-Healthy-GAN-StyleGAN2",
    "Apple-Healthy-GAN-Stylegan3",
    "Apple-Healthy-Real-Real",
    "Apple-Unhealthy-Diffusion-DS8",
    "Apple-Unhealthy-Diffusion-SPx2Px",
    "Apple-Unhealthy-GAN-StyleGAN2",
    "Apple-Unhealthy-GAN-Stylegan3",
    "Apple-Unhealthy-Real-Real"
]

print("Testing 10-way classification mapping:")
print("="*60)

for folder in test_folders:
    try:
        mapped_class, mapped_idx = manager.get_mapped_label(folder, ClassificationScheme.TEN_WAY)
        print(f"{folder:40} -> {mapped_class:20} (idx={mapped_idx})")
    except Exception as e:
        print(f"{folder:40} -> ERROR: {e}")

# Test what the actual 10-way mapping creates
print("\n\nActual 10-way mapping from _create_ten_way_mapping:")
print("="*60)
mapping = manager._create_ten_way_mapping(test_folders)
for class_name, idx in sorted(mapping.items(), key=lambda x: x[1]):
    print(f"{class_name:30} -> {idx}")