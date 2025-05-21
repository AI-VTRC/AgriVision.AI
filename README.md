# CV for Ag

A system for classifying healthy and unhealthy leaf images across different plant species using ResNet, and OpenAI's CLIP.

## Installation

```bash
git clone https://github.com/yourusername/Foundation.git
cd Foundation
pip install -r requirements.txt
```

## Dataset Structure

The dataset should be organized as follows:
```
datasets/
└── Apple/
|   ├── Apple-Healthy-Real-Real/
|   ├── Apple-Unhealthy-Real-Real/
|   ├── Apple-Healthy-Diffusion-SDXL/
|   ├── Apple-Unhealthy-Diffusion-SDXL/
|   ├── Apple-Healthy-Diffusion-SPx2Px/
|   ├── Apple-Unhealthy-Diffusion-SPx2Px/
|   ├── Apple-Healthy-GAN-DCGAN/
|   ├── Apple-Unhealthy-GAN-DCGAN/
|   ├── Apple-Healthy-GAN-StyleGAN2/
|   └── Apple-Unhealthy-GAN-StyleGAN2/
└── Maize/
| ...
└── Tomato/
| ...
```
## Usage

### Sample Commands

```bash
# Train with CLIP ViT-B/32 model
python src/main.py --model_name clip-ViT-B/32 --plant Apple --data_dir ./datasets --output_dir ./outputs

# Train with EfficientNet B0 model
python src/main.py --model_name efficientnet_b0 --plant Apple

# Evaluate a trained model
python src/main.py --eval --checkpoint ./outputs/Apple_best_model.pth --plant Apple
```

### Available Models

- `clip-ViT-B/32`: CLIP ViT-B/32 model
- `clip-ViT-B/16`: CLIP ViT-B/16 model
- `clip-ViT-L/14`: CLIP ViT-L/14 model
- `efficientnet_b0`: EfficientNet B0 model
- `resnet50`: ResNet-50 model

### Key Parameters

- `--plant`: Plant name (default: Apple)
- `--data_dir`: Data directory (default: ./datasets)
- `--output_dir`: Output directory (default: ./outputs)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 20)
- `--lr`: Learning rate (default: 0.001)
- `--eval`: Evaluate model instead of training
- `--checkpoint`: Path to model checkpoint

## Outputs

The system generates the following in your output directory:
- `[plant]_best_model.pth`: Best model checkpoint
- `[plant]_training_metrics.png`: Training and validation loss/accuracy curves
- `[plant]_evaluation/`: Directory containing:
  - Confusion matrix visualization
  - Sample predictions
  - Classification metrics

## License

This project is licensed under the MIT License - see the LICENSE file for details. 