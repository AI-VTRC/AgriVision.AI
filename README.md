# AgriVision: Real vs Generated Image Detection  


We present a novel framework (namely: AgriVision) capable of classifying high-fidelity adversarial plant images. This supervised approach not only detects attacks but also identifies their specific source models. This study employs state-of-the-art GAN architectures, including StyleGAN2 and
StyleGAN3, alongside powerful diffusion models such as Dreamshaper-8, BLIP-Diffusion, and Pix2Pix, to produce diverse adversarial images through both image-to-image and text-to-image methods. As GANs and diffusion models advance, the proposed approach is designed to evolve, maintaining strong generation and detection capabilities against emerging threats in agricultural cyber-physical systems.  

---

## 🚀 Features
- Train and evaluate classifiers for **binary (real vs fake)**, **3-way**, and **multi-class attribution** (Pix2Pix, BLIP diffusion, DreamShaper-8, StyleGAN2/3).  
- Reproduce **adversarial image generation pipelines** used in the paper.  
- Support for **ResNet**, **EfficientNet**, and **OpenAI CLIP** backbones.  
- Dataset schema that scales across **multiple plants, health states, and sources**.  

---

## 🛠️ Installation
```bash
git clone https://github.com/yourusername/Foundation.git
cd Foundation
pip install -r requirements.txt
```

---

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

Recommended label schema:  
```
plant={apple|maize|tomato}
health={healthy|unhealthy}
source={real|pix2pix|blip|ds8|stylegan2|stylegan3}
split={train|val|test}
```

---

## 📊 Usage

### Training
```bash
# Train with CLIP ViT-B/32 model
python src/main.py --model_name clip-ViT-B/32 --plant Apple --data_dir ./datasets --output_dir ./outputs
```

### Evaluation
```bash
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

---

## 📜 Citation
If you use this repo or paper:  
```bibtex
@article{Yardimci2026AgriAdversarial,
  title   = {Adversarial Image Detection Using Deep Learning in Agricultural Contexts},
  author  = {Yardimci, M., Sikder, N. K., Ward, T., and Batarseh, F. A.},
  journal = {Preprint (in submission)},
  year    = {2026},
  month   = {October}
}
```

---

## 👥 Authors
- Mehmet Yardimci
- Md Nazmul Kabir Sikder    
- Trey Ward  
- Feras A. Batarseh  

Affiliation: Virginia Tech; A3 Lab.  
Web link: https://ai-vtrc.github.io/AgriVision.AI/

---

## 📄 License
MIT License (code). For datasets or generated samples, consider CC BY 4.0.  
