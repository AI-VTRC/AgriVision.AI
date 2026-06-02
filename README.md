# AgriVision: Adversarial Image Detection Using Deep Learning in Agricultural Contexts

AgriVision is a unified **detection and attribution** framework for agricultural cyber-physical systems (CPSs). It not only decides whether a leaf image is real or synthetic, but also attributes synthetic images to their underlying generative family, supporting forensic analysis of adversarial manipulation in disease-detection and yield-estimation pipelines.

Synthetic concealment and fabrication attacks are created with multiple GANs (**StyleGAN2**, **StyleGAN3**, **R3GAN**) and diffusion models (**Instruct-Pix2Pix**, **BLIP-Diffusion**, **DreamShaper-8**) across **apple, maize, and tomato** imagery derived from PlantVillage. Three vision classifiers (**EfficientNet-B0**, **ResNet-50**, **CLIP ViT-B/32**) are evaluated on three tasks, and robustness is stress-tested against **previously unseen generators** (R3GAN and DreamShaper-8 are held out from training) to emulate "zero-day" generative threats.

---

## 🚀 Features
- Train and evaluate classifiers for three tasks:
  - **Binary authenticity** — Real vs. Synthetic
  - **Generation source** — Real vs. GAN vs. Diffusion (3-way)
  - **Detailed attribution** — joint plant–health–source classification
- Reproduce the **adversarial image generation pipelines** used in the paper (GAN training + diffusion image-to-image / inpainting).
- **Unseen-generator generalization**: hold out R3GAN and DreamShaper-8 to measure out-of-distribution robustness.
- Operational **concealment** and **fabrication** attack scenarios, plus **transformation robustness** (JPEG, resize, blur, noise, color).
- Support for **EfficientNet**, **ResNet**, and **OpenAI CLIP** backbones.
- Dataset schema that scales across **multiple plants, health states, and generative sources**.

---

## 🛠️ Installation
```bash
git clone https://github.com/AI-VTRC/AgriVision.AI.git
cd AgriVision.AI
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
|   ├── Apple-Healthy-Diffusion-SPx2Px/
|   ├── Apple-Unhealthy-Diffusion-SPx2Px/
|   ├── Apple-Healthy-Diffusion-BLIP/
|   ├── Apple-Unhealthy-Diffusion-BLIP/
|   ├── Apple-Healthy-GAN-StyleGAN2/
|   ├── Apple-Unhealthy-GAN-StyleGAN2/
|   ├── Apple-Healthy-GAN-StyleGAN3/
|   ├── Apple-Unhealthy-GAN-StyleGAN3/
|   ├── Apple-Healthy-Diffusion-DS8/        # held out: unseen-generator eval only
|   ├── Apple-Unhealthy-Diffusion-DS8/      # held out: unseen-generator eval only
|   ├── Apple-Healthy-GAN-R3GAN/            # held out: unseen-generator eval only
|   └── Apple-Unhealthy-GAN-R3GAN/          # held out: unseen-generator eval only
└── Maize/
| ...
└── Tomato/
| ...
```

Recommended label schema:  
```
plant={apple|maize|tomato}
health={healthy|unhealthy}
source={real|pix2pix|blip|stylegan2|stylegan3|ds8|r3gan}
split={train|val|test}
```

> **Note:** `ds8` (DreamShaper-8) and `r3gan` (R3GAN) are **held out from training and validation** and used exclusively for the unseen-generator generalization study.

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
@article{Yardimci2026AgriVision,
  title   = {AgriVision: Adversarial Image Detection Using Deep Learning in Agricultural Contexts},
  author  = {Yardimci, Mehmet Oguz and Sikder, Md Nazmul Kabir and Ward, Trey and Batarseh, Feras A.},
  journal = {Preprint (in submission)},
  year    = {2026}
}
```

---

## 👥 Authors
- Mehmet Oguz Yardimci — Department of Computer Science, Virginia Tech *(corresponding author)*
- Md Nazmul Kabir Sikder — School of Cybersecurity, Old Dominion University
- Trey Ward — Department of Biological Systems Engineering, Virginia Tech
- Feras A. Batarseh — Department of Biological Systems Engineering & The Commonwealth Cyber Initiative, Virginia Tech

Mehmet Oguz Yardimci and Md Nazmul Kabir Sikder contributed equally to this work.

Affiliation: Virginia Tech (A3 Lab) and Old Dominion University.  
Web link: https://ai-vtrc.github.io/AgriVision.AI/

---

## 📄 License
MIT License (code). For datasets or generated samples, consider CC BY 4.0.  
