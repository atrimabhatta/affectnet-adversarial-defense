# affectnet-adversarial-defense
<div align="center">

# AffectNet Adversarial Defense 🛡️😄

**Robustifying Facial Expression Recognition against Adversarial Attacks**  
Experiments on defending deep models trained on the [AffectNet](http://mohammadmahoor.com/affectnet/) dataset

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stars](https://img.shields.io/github/stars/atrimabhatta/affectnet-adversarial-defense?style=social)](https://github.com/atrimabhatta/affectnet-adversarial-defense)
[![Issues](https://img.shields.io/github/issues/atrimabhatta/affectnet-adversarial-defense)](https://github.com/atrimabhatta/affectnet-adversarial-defense/issues)

</div>

## 🌟 Overview

This repository contains experiments on **adversarial robustness** for **facial expression/emotion recognition** models trained on the large-scale **AffectNet** dataset (in-the-wild facial expressions with 8 discrete categories + valence/arousal annotations).

We explore:

- White-box & black-box adversarial attacks (FGSM, PGD, CW, etc.)
- Defense strategies (adversarial training, input preprocessing, certified defenses, etc.)
- Trade-offs between clean accuracy and robust accuracy on AffectNet

Goal: Build more reliable emotion AI systems that resist malicious perturbations — important for real-world applications like mental health monitoring, human-robot interaction, and affective computing.

## 🚀 Key Features

- Modular PyTorch codebase for attacks & defenses
- Pre-trained baselines (ResNet, EfficientNet, Vision Transformers, etc.) on AffectNet
- Evaluation on standard adversarial robustness metrics
- Visualizations: adversarial examples, perturbation heatmaps, confusion matrices
- Reproducible training & attack scripts

## 📊 Results Highlights (example — update with your numbers!)

| Model                  | Clean Acc (%) | PGD-20 Robust Acc (%) | Attack Used     | Defense Strategy          |
|------------------------|---------------|-----------------------|-----------------|---------------------------|
| ResNet-50 Baseline     | 62.1          | 18.4                  | PGD ε=8/255     | None                      |
| Adv-Trained ResNet-50  | 58.7          | 41.2                  | PGD ε=8/255     | Adversarial Training      |
| EfficientNet-B4 + Preproc | 64.5       | 35.8                  | CW              | JPEG + Randomization      |
| ViT-B/16 (best so far) | **66.9**      | **44.1**              | AutoAttack      | TRADES + Mixup            |

*Last updated: March 2026*

## 🛠️ Installation

```bash
# Clone repo
git clone https://github.com/atrimabhatta/affectnet-adversarial-defense.git
cd affectnet-adversarial-defense

# Create virtual env (recommended)
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt