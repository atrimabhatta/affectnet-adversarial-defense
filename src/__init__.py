# src/__init__.py
from .config import *
from .data import get_dataloaders
from .models import get_model
from .train import train_model
from .evaluate import evaluate_clean, plot_training_history, plot_confusion_matrix_heatmap
from .attacks import *   # or be more specific
# src/__init__.py
# This file makes 'src' a proper Python package and exposes the most commonly used functions

# Config & core utilities
from .config import DEVICE, IMG_SIZE, BATCH_SIZE, EPOCHS, BASE_LR, PATIENCE, EVAL_EVERY

# Data loading
from .data import get_dataloaders

# Models
from .models import get_model

# Training
from .train import train_model, WarmupCosineScheduler

# Evaluation & plotting
from .evaluate import (
    evaluate_clean,
    plot_training_history,
    plot_confusion_matrix_heatmap
)

# Attacks ────────────────────────────────────────────────
# White-box
from .attacks.whitebox import (
    fgsm_attack,
    pgd_attack,
    evaluate_whitebox_attacks   # ← the main evaluation function
)

# Black-box from .attacks.blackbox import evaluate_score_based_blackbox

# Pixel-based
from .attacks.pixel import random_single_pixel_flip