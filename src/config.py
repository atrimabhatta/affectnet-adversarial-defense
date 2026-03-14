# src/config.py
import torch
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 224
BATCH_SIZE = 24      # lower to 32/24 if you get OOM
EPOCHS     = 20
BASE_LR    = 1e-4
PATIENCE   = 7
EVAL_EVERY = 3

# Where models are saved
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Common normalization (used in both train & val)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]