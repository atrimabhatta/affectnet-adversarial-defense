# run_attacks_only.py
"""
Run only the adversarial attacks + clean evaluation on a pre-trained model.

Usage:
  python run_attacks_only.py resnet50
  python run_attacks_only.py convnext_tiny
  python run_attacks_only.py efficientnetv2_rw_s
"""

import sys
import torch

# Import from your package
from src import (
    get_dataloaders,
    get_model,
    random_single_pixel_flip,
    evaluate_whitebox_attacks,
    evaluate_score_based_blackbox,
    evaluate_clean,
    plot_confusion_matrix_heatmap
)
from src.config import DEVICE


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_attacks_only.py <model_name>")
        print("Example: python run_attacks_only.py resnet50")
        sys.exit(1)

    # ── Handle model name (case-insensitive + typo correction) ──────────────
    requested = sys.argv[1].strip().lower()

    model_name_map = {
        "restnet50":   "resnet50",
        "resnet50":    "resnet50",
        "resnet":      "resnet50",
        "convnext":    "convnext_tiny",
        "convnext_tiny": "convnext_tiny",
        "efficientnet": "efficientnetv2_rw_s",
        "efficientnetv2_rw_s": "efficientnetv2_rw_s",
    }

    model_name = model_name_map.get(requested, requested)

    if model_name not in ["resnet50", "convnext_tiny", "efficientnetv2_rw_s"]:
        print(f"Unknown model: {requested}")
        print("Supported: resnet50, convnext_tiny, efficientnetv2_rw_s")
        sys.exit(1)

    checkpoint_path = f"models/best_{model_name}.pth"

    print(f"Loading model : {model_name.upper()}")
    print(f"Checkpoint    : {checkpoint_path}")
    print(f"Device        : {DEVICE}")

    # ── 1. Load validation data ─────────────────────────────────────────────
    _, val_loader, class_names, _ = get_dataloaders()
    NUM_CLASSES = len(class_names)

    # ── 2. Create model + load trained weights ──────────────────────────────
    model = get_model(model_name, NUM_CLASSES).to(DEVICE)

    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("→ Model weights loaded successfully")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found → {checkpoint_path}")
        print("→ You need to train the model first:")
        print(f"   python run_train.py {model_name}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        print("Possible cause: model architecture mismatch or corrupted file.")
        sys.exit(1)

    model.eval()  # Important!

    # ── 3. Clean evaluation + confusion matrix ──────────────────────────────
    print("\n" + "═"*90)
    print("Clean evaluation (no attack)")
    print("═"*90)
    evaluate_clean(model, val_loader, verbose=True)
    plot_confusion_matrix_heatmap(model, val_loader, class_names, model_name.upper())

    # ── 4. Attacks ──────────────────────────────────────────────────────────
    print("\n" + "═"*90)
    print("Single-pixel attack")
    print("═"*90)
    random_single_pixel_flip(model, val_loader, n_images=400, trials=50)

    print("\n" + "═"*90)
    print("White-box attacks (FGSM + PGD)")
    print("═"*90)
    evaluate_whitebox_attacks(model, val_loader, n_samples=400)

    print("\n" + "═"*90)
    print("All attacks completed.")
    print("═"*90)


if __name__ == "__main__":
    main()