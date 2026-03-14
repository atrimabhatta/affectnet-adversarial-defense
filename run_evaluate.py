# run_eval_attacks.py
import sys
from src.config import *
from src.data import get_dataloaders
from src.models import get_model
from src.evaluate import evaluate_clean, plot_training_history, plot_confusion_matrix_heatmap
from src.attacks import (
    random_single_pixel_flip,
    evaluate_attacks,
    evaluate_score_based_blackbox
)
from src.attacks.whitebox import evaluate_whitebox_attacks
from src.attacks.blackbox import evaluate_score_based_blackbox
from src.attacks.pixel import random_single_pixel_flip

# example usage:
random_single_pixel_flip(model, val_loader)
evaluate_whitebox_attacks(model, val_loader, n_samples=400)
evaluate_score_based_blackbox(model, val_loader, n_samples=300)
if __name__ == "__main__":
    _, val_loader, class_names, _ = get_dataloaders()
    NUM_CLASSES = len(class_names)

    if len(sys.argv) < 2:
        print("Usage: python run_eval_attacks.py resnet50")
        print("       python run_eval_attacks.py convnext_tiny")
        sys.exit(1)

    model_name = sys.argv[1]
    print(f"\nEvaluating & attacking: {model_name.upper()}")

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(f"models/best_{model_name}.pth", map_location=DEVICE))
        print("→ Model loaded successfully")
    except FileNotFoundError:
        print(f"Error: models/best_{model_name}.pth not found. Train first!")
        sys.exit(1)

    model.eval()

    evaluate_clean(model, val_loader, verbose=True)
    plot_confusion_matrix_heatmap(model, val_loader, class_names, model_name.upper())

    print("\nSingle-pixel attack:")
    random_single_pixel_flip(model, val_loader, n_images=400, trials=50)

    print("\nWhite-box attacks:")
    evaluate_attacks(model, val_loader, n_samples=400)

    print("\nBlack-box attacks:")
    evaluate_score_based_blackbox(model, val_loader, n_samples=300)