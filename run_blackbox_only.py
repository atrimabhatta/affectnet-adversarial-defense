# run_blackbox_only.py
"""
Run ONLY black-box attacks (ZOO + Square + SIMBA) on a pre-trained AffectNet model.
Shows ASR (Attack Success Rate) for each attack across different epsilon values.

Usage:
  python run_blackbox_only.py resnet50
"""

import sys
import torch

from src import (
    get_dataloaders,
    get_model,
)
from src.attacks.zoo import zoo_attack
from src.attacks.square import SquareAttack
from src.attacks.simba import simba_attack
from src.config import DEVICE


def calculate_asr(adv_images, original_images, labels, model, targeted=False):
    """Calculate Attack Success Rate"""
    with torch.no_grad():
        preds = model(adv_images).argmax(dim=1)

    if targeted:
        success = (preds == labels).float().mean().item() * 100
    else:
        success = (preds != labels).float().mean().item() * 100

    return success


def main():

    if len(sys.argv) < 2:
        print("Usage: python run_blackbox_only.py <model_name>")
        print("Example: python run_blackbox_only.py resnet50")
        sys.exit(1)

    requested = sys.argv[1].strip().lower()

    model_map = {
        "restnet50": "resnet50",
        "resnet": "resnet50",
        "convnext": "convnext_tiny",
        "efficientnet": "efficientnetv2_rw_s",
    }

    model_name = model_map.get(requested, requested)

    valid_models = ["resnet50", "convnext_tiny", "efficientnetv2_rw_s"]

    if model_name not in valid_models:
        print(f"Unknown model: {requested}")
        print(f"Supported: {', '.join(valid_models)}")
        sys.exit(1)

    checkpoint_path = f"models/best_{model_name}.pth"

    print(f"Loading model: {model_name.upper()}")
    print(f"Checkpoint   : {checkpoint_path}")
    print(f"Device       : {DEVICE}")

    # Load validation loader
    _, val_loader, class_names, _ = get_dataloaders()

    NUM_CLASSES = len(class_names)

    model = get_model(model_name, NUM_CLASSES).to(DEVICE)

    try:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print("→ Model loaded successfully")

    except FileNotFoundError:
        print(f"Checkpoint not found: {checkpoint_path}")
        print(f"Train first: python run_train.py {model_name}")
        sys.exit(1)

    except Exception as e:
        print(f"Load error: {e}")
        sys.exit(1)

    model.eval()

    # Take one batch for quick evaluation
    print("\nTaking one batch from validation set...")

    images, labels = next(iter(val_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    batch_size = images.size(0)

    print(f"Batch size: {batch_size} images")

    # Epsilon values to test
    epsilons = [0.005, 0.01, 0.02, 0.03, 0.05]

    for eps in epsilons:

        print("\n" + "═" * 90)
        print(f"Black-box attacks only: {model_name.upper()} | ε = {eps}")
        print("═" * 90)

        # ── ZOO Attack ─────────────────────────────────────────
        print("ZOO (Zeroth-Order Optimization) Attack")
        print("-" * 60)

        adv_zoo, queries_zoo = zoo_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=eps,
            max_queries=1000,
            learning_rate=0.01,
            targeted=False,
            device=DEVICE,
            verbose=True
        )

        asr_zoo = calculate_asr(adv_zoo, images, labels, model)

        print(f"ZOO → ASR: {asr_zoo:.2f}% | Queries: {queries_zoo}")

        # ── Square Attack ─────────────────────────────────────
        print("\nSquare Attack")
        print("-" * 60)

        square = SquareAttack(
            model=model,
            norm='Linf',
            eps=eps,
            n_queries=1500,
            n_restarts=1,
            p_init=0.8,
            loss='margin',
            resc_schedule=True,
            seed=0,
            verbose=False,
            targeted=False,
            device=DEVICE
        )

        try:
            adv_square = square.perturb(images, labels)
            asr_square = calculate_asr(adv_square, images, labels, model)

            print(f"Square → ASR: {asr_square:.2f}%")

        except Exception as e:
            print(f"Square failed: {e}")
            asr_square = 0

        # ── SIMBA Attack ─────────────────────────────────────
        print("\nSIMBA (Simple Black-box Attack)")
        print("-" * 60)

        adv_simba, queries_simba = simba_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=eps,
            num_queries=1000,
            device=DEVICE
        )

        asr_simba = calculate_asr(adv_simba, images, labels, model)

        print(f"SIMBA → ASR: {asr_simba:.2f}% | Queries: {queries_simba}")

        # ── Summary Table ─────────────────────────────────────
        print("\n" + "═" * 90)
        print(f"Black-box Attack Results Summary (ε = {eps})")
        print("═" * 90)

        print(f"{'Attack':<15} {'ASR (%)':<10} {'Queries used':<15}")
        print("-" * 50)

        print(f"{'ZOO':<15} {asr_zoo:<10.2f} {queries_zoo:<15}")
        print(f"{'Square':<15} {asr_square:<10.2f} {'N/A':<15}")
        print(f"{'SIMBA':<15} {asr_simba:<10.2f} {queries_simba:<15}")

        print("═" * 90)


if __name__ == "__main__":
    main()