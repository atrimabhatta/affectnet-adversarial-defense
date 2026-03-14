# run_train.py
# ────────────────────────────────────────────────
# This script trains one or all models and saves them to models/
# Usage:
#   python run_train.py               → trains all three backbones
#   python run_train.py resnet50      → trains only resnet50

import sys
import torch
from tqdm.auto import tqdm

# Import from your package
from src import (
    get_dataloaders,
    get_model,
    train_model
)
from src.config import DEVICE, EPOCHS, BASE_LR


def main():
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data once (shared across all models)
    print("\nPreparing data loaders...")
    train_loader, val_loader, class_names, criterion = get_dataloaders()
    NUM_CLASSES = len(class_names)

    # Choose which models to train
    backbones = ["resnet50", "convnext_tiny", "efficientnetv2_rw_s"]
    
    if len(sys.argv) > 1:
        requested = sys.argv[1].lower()
        if requested in [b.lower() for b in backbones]:
            backbones = [b for b in backbones if b.lower() == requested]
        else:
            print(f"Unknown model '{requested}'. Training all.")
    
    print(f"\nWill train: {', '.join(backbones)}")

    for bb in backbones:
        print(f"\n{'═'*80}\nTraining {bb.upper()}\n{'═'*80}")
        
        model = get_model(bb, NUM_CLASSES).to(DEVICE)
        
        try:
            history, best_f1 = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                epochs=EPOCHS,
                base_lr=BASE_LR,
                model_name=bb
            )
            
            print(f"\nFinished {bb.upper()} | Best Val Macro F1: {best_f1:.4f}")
            
            # Optional: you can plot here if you want immediate visualization
            # from src.evaluate import plot_training_history
            # plot_training_history(history, bb.upper())
            
        except Exception as e:
            print(f"Training failed for {bb}: {str(e)}")
            continue

    print("\nAll requested trainings completed.")


if __name__ == "__main__":
    main()