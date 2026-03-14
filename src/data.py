# src/data.py
from pathlib import Path
import kagglehub
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight

# Import constants from the same package
from .config import IMG_SIZE, BATCH_SIZE, DEVICE, NORM_MEAN, NORM_STD


def get_dataloaders():
    """
    Loads AffectNet dataset via kagglehub, creates train/val transforms,
    handles class imbalance with weighted sampler and weighted loss.
    Returns: train_loader, val_loader, class_names, criterion
    """
    print("Fetching AffectNet via kagglehub...")
    dataset_path = kagglehub.dataset_download("mstjebashazida/affectnet")
    DATA_ROOT = Path(dataset_path)

    # Handle possible nested archive folder
    if (DATA_ROOT / "archive (3)").exists():
        DATA_ROOT = DATA_ROOT / "archive (3)"

    train_dir = DATA_ROOT / "Train"
    val_dir   = DATA_ROOT / "Test"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("Train or Test folder missing in dataset")

    # ── Transforms ───────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.4, scale=(0.02, 0.25), ratio=(0.3, 3.3)),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    # ── Imbalance handling ───────────────────────────────────────────────
    labels = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(labels)
    class_weights_np = compute_class_weight('balanced', classes=np.arange(len(class_counts)), y=labels)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # Weighted sampler
    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    class_names = train_dataset.classes
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train: {len(train_dataset):,}    Val: {len(val_dataset):,}")
    print("Using WeightedRandomSampler for training")

    return train_loader, val_loader, class_names, criterion