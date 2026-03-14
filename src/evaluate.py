# src/evaluate.py

import torch
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from .config import DEVICE


@torch.no_grad()
def evaluate_clean(model, loader, verbose=True):
    """
    Evaluate model on clean data (no attacks).
    Returns: accuracy, macro F1
    """
    model.eval()
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, desc="Clean eval", leave=False):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    acc = np.mean(np.array(y_true) == np.array(y_pred))
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    if verbose:
        print(f"Clean Accuracy : {acc:.4f}")
        print(f"Macro F1       : {f1_macro:.4f}\n")

    return acc, f1_macro


def plot_training_history(history, model_name="Model"):
    """Plot loss, train/val acc, and val macro F1 over epochs"""
    epochs = history['epochs']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.set_title(f'{model_name} – Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy & F1
    ax2.plot(epochs, history['train_acc'], label='Train Acc', color='green')
    ax2.plot(epochs, history['val_acc'], label='Val Acc', color='orange', linestyle='--')
    ax2.plot(epochs, history['val_f1'], label='Val Macro F1', color='red', linestyle='-.')
    ax2.set_title(f'{model_name} – Accuracy & Macro F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training History: {model_name}", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_heatmap(model, loader, class_names, model_name="Model"):
    """Generate and display confusion matrix heatmap"""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="CM predictions"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title(f'Confusion Matrix – {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()