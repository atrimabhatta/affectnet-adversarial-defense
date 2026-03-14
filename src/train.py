# src/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
import math

from .config import DEVICE, EPOCHS, BASE_LR, PATIENCE, EVAL_EVERY


class WarmupCosineScheduler:
    """Warmup + cosine annealing scheduler"""
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=EPOCHS,
                 warmup_lr=1e-6, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            lr = self.warmup_lr + (BASE_LR - self.warmup_lr) * (self.epoch / self.warmup_epochs)
        else:
            progress = (self.epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (BASE_LR - self.min_lr) * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_model(model, train_loader, val_loader, criterion,
                epochs=EPOCHS, base_lr=BASE_LR, model_name="model"):
    """
    Full training loop with:
    - AdamW + weight decay
    - Warmup + cosine LR scheduler
    - Automatic Mixed Precision (AMP)
    - Gradient clipping (max_norm=1.0)
    - Early stopping on validation macro F1
    - Training history for later plotting

    Args:
        model:          The model (already on DEVICE)
        train_loader:   Weighted DataLoader
        val_loader:     Validation DataLoader
        criterion:      Weighted CrossEntropyLoss
        epochs, base_lr, model_name: obvious

    Returns:
        history (dict), best_val_f1 (float)
    """
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer)
    scaler = GradScaler('cuda')

    best_f1 = 0.0
    patience_counter = 0

    history = {
        'train_loss': [],
        'train_acc':  [],
        'val_acc':    [],
        'val_f1':     [],
        'epochs':     []
    }

    print(f"\nStarting training: {model_name}  |  max epochs: {epochs}")

    try:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = total = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for images, labels in pbar:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)

                with autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}"
                )

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            current_lr = scheduler.step()

            print(f"Epoch {epoch+1:2d} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | LR: {current_lr:.2e}")

            history['train_loss'].append(epoch_loss)
            history['train_acc'].append(epoch_acc)
            history['epochs'].append(epoch + 1)

            # Validation step
            if (epoch + 1) % EVAL_EVERY == 0 or epoch == 0:
                # Late import to avoid circular dependency with evaluate.py
                from .evaluate import evaluate_clean

                val_acc, val_f1 = evaluate_clean(model, val_loader, verbose=False)
                print(f"Val Acc: {val_acc:.4f} | Val Macro F1: {val_f1:.4f}")

                history['val_acc'].append(val_acc)
                history['val_f1'].append(val_f1)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    patience_counter = 0
                    print(f"→ New best Macro F1: {best_f1:.4f}")
                    torch.save(model.state_dict(), f"models/best_{model_name}.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} checks)")
                        break
            else:
                # For plotting continuity — repeat last validation value
                if history['val_acc']:
                    history['val_acc'].append(history['val_acc'][-1])
                    history['val_f1'].append(history['val_f1'][-1])
                else:
                    history['val_acc'].append(float('nan'))
                    history['val_f1'].append(float('nan'))

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt caught! Saving checkpoint...")
        torch.save(model.state_dict(), f"models/interrupt_{model_name}_epoch{epoch+1}.pth")
        raise

    return history, best_f1