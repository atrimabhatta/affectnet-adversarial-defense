# src/attacks/whitebox.py

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from ..config import DEVICE


def fgsm_attack(images, epsilon, grad):
    """Fast Gradient Sign Method - single step attack"""
    return torch.clamp(images + epsilon * grad.sign(), -3.0, 3.0)


def pgd_attack(model, images, labels, epsilon=0.03, alpha=0.007, steps=20):
    """Projected Gradient Descent - iterative attack"""
    adv = images.clone().detach()
    for _ in range(steps):
        adv.requires_grad_(True)
        outputs = model(adv)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        eta = alpha * adv.grad.sign()
        adv = adv + eta
        adv = torch.clamp(adv - images, -epsilon, epsilon) + images
        adv = torch.clamp(adv, -3.0, 3.0).detach()
    return adv


def evaluate_whitebox_attacks(model, loader, n_samples=400):
    """
    Evaluate white-box attacks (FGSM + PGD) on a subset of the validation set.
    Prints ASR (Attack Success Rate) for various epsilon values.
    """
    model.eval()
    epsilons = [0.00, 0.01, 0.03, 0.05, 0.10, 0.20]

    print("FGSM results:")
    for eps in epsilons:
        mis = tot = 0
        for images, labels in loader:
            take = min(images.size(0), n_samples - tot)
            if take <= 0:
                break
            batch_images = images[:take].to(DEVICE)
            batch_labels = labels[:take].to(DEVICE)

            if eps == 0:
                preds = model(batch_images).argmax(dim=1)
            else:
                batch_images.requires_grad_(True)
                outputs = model(batch_images)
                loss = nn.CrossEntropyLoss()(outputs, batch_labels)
                model.zero_grad()
                loss.backward()
                adv = fgsm_attack(batch_images, eps, batch_images.grad.data)
                preds = model(adv).argmax(dim=1)

            mis += (preds != batch_labels).sum().item()
            tot += take
            if tot >= n_samples:
                break

        rate = mis / tot * 100 if tot > 0 else 0
        print(f" ε = {eps:5.3f} → {rate:5.2f}% ASR")

    print("\nPGD-20 results:")
    for eps in epsilons[1:]:
        mis = tot = 0
        for images, labels in loader:
            take = min(images.size(0), n_samples - tot)
            if take <= 0:
                break
            batch_images = images[:take].to(DEVICE)
            batch_labels = labels[:take].to(DEVICE)
            adv = pgd_attack(model, batch_images, batch_labels, epsilon=eps)
            preds = model(adv).argmax(dim=1)
            mis += (preds != batch_labels).sum().item()
            tot += take
            if tot >= n_samples:
                break

        rate = mis / tot * 100 if tot > 0 else 0
        print(f" ε = {eps:5.3f} → {rate:5.2f}% ASR")

    print()