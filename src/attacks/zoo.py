# src/attacks/zoo.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from .base_attack import Attack

def zoo_attack(
    model,
    images,
    labels,
    epsilon=0.03,
    max_queries=2000,
    learning_rate=0.01,
    targeted=False,
    device="cuda",
    verbose=False
):
    """
    ZOO: Zeroth Order Optimization based black-box attack
    Estimates gradients using finite differences, then performs Adam-like update
    """
    model.eval()
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    batch_size = images.size(0)
    adv_images = images.clone()

    # Optimizer for perturbation
    delta = torch.zeros_like(images, requires_grad=False)
    optimizer = optim.Adam([delta], lr=learning_rate)

    queries_used = 0

    for query in tqdm(range(max_queries), desc="ZOO queries", disable=not verbose):
        optimizer.zero_grad()

        # Finite difference gradient estimation
        noise = torch.randn_like(images) * 1e-4
        noise = noise.sign() * 1e-4  # small step

        with torch.no_grad():
            loss_plus = nn.CrossEntropyLoss()(model(images + noise), labels)
            loss_minus = nn.CrossEntropyLoss()(model(images - noise), labels)

        estimated_grad = (loss_plus - loss_minus) / (2 * 1e-4) * noise.sign()
        delta.grad = estimated_grad

        optimizer.step()

        # Project perturbation
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        adv_images = torch.clamp(images + delta, 0, 1)

        queries_used += 2  # two forward passes per query

        # Early stop if all samples are successful
        with torch.no_grad():
            preds = model(adv_images).argmax(dim=1)
            if targeted:
                success = (preds == labels).float().mean()
            else:
                success = (preds != labels).float().mean()
            if success >= 0.99:
                if verbose:
                    print(f"Early stop at query {query+1}")
                break

    return adv_images, queries_used