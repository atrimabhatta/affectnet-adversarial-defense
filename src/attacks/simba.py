# src/attacks/simba.py

import torch
from tqdm.auto import tqdm


def simba_attack(
    model,
    images,
    labels,
    epsilon=0.2,
    num_queries=1000,
    device="cuda"
):
    """
    SIMBA: Simple Black-box Attack

    Args:
        model: target model
        images: batch of images [B,C,H,W]
        labels: true labels
        epsilon: perturbation step
        num_queries: max queries

    Returns:
        adversarial images
        total queries
    """

    model.eval()

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    batch_size = images.size(0)

    # flatten dimension
    flat_dim = images.numel() // batch_size

    delta = torch.zeros_like(images)

    # create random order of pixels
    perm = torch.randperm(flat_dim)

    queries = 0

    adv_images = images.clone()

    for i in tqdm(range(num_queries), desc="SIMBA queries"):

        if i >= flat_dim:
            break

        idx = perm[i]

        # convert flat index → pixel index
        delta_flat = delta.view(batch_size, -1)

        perturb = torch.zeros(batch_size, flat_dim, device=device)

        perturb[:, idx] = epsilon

        perturb = perturb.view_as(images)

        # positive direction
        x_pos = torch.clamp(adv_images + perturb, 0, 1)

        logits = model(x_pos)
        preds = logits.argmax(dim=1)

        improved = preds != labels

        adv_images[improved] = x_pos[improved]

        # negative direction
        x_neg = torch.clamp(adv_images - perturb, 0, 1)

        logits = model(x_neg)
        preds = logits.argmax(dim=1)

        improved = preds != labels

        adv_images[improved] = x_neg[improved]

        queries += 2

    return adv_images, queries