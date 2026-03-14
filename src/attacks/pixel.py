# src/attacks/pixel.py

import torch
from tqdm.auto import tqdm

from ..config import DEVICE


def random_single_pixel_flip(model, loader, n_images=400, trials=50):
    """
    Evaluate robustness against random single-pixel perturbations.
    Returns Attack Success Rate (%)
    """
    model.eval()
    flip_count = total_tests = 0

    with torch.no_grad():
        for images, labels in loader:
            if total_tests >= n_images * trials:
                break
            images = images.to(DEVICE)
            orig_preds = model(images).argmax(dim=1)

            for i in range(images.size(0)):
                if total_tests >= n_images * trials:
                    break
                img = images[i]
                orig = orig_preds[i].item()

                for _ in range(trials):
                    x = torch.randint(0, img.size(2), (1,)).item()
                    y = torch.randint(0, img.size(1), (1,)).item()
                    delta = (torch.rand(1).item() - 0.5) * 0.9

                    pert = img.clone()
                    pert[:, y, x] += delta
                    pert.clamp_(-3.0, 3.0)

                    pred = model(pert.unsqueeze(0)).argmax(1).item()
                    if pred != orig:
                        flip_count += 1
                    total_tests += 1

    asr = flip_count / total_tests * 100 if total_tests > 0 else 0
    print(f"Random single-pixel flip ASR: {asr:.2f}%  ({flip_count}/{total_tests})")
    return asr