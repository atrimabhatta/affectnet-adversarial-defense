import torch
from src.models import load_model
from src.data import get_dataloader

from src.attacks.decision.boundary import BoundaryAttack
from src.attacks.decision.hsja import HSJA
from src.attacks.decision.qeba import QEBA
from src.attacks.decision.geoda import GeoDA
from src.attacks.decision.surfree import SurFree


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = {
    "resnet50": "models/best_resnet50.pth",
    "efficientnetv2_rw_s": "models/best_efficientnetv2_rw_s.pth",
    "convnext_tiny": "models/best_convnext_tiny.pth"
}


ATTACKS = {
    "boundary": BoundaryAttack,
    "hsja": HSJA,
    "qeba": QEBA,
    "geoda": GeoDA,
    "surfree": SurFree
}


def run_attack(model_name, model_path, dataloader):

    print(f"\nRunning attacks on {model_name}")

    model = load_model(model_name)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    for attack_name, attack_class in ATTACKS.items():

        print(f"\nAttack: {attack_name}")

        attack = attack_class(model)

        success = 0
        total = 0

        for images, labels in dataloader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            adv = attack.attack(images, labels)

            with torch.no_grad():
                preds = model(adv).argmax(dim=1)

            success += (preds != labels).sum().item()
            total += labels.size(0)

        print(f"{attack_name} success rate: {success/total:.3f}")


def main():

    dataloader = get_dataloader(split="val")

    for model_name, model_path in MODELS.items():

        run_attack(model_name, model_path, dataloader)


if __name__ == "__main__":
    main()