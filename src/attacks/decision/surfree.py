import torch
from .base_decision_attack import DecisionAttack

class SurFree(DecisionAttack):

    def __init__(self, model, steps=300, device="cuda"):
        super().__init__(model, device)
        self.steps = steps

    def attack(self, image, label):

        image = image.to(self.device)
        adv = torch.rand_like(image)

        while not self.is_adversarial(adv,label):
            adv = torch.rand_like(image)

        for _ in range(self.steps):

            direction = image - adv
            direction = direction / torch.norm(direction)

            orthogonal = torch.randn_like(direction)
            orthogonal = orthogonal - torch.dot(
                orthogonal.flatten(), direction.flatten()
            ) * direction

            orthogonal = orthogonal / torch.norm(orthogonal)

            candidate = adv + 0.01 * direction + 0.01 * orthogonal
            candidate = torch.clamp(candidate,0,1)

            if self.is_adversarial(candidate,label):
                adv = candidate

        return adv