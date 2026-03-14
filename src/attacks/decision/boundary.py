import torch
from .base_decision_attack import DecisionAttack

class BoundaryAttack(DecisionAttack):

    def __init__(self, model, steps=1000, epsilon=0.01, device="cuda"):
        super().__init__(model, device)
        self.steps = steps
        self.epsilon = epsilon

    def attack(self, image, label):

        image = image.to(self.device)
        label = label.to(self.device)

        adv = torch.rand_like(image).to(self.device)

        while not self.is_adversarial(adv, label):
            adv = torch.rand_like(image).to(self.device)

        for _ in range(self.steps):

            direction = image - adv
            direction = direction / torch.norm(direction)

            candidate = adv + self.epsilon * direction

            noise = torch.randn_like(candidate) * self.epsilon
            candidate = torch.clamp(candidate + noise, 0, 1)

            if self.is_adversarial(candidate, label):
                adv = candidate

        return adv