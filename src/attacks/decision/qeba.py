import torch
from .base_decision_attack import DecisionAttack

class QEBA(DecisionAttack):

    def __init__(self, model, steps=300, sigma=0.001, device="cuda"):
        super().__init__(model, device)
        self.steps = steps
        self.sigma = sigma

    def attack(self, image, label):

        image = image.to(self.device)
        adv = torch.rand_like(image)

        while not self.is_adversarial(adv,label):
            adv = torch.rand_like(image)

        for _ in range(self.steps):

            noise = torch.randn_like(image)
            grad = noise / torch.norm(noise)

            candidate = adv + self.sigma * grad
            candidate = torch.clamp(candidate,0,1)

            if self.is_adversarial(candidate,label):
                adv = candidate

        return adv