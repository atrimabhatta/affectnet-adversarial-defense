import torch
from .base_decision_attack import DecisionAttack

class HSJA(DecisionAttack):

    def __init__(self, model, steps=50, device="cuda"):
        super().__init__(model, device)
        self.steps = steps

    def binary_search(self, x, adv):

        low = 0.0
        high = 1.0

        for _ in range(10):

            mid = (low + high) / 2
            blended = mid * adv + (1 - mid) * x

            if self.is_adversarial(blended, self.label):
                high = mid
            else:
                low = mid

        return high * adv + (1 - high) * x


    def attack(self, image, label):

        self.label = label
        image = image.to(self.device)

        adv = torch.rand_like(image)

        while not self.is_adversarial(adv, label):
            adv = torch.rand_like(image)

        for _ in range(self.steps):

            grad = torch.randn_like(image)
            grad = grad / torch.norm(grad)

            candidate = adv + 0.01 * grad
            candidate = torch.clamp(candidate,0,1)

            if self.is_adversarial(candidate,label):
                adv = candidate

            adv = self.binary_search(image, adv)

        return adv