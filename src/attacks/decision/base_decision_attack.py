import torch

class DecisionAttack:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device
        self.model.eval()

    def predict(self, x):
        with torch.no_grad():
            logits = self.model(x)
            return torch.argmax(logits, dim=1)

    def is_adversarial(self, x, label):
        pred = self.predict(x)
        return pred != label