import torch


class Attack:

    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def get_logits(self, x):
        """
        Forward pass through the model
        """
        return self.model(x)

    def forward(self, images, labels):
        raise NotImplementedError

    def __call__(self, images, labels):
        return self.forward(images, labels)