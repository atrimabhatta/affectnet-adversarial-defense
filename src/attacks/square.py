# src/attacks/square.py

import math
import torch
import torch.nn.functional as F

from .base_attack import Attack


class SquareAttack(Attack):
    """
    Square Attack
    Query-efficient black-box adversarial attack via random search
    Paper: https://arxiv.org/abs/1912.00049
    """

    def __init__(
        self,
        model,
        norm="Linf",
        eps=8/255,
        n_queries=1000,
        n_restarts=1,
        p_init=0.8,
        loss="margin",
        resc_schedule=True,
        seed=0,
        verbose=False,
        targeted=False,
        device=None
    ):

        super().__init__(model)

        self.norm = norm
        self.eps = eps
        self.n_queries = n_queries
        self.n_restarts = n_restarts
        self.p_init = p_init
        self.loss = loss
        self.rescale_schedule = resc_schedule
        self.seed = seed
        self.verbose = verbose
        self.targeted = targeted

        # automatic device detection
        self.device = device if device else next(model.parameters()).device

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

    def random_choice(self, shape):
        """Generate random ±1 tensor"""
        return torch.sign(2 * torch.rand(*shape, device=self.device) - 1)

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        return self.perturb(images, labels)

    def margin_and_loss(self, x, y):

        logits = self.get_logits(x)

        xent = F.cross_entropy(logits, y, reduction="none")

        u = torch.arange(x.shape[0], device=self.device)

        y_corr = logits[u, y].clone()
        logits[u, y] = -float("inf")

        y_others = logits.max(dim=-1)[0]

        if not self.targeted:

            if self.loss == "ce":
                return y_corr - y_others, -xent

            return y_corr - y_others, y_corr - y_others

        else:

            if self.loss == "ce":
                return y_others - y_corr, xent

            return y_others - y_corr, y_others - y_corr

    def p_selection(self, it):

        if self.rescale_schedule:
            it = int(it / self.n_queries * 10000)

        if 10 < it <= 50:
            p = self.p_init / 2
        elif 50 < it <= 200:
            p = self.p_init / 4
        elif 200 < it <= 500:
            p = self.p_init / 8
        elif 500 < it <= 1000:
            p = self.p_init / 16
        elif 1000 < it <= 2000:
            p = self.p_init / 32
        elif 2000 < it <= 4000:
            p = self.p_init / 64
        elif 4000 < it <= 6000:
            p = self.p_init / 128
        elif 6000 < it <= 8000:
            p = self.p_init / 256
        elif 8000 < it:
            p = self.p_init / 512
        else:
            p = self.p_init

        return p

    def perturb(self, images, labels):

        adv = images.clone()

        batch_size = images.shape[0]

        for restart in range(self.n_restarts):

            delta = torch.zeros_like(images)

            if self.norm == "Linf":
                delta = torch.clamp(
                    delta + self.eps * self.random_choice(images.shape),
                    -self.eps,
                    self.eps,
                )

            adv_curr = torch.clamp(images + delta, 0, 1)

            margin_min, loss_min = self.margin_and_loss(adv_curr, labels)

            n_queries = torch.zeros(batch_size, device=self.device)

            for i_iter in range(self.n_queries):

                idx_to_fool = (margin_min > 0).nonzero(as_tuple=True)[0]

                if idx_to_fool.numel() == 0:
                    break

                x_curr = images[idx_to_fool]
                x_best = adv_curr[idx_to_fool]
                y_curr = labels[idx_to_fool]

                margin_curr = margin_min[idx_to_fool]
                loss_curr = loss_min[idx_to_fool]

                p = self.p_selection(i_iter)

                n_features = (
                    x_curr.shape[1] * x_curr.shape[2] * x_curr.shape[3]
                )

                s = max(
                    int(
                        round(
                            math.sqrt(
                                p * n_features / x_curr.shape[1]
                            )
                        )
                    ),
                    1,
                )

                s = min(s, x_curr.shape[2] - 1, x_curr.shape[3] - 1)

                if s <= 0:
                    continue

                vh = torch.randint(
                    0,
                    x_curr.shape[2] - s + 1,
                    (1,),
                    device=self.device,
                ).item()

                vw = torch.randint(
                    0,
                    x_curr.shape[3] - s + 1,
                    (1,),
                    device=self.device,
                ).item()

                new_deltas = torch.zeros_like(x_curr)

                new_deltas[:, :, vh:vh+s, vw:vw+s] = (
                    2.0
                    * self.eps
                    * self.random_choice(
                        (x_curr.shape[0], x_curr.shape[1], 1, 1)
                    )
                )

                x_new = torch.clamp(x_best + new_deltas, 0, 1)

                # enforce Linf constraint
                x_new = torch.max(
                    torch.min(x_new, images[idx_to_fool] + self.eps),
                    images[idx_to_fool] - self.eps,
                )

                with torch.no_grad():
                    margin, loss = self.margin_and_loss(x_new, y_curr)

                idx_improved = (loss < loss_curr).float()

                loss_min[idx_to_fool] = torch.where(
                    idx_improved.bool(), loss, loss_curr
                )

                idx_miscl = (margin <= 0).float()

                idx_improved = torch.max(idx_improved, idx_miscl)

                margin_min[idx_to_fool] = torch.where(
                    idx_improved.bool(), margin, margin_curr
                )

                idx_improved = idx_improved.view(-1, 1, 1, 1)

                adv_curr[idx_to_fool] = (
                    idx_improved * x_new
                    + (1 - idx_improved) * x_best
                )

                n_queries[idx_to_fool] += 1

                if self.verbose and idx_miscl.any():
                    print(
                        f"Iteration {i_iter+1} | "
                        f"success rate: {idx_miscl.mean().item():.2%}"
                    )

            adv = torch.where(
                (margin_min <= 0).view(-1, 1, 1, 1),
                adv_curr,
                adv,
            )

        return adv