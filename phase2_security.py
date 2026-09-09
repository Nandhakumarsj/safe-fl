import contextlib
import torch
import torch.nn as nn


class PGD10Attacker:
    def __init__(self, eps=8 / 255, alpha=2 / 255, steps=10):
        self.eps = float(eps)
        self.alpha = float(alpha)
        self.steps = int(steps)
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, model, x, y, stream=None):
        if stream is not None and torch.cuda.is_available():
            ctx = torch.cuda.stream(stream)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            was_training = model.training
            model.eval()

            x_orig = x.detach().clone()
            delta = torch.empty_like(x_orig).uniform_(-self.eps, self.eps)
            x_adv = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

            for _ in range(self.steps):
                x_adv.requires_grad_(True)

                outputs = model(x_adv)
                loss = self.loss_fn(outputs, y)

                grad = torch.autograd.grad(
                    outputs=loss,
                    inputs=x_adv,
                    only_inputs=True,
                )[0]

                x_adv = x_adv.detach() + self.alpha * grad.sign()
                eta = torch.clamp(x_adv - x_orig, min=-self.eps, max=self.eps)
                x_adv = torch.clamp(x_orig + eta, min=0.0, max=1.0).detach()

            if was_training:
                model.train()

            return x_adv


class ByzantineAttackEngine:
    def __init__(self, num_peers=8, byzantine_indices=(6, 7), attack_type="sign_flip"):
        self.num_peers = int(num_peers)
        self.byzantine_indices = set(int(i) for i in byzantine_indices)
        self.attack_type = attack_type

    def _honest_stats(self, updates):
        honest_idx = [
            i for i in range(self.num_peers)
            if i not in self.byzantine_indices
        ]

        if len(honest_idx) == 0:
            raise ValueError("No honest peers available for Byzantine statistics.")

        honest_stack = torch.stack([updates[i] for i in honest_idx], dim=0)
        mean_honest = torch.mean(honest_stack, dim=0)
        std_honest = torch.std(honest_stack, dim=0, unbiased=False) + 1e-8

        return mean_honest, std_honest

    def poison(self, honest_updates, attack_type=None):
        attack = attack_type or self.attack_type

        if len(honest_updates) != self.num_peers:
            raise ValueError(
                f"Expected {self.num_peers} updates, got {len(honest_updates)}"
            )

        if not self.byzantine_indices:
            return [u.clone() for u in honest_updates]

        mean_honest, std_honest = self._honest_stats(honest_updates)
        poisoned = []

        for i in range(self.num_peers):
            update = honest_updates[i]

            if i not in self.byzantine_indices:
                poisoned.append(update.clone())
                continue

            if attack == "sign_flip":
                poisoned.append(-1.5 * update)

            elif attack == "gaussian_noise":
                poisoned.append(
                    mean_honest + torch.randn_like(update) * 3.0 * std_honest
                )

            elif attack == "zero_gradient":
                poisoned.append(torch.zeros_like(update))

            elif attack == "alie":
                z_max = 0.84
                poisoned.append(mean_honest - z_max * std_honest)

            elif attack == "targeted_dimension":
                poisoned_vector = update.clone()
                top_k = max(1, int(0.05 * update.numel()))
                top_k = min(top_k, update.numel())

                _, top_indices = torch.topk(std_honest, top_k)
                poisoned_vector[top_indices] = -3.0 * mean_honest[top_indices]
                poisoned.append(poisoned_vector)

            else:
                raise ValueError(f"Unsupported attack_type: {attack}")

        return poisoned
