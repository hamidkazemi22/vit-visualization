import torch
import torch.nn as nn
# from advertorch.attacks import LinfPGDAttack


class Modifier:
    def __call__(self, *inputs) -> torch.tensor:
        return self.attack(*inputs)

    def attack(self, *inputs) -> torch.tensor:
        raise NotImplementedError

"""
class LInfPGDAttacker(Modifier):
    def __init__(self, model: nn.Module, loss: torch.nn.Module):
        self.attacker = LinfPGDAttack(model, loss, eps=8. / 255., nb_iter=7, eps_iter=2. / 255., rand_init=True,
                                      clip_min=0., clip_max=1., targeted=False)

    def attack(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        with torch.enable_grad():
            return self.attacker.perturb(x, y)
"""
