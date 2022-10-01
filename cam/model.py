from hooks.transformer.vit import ViTAttHookHolder
import torch
import numpy as np


@torch.no_grad()
def project_back(act: torch.tensor) -> torch.tensor:
    if act.dim() == 3:
        """
            Assuming Batch x Patch x Feature 
        """
        x_dim = np.int(np.ceil(np.sqrt(act.shape[1])))
        assert x_dim * x_dim == act.shape[1]
        return act.view(act.shape[0], x_dim, x_dim, act.shape[-1])
    elif act.dim() == 2:
        """
            Assuming Batch x Patch 
        """
        x_dim = np.int(np.ceil(np.sqrt(act.shape[1])))
        assert x_dim * x_dim == act.shape[1]
        return act.view(act.shape[0], x_dim, x_dim)
    raise NotImplementedError


class ViTPatchFeatWrapper:
    def __init__(self, hook: ViTAttHookHolder, key: str, feature: int):
        self.hook, self.key, self.feature = hook, key, feature

    @torch.no_grad()
    def __call__(self, x: torch.tensor):
        d, o = self.hook(x)
        return project_back(d[self.key][0][:, :-1, self.feature])  # -1 Removes the CLS Token!
