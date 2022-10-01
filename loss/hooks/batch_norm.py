from datetime import datetime

import torch
import torch.nn as nn

from .activation import AbsActivationHook


class BatchNormHookHookAbs(AbsActivationHook):
    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    @staticmethod
    def get_mean_var(x: torch.tensor) -> (torch.tensor, torch.tensor):
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1]).to('cuda:0')
        return view.mean(1), view.var(1, unbiased=False)

    @staticmethod
    def normalize_eval(model: nn.Module, x: torch.tensor) -> torch.tensor:
        extra_dim = [1] * (x.dim() - 2)
        mean = model.running_mean.data.view(1, -1, *extra_dim)
        var = model.running_var.data.view(1, -1, *extra_dim)
        return (x - mean) / var


class MatchModelBNStatsHook(BatchNormHookHookAbs):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        mean, var = self.get_mean_var(input_t)
        cur_value = torch.norm(model.running_var.data - var, 2) + torch.norm(model.running_mean.data - mean, 2)
        self.activations.append((datetime.now(), cur_value))
