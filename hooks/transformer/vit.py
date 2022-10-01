import pdb

from clip.model import QuickGELU
from torch import nn
from pytorch_pretrained_vit.transformer import MultiHeadedSelfAttention, PositionWiseFeedForward
from ..base import BasicHook
import torch
from torch.nn import functional as F


class ViTHook(BasicHook):
    def __init__(self, module: nn.Module, return_output: bool, name: str):
        super().__init__(module)
        self.mode = return_output
        self.name = name

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t if not self.mode else output_t
        x = x[0] if isinstance(x, tuple) else x
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        self.activations = x


class FakeHookWrapper:
    def __init__(self, value):
        self.activations = value


class ViTAbsHookHolder(nn.Module):
    pass


class ViTAttHookHolder(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, in_feat: bool = False, keys: bool = False, queries: bool = False,
                 values: bool = False, scores: bool = False, out_feat: bool = False, sl: slice = None):
        super().__init__()
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, MultiHeadedSelfAttention)]
        self.attentions = self.just_save[sl]
        self.in_features = [ViTHook(m, False, 'in') for m in self.attentions] if in_feat else None
        self.keys = [ViTHook(a.proj_k, True, 'k') for a in self.attentions] if keys else None
        self.queries = [ViTHook(a.proj_q, True, 'q') for a in self.attentions] if queries else None
        self.value = [ViTHook(a.proj_v, True, 'v') for a in self.attentions] if values else None
        self.score_behaviour = scores
        self.out_features = [ViTHook(m, True, 'out') for m in self.attentions] if out_feat else None
        # print(in_feat, keys, queries, values, out_feat)

        self.model = classifier

    @property
    def scores(self):
        # for a in self.attentions:
        #     a.scores = None
        # return None
        return [FakeHookWrapper(a.scores) for a in self.attentions] if self.score_behaviour else None

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        # for a in self.just_save:
        #     a.scores = None
        out = None
        if x is not None:
            out = self.model(x)
        options = [self.in_features, self.keys, self.queries, self.value, self.scores, self.out_features]
        options = [[o.activations for o in l] if l is not None else None for l in options]
        names = ['in_feat', 'keys', 'queries', 'values', 'scores', 'out_feat']
        return {n: o for n, o in zip(names, options) if o is not None}, out


class ViTGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, PositionWiseFeedForward)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m.fc1, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[F.gelu(o.activations) for o in l] if l is not None else None for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out


class ClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, QuickGELU)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out


class SimpleViTGeLUHook(ViTGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        # :-1 excludes CLS token
        acts = torch.cat([(F.gelu(l.activations)[:, 1:, :]).mean(dim=1) for l in self.high], dim=-1).clone().detach()
        return acts


class SimpleClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        # :-1 excludes CLS token
        acts = torch.cat([((l.activations.transpose(0, 1))[:, 1:, :]).mean(dim=1).float() for l in self.high
                          if l.activations is not None], dim=-1).clone().detach()
        return acts


class SaliencyViTGeLUHook(ViTGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        acts = F.gelu(self.high[l].activations[:, 1:, f])
        return acts


class SaliencyClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[l].activations.transpose(0, 1)[:, 1:, f]
        return acts


class ReconstructionViTGeLUHook(ViTGeLUHook):
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        acts = F.gelu(self.high[0].activations)
        return acts


class ReconstructionClipGeLUHook(ClipGeLUHook):
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[0].activations.transpose(0, 1)
        return acts
