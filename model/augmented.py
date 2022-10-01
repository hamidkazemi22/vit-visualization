import torch
import torch.nn as nn


class AugmentedModel(nn.Module):
    def __init__(self, classifier: nn.Module, augmentations: nn.Module = None):
        super().__init__()
        self.augmentations = augmentations
        self.classifier = classifier.module if isinstance(classifier, nn.DataParallel) else classifier

    def forward(self, *inputs: [torch.Tensor]) -> [torch.Tensor]:
        augmented = self.augmentations(*inputs)
        return self.classifier(augmented)


class BNModel(nn.Module):
    def __init__(self, conv: nn.Module, bn: nn.Module):
        super().__init__()
        self.conv = conv
        self.bn = bn

    def forward(self, x: torch.tensor):
        return self.bn(self.conv(x))
