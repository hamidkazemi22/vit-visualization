from torchvision.models.mnasnet import mnasnet0_5, mnasnet1_0

from model.library.base import ModelLibrary, TorchVisionModel


class MNasNet(TorchVisionModel):
    pass


mnasnets = ModelLibrary([
    MNasNet(mnasnet0_5, 97),
    MNasNet(mnasnet1_0, 56),
])
