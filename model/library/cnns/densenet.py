from torchvision.models.densenet import densenet121, densenet161, densenet169, densenet201

from model.library.base import ModelLibrary, TorchVisionModel


class DensNet(TorchVisionModel):
    pass


dense_nets = ModelLibrary([
    DensNet(densenet121, 24),
    DensNet(densenet161, 12),
    DensNet(densenet169, 20),
    DensNet(densenet201, 16),
])
