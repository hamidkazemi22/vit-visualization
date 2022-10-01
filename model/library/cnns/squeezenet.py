from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1

from model.library.base import ModelLibrary, TorchVisionModel


class SqueezeNet(TorchVisionModel):
    pass


squeezenets = ModelLibrary([
    SqueezeNet(squeezenet1_0, 80),
    SqueezeNet(squeezenet1_1, 126),
])
