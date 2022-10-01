from torchvision.models.alexnet import alexnet

from model.library.base import ModelLibrary, TorchVisionModel


class AlexNet(TorchVisionModel):
    pass


alexnets = ModelLibrary([
    AlexNet(alexnet, 33),
])
