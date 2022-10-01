from torchvision.models.inception import inception_v3

from model.library.base import ModelLibrary, TorchVisionModel


class Inception(TorchVisionModel):
    pass


inceptions = ModelLibrary([
    Inception(inception_v3, 38),
])
