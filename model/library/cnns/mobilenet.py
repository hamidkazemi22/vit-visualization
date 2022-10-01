from torchvision.models.mobilenet import mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small

from model.library.base import TorchVisionModel, ModelLibrary


class MobileNet(TorchVisionModel):
    pass


mobilenets = ModelLibrary([
    MobileNet(mobilenet_v2, 48),
    MobileNet(mobilenet_v3_large, 60),
    MobileNet(mobilenet_v3_small, 157),
])
