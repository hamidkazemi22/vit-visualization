from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, shufflenet_v2_x1_0

from model.library.base import ModelLibrary, TorchVisionModel


class ShuffleNet(TorchVisionModel):
    pass


shufflenets = ModelLibrary([
    ShuffleNet(shufflenet_v2_x0_5, 219),
    ShuffleNet(shufflenet_v2_x1_0, 129),
])
