from torchvision.models.vgg import vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn

from model.library.base import ModelLibrary, TorchVisionModel


class VGG(TorchVisionModel):
    pass


vggs = ModelLibrary([
    VGG(vgg11, 13),
    VGG(vgg13, 12),
    VGG(vgg16, 11),
    VGG(vgg19, 10),
    VGG(vgg11_bn, 12),
    VGG(vgg13_bn, 10),
    VGG(vgg16_bn, 10),
    VGG(vgg19_bn, 9),
])
