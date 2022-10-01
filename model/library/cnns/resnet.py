from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, \
    wide_resnet50_2, wide_resnet101_2

from model.library.base import TorchVisionModel, ModelLibrary


class ResNet(TorchVisionModel):
    pass


res_nets = ModelLibrary([
    ResNet(resnet18, 72),
    ResNet(resnet34, 44),
    ResNet(resnet50, 21),
    ResNet(resnet101, 14),
    ResNet(resnet152, 14),
    ResNet(resnext50_32x4d, 11),
    ResNet(resnext101_32x8d, 8),
    ResNet(wide_resnet50_2, 13),
    ResNet(wide_resnet101_2, 8),
])
