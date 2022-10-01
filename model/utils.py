from torch import nn as nn

from .augmented import BNModel
from datasets.imagenet import image_net
from loss import BaseFakeBN
from torchvision.models import resnet18


def default_bn():
    model = resnet18(pretrained=True)
    bn = BNModel(model.conv1, nn.BatchNorm2d(model.conv1.out_channels)).cuda()
    bn = BaseFakeBN(bn, image_net, 'imagenet_0.pth').cuda()
    bn.eval()
