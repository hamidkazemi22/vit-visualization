from torch.utils import model_zoo

from model.library.base import InvertModel, ModelLibrary
from datasets import image_net
from torch import nn
from torchvision.models import resnet50


def robust_resnet50():
    model = nn.DataParallel(resnet50())
    checkpoint = model_zoo.load_url('https://github.com/AminJun/PublicModels/releases/download/main/free.pt',
                                    map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    return model.module


class RobustResNet50(InvertModel):
    def __init__(self, constructor, batch_size: int):
        normalizer = image_net.normalizer
        image_size = 224
        constructor_args = {}
        name = f'{self.__class__.__name__}'

        super().__init__(name, constructor, constructor_args, image_size, batch_size, normalizer)


robust_models = ModelLibrary([
    RobustResNet50(robust_resnet50, 21),
])
