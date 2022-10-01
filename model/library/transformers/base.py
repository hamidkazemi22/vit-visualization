import timm

from datasets import image_net
from model.augmented import AugmentedModel
from model.library.base import InvertModel
import torch


class TransformerModel(InvertModel):
    _device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, normalize: bool = True, grad: bool = False):
        model, image_size, batch_size, name = super(TransformerModel, self).__call__(normalize, grad)
        up = torch.nn.Upsample(size=(image_size, image_size), mode='bilinear', align_corners=False).cuda()
        model = AugmentedModel(model, up).to(self._device)
        return model, image_size, batch_size, name


class TimmModel(TransformerModel):
    options = []

    def get_size_based_on_name(self, name: str) -> int:
        return 384 if '384' in name else 224

    def __init__(self, o: int, batch_size: int):
        def get_from_timm(option: int = 0) -> torch.nn.Module:
            """ Hats off to: https://github.com/facebookresearch/DeiT """
            return timm.create_model(self.options[option], pretrained=True)

        image_size = self.get_size_based_on_name(self.options[o])
        super(TimmModel, self).__init__(f'{self.__class__.__name__}_{o}_{self.options[o]}', get_from_timm,
                                        {'option': o},
                                        image_size, batch_size, image_net.normalizer)
