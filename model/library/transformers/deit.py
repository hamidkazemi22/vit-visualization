import torch

from datasets import image_net
from model.library.base import ModelLibrary
from .base import TransformerModel


class DeiT(TransformerModel):
    def __init__(self, o: int, batch_size: int):
        options = [
            'deit_base_patch16_224',
            'deit_base_distilled_patch16_384',
            'deit_base_patch16_384',
            'deit_tiny_distilled_patch16_224',
            'deit_small_distilled_patch16_224',
            'deit_base_distilled_patch16_224',
        ]

        def get_DeiT(option: int = 0) -> torch.nn.Module:
            """ Hats off to: https://github.com/facebookresearch/DeiT """
            return torch.hub.load('facebookresearch/DeiT:main', options[option], pretrained=True)

        image_size = 224 if '224' in options[o] else 384
        super(DeiT, self).__init__(f'DeiT_{o}_{options[o]}', get_DeiT, {'option': o}, image_size, batch_size,
                                   image_net.normalizer)


deit_models = ModelLibrary([
    DeiT(0, 11),
    DeiT(1, 5),
    DeiT(2, 5),
    DeiT(3, 62),
    DeiT(4, 28),
    DeiT(5, 11),
])
