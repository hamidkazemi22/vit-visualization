import torch
from pytorch_pretrained_vit import ViT as ActualVit

from datasets import weird_image_net
from model.library.base import ModelLibrary
from .base import TransformerModel


class ViT(TransformerModel):
    def __init__(self, o: int, batch_size: int):
        vit_options = [
            'B_16_imagenet1k',
            'B_32_imagenet1k',
            'L_16_imagenet1k',
            'L_32_imagenet1k',
            'B_16',
            'B_32',
            'L_32',
            'L_16',
        ]

        def get_vit(option: int = 0) -> torch.nn.Module:
            """ Hats off to: https://github.com/lukemelas/PyTorch-Pretrained-ViT """
            return ActualVit(vit_options[option], pretrained=True)

        image_size = 384 if 'imagenet1k' in vit_options[o] else 224
        super().__init__(f'ViT{o}_{vit_options[o]}', get_vit, {'option': o}, image_size, batch_size,
                         weird_image_net.normalizer)


vit_models = ModelLibrary(other_models=[
    ViT(0, 6),
    ViT(1, 15),
    ViT(2, 2),
    ViT(3, 5),
    ViT(4, 12),
    ViT(5, 18),
    ViT(6, 6),
])
