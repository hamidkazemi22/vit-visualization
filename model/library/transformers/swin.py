from .base import TimmModel
from model.library.base import ModelLibrary


class Swin(TimmModel):
    options = ['swin_base_patch4_window7_224', 'swin_base_patch4_window7_224_in22k', 'swin_base_patch4_window12_384',
               'swin_base_patch4_window12_384_in22k', 'swin_large_patch4_window7_224',
               'swin_large_patch4_window7_224_in22k', 'swin_large_patch4_window12_384',
               'swin_large_patch4_window12_384_in22k', 'swin_small_patch4_window7_224', 'swin_tiny_patch4_window7_224']


swin_models = ModelLibrary([
    Swin(0, 10),
    Swin(1, 9),
    Swin(2, 4),
    Swin(3, 4),
    Swin(4, 5),
    Swin(5, 5),
    Swin(6, 3),
    Swin(7, 3),
    Swin(8, 14),
    Swin(9, 23),
])
