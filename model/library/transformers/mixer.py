from .base import TimmModel
from model.library.base import ModelLibrary


class Mixer(TimmModel):
    options = ['gmixer_24_224', 'mixer_b16_224', 'mixer_b16_224_in21k', 'mixer_b16_224_miil',
               'mixer_b16_224_miil_in21k', 'mixer_l16_224', 'mixer_l16_224_in21k']


mixer_models = ModelLibrary([
    Mixer(0, 20),
    Mixer(1, 13),
    Mixer(2, 12),
    Mixer(3, 13),
    Mixer(4, 13),
    Mixer(5, 4),
    Mixer(6, 7),
])
