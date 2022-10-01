from .base import TimmModel
from model.library.base import ModelLibrary


class PiT(TimmModel):
    options = ['pit_b_224', 'pit_b_distilled_224', 'pit_s_224', 'pit_s_distilled_224', 'pit_ti_224',
               'pit_ti_distilled_224', 'pit_xs_224', 'pit_xs_distilled_224']


pit_models = ModelLibrary([
    PiT(0, 26),
    PiT(1, 26),
    PiT(2, 78),
    PiT(3, 77),
    PiT(4, 200),
    PiT(5, 194),
    PiT(6, 149),
])
