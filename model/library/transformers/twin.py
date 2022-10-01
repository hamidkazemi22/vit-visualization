from .base import TimmModel
from model.library.base import ModelLibrary


class Twin(TimmModel):
    options = ['twins_pcpvt_base', 'twins_pcpvt_large', 'twins_pcpvt_small', 'twins_svt_base', 'twins_svt_large',
               'twins_svt_small']


twin_models = ModelLibrary([
    Twin(0, 45),
    Twin(1, 33),
    Twin(2, 69),
    Twin(3, 34),
    Twin(4, 20),
    Twin(5, 77),
])
