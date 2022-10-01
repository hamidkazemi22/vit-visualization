from .base import TimmModel
from model.library.base import ModelLibrary


class CoaT(TimmModel):
    options = ['coat_lite_mini', 'coat_lite_small', 'coat_lite_tiny', 'coat_mini', 'coat_tiny']


coat_models = ModelLibrary([
    CoaT(0, 25),
    CoaT(1, 12),
    CoaT(2, 27),
    CoaT(3, 1),
    CoaT(4, 1),
])
