from .base import TimmModel
from model.library.base import ModelLibrary


class ResMLP(TimmModel):
    options = ['resmlp_12_224', 'resmlp_12_distilled_224', 'resmlp_24_224', 'resmlp_24_distilled_224', 'resmlp_36_224',
               'resmlp_36_distilled_224', 'resmlp_big_24_224', 'resmlp_big_24_224_in22ft1k',
               'resmlp_big_24_distilled_224']


resmlp_models = ModelLibrary([
    ResMLP(0, 39),
    ResMLP(1, 39),
    ResMLP(2, 20),
    ResMLP(3, 20),
    ResMLP(4, 14),
    ResMLP(5, 14),
    ResMLP(6, 3),
    ResMLP(7, 3),
    ResMLP(8, 3),
])
