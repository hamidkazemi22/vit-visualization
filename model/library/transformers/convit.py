from .base import TimmModel
from model.library.base import ModelLibrary


class ConViT(TimmModel):
    options = ['convit_base', 'convit_small', 'convit_tiny']


convit_models = ModelLibrary([
    ConViT(0, 25),
    ConViT(1, 22),
    ConViT(2, 113),
])
