from .base import TimmModel
from model.library.base import ModelLibrary


class LeViT(TimmModel):
    options = ['levit_128', 'levit_128s', 'levit_192', 'levit_256', 'levit_384']
    def get_size_based_on_name(self, name:str) -> int:
        return 224


levit_models = ModelLibrary([
    LeViT(0, 114),
    LeViT(1, 136),
    LeViT(2, 92),
    LeViT(3, 63),
    LeViT(4, 56),
])
