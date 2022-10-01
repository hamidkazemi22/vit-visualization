from .vit import vit_models
from .deit import deit_models
from model.library.base import ModelLibrary
from .coat import coat_models
from .convit import convit_models
# from .levit import levit_models Cannot backward twice for some reason!
from .mixer import mixer_models
from .pit import pit_models
from .resmpl import resmlp_models
from .swin import swin_models
from .twin import twin_models
from .clip import clip_models

"""
ResNetModel_resnext50_32x4d 17
"""
vits = ModelLibrary(
    [vit_models, deit_models, coat_models, convit_models, mixer_models, pit_models, resmlp_models, swin_models,
     twin_models, clip_models])
