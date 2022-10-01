from .base import ModelLibrary
from .cnns import convolutional_models
from .transformers import vits

model_library = ModelLibrary([convolutional_models, vits])
