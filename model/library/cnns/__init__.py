from .alexnet import alexnets
from .densenet import dense_nets
from .googlenet import googlenets
from .inception import inceptions
from .mobilenet import mobilenets
from .mnasnet import mnasnets
from .resnet import res_nets
from .shufflenet import shufflenets
from .squeezenet import squeezenets
from .vggs import vggs

from model.library.base import ModelLibrary
from .robust import robust_models

convolutional_models = ModelLibrary(
    [alexnets, dense_nets, googlenets, inceptions, mobilenets, mnasnets, res_nets, shufflenets, squeezenets, vggs,
     robust_models
     ])
