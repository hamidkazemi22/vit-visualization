import pdb 
import torch


class VanillaBackProp:
    """
        Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_gradients(self, input_image, target_class):
        self.model(input_image)[0, target_class].backward()
        return input_image.grad.detach().clone()
