import os
from typing import Any

import torch

from utils.experiment import _get_exp_name
import torchvision


class AbstractSaver:
    def __init__(self, extension: str, folder: str = None, save_id: bool = False):
        self.folder = _get_exp_name() if folder is None else folder
        self._id = 0
        self.extension = extension
        self.folder = '{}/{}_{}'.format(self.folder, self.extension, self.__class__.__name__)
        self.save_id = save_id

    def _get_mkdir_path(self, *path):
        path = [str(folder) for folder in path]
        to_join = '_'.join(path)
        child = '{}_{}'.format(self._id, to_join) if self.save_id else to_join
        par = os.path.join('desktop', self.folder)
        os.makedirs(par, exist_ok=True)
        return "{}/{}.{}".format(par, child, self.extension)

    def save(self, result: torch.Tensor, *path):
        self.save_function(result, self._get_mkdir_path(*path))
        self._id += 1

    def save_function(self, result: Any, path: str):
        raise NotImplementedError


class ImageSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        # torchvision.utils.save_image(result, path, nrow=self.nrow[len(result) - 1])
        torchvision.utils.save_image(result, path, nrow=1)

    @staticmethod
    def get_nrow() -> torch.tensor:
        bs = torch.arange(1, 1000)
        p = bs.view(1, -1).repeat(len(bs), 1)
        q = (bs.view(-1, 1) / p).floor().int()
        feasible = ((p * q) == bs.view(-1, 1))
        sm = p + q
        sm[feasible == False] = (len(bs) + 1) * 10
        return p[torch.arange(len(bs)), sm.argmin(dim=-1)]

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('png', folder, save_id)
        self.nrow = self.get_nrow()


class TensorSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        torch.save(result, path)

    def __init__(self, folder: str, save_id: bool = False):
        super().__init__('pth', folder, save_id)


class ExperimentSaver(AbstractSaver):
    def save_function(self, result: Any, path: str):
        pass

    def save(self, result: torch.Tensor, *path):
        self.image.save(result, *path)
        if not self.disk_saver:
            self.tensor.save(result, *path)

    def __init__(self, folder: str, save_id: bool = False, disk_saver: bool = False):
        super().__init__('none', folder, save_id)
        self.image = ImageSaver(folder=folder, save_id=save_id)
        self.disk_saver = disk_saver
        self.tensor = TensorSaver(folder=folder, save_id=save_id)


""" 


 sq = int(np.ceil(np.sqrt(batch_size)) + 1)
        p = torch.arange(1, sq)
        squares = (((batch_size / p).ceil().int() * p) == batch_size)
        xs = p[squares][-1].item()
        ys = batch_size // xs

"""
