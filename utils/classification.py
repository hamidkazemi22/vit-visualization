import pdb
from typing import List, Any

import torch
import numpy as np
from utils.log import j_header, j_print
from tqdm import tqdm


def get_acc(output: torch.Tensor, y: torch.Tensor) -> (float, int):
    pred = output.argmax(1)
    correct = len(pred[pred == y])
    return 100. * float(correct) / float(len(pred)), correct


def get_acc5(output: torch.Tensor, y: torch.Tensor) -> (float, int):
    pdb.set_trace()
    pred = output.argmax(1)
    correct = len(pred[pred == y])
    return 100. * float(correct) / float(len(pred)), correct


def print_acc_stats(pred: np.ndarray, y: np.ndarray):
    tr = pred[pred == y]
    fl = pred[pred != y]
    tp = len(tr[tr == True])
    tn = len(tr[tr == False])
    fp = len(fl[fl == True])
    fn = len(fl[fl == False])
    sz = len(y) / 100.
    j_header('acc', 'tp', 'tn', 'fp', 'fn')
    j_print(len(tr) / sz, tp / sz, tn / sz, fp / sz, fn / sz)


def get_confusion(e_pred: np.ndarray, e_label: np.ndarray) -> np.ndarray:
    confusion = np.zeros(shape=(100, 100))
    for i, j in zip(e_pred, e_label):
        if i != j:
            confusion[i, j] += 1
            confusion[j, i] += 1
    return confusion


def tqdm_stat(tqdm_obj: tqdm, template: str, values: List[Any]):
    styled = [str(round(v, 2)) if hasattr(v, '__round__') else v for v in values]
    tqdm_obj.set_description(template.format(*styled))
