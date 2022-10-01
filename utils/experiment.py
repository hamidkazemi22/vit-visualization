import os
import datetime
import random
from argparse import Namespace
import time

from args import Args

 
def _load(file_address: str) -> int:
    with open(file_address, 'r') as f:
        for l in f:
            cur = int(l.split(' ')[0])
            if not isinstance(cur, int):
                cur = int(time.time())
            return cur
    return int(time.time())



def _save(file_address: str, cur: int):
    with open(file_address, 'w') as f:
        f.write(str(cur + 1))


def get_exp_no() -> int:
    file_address = os.path.join(os.path.split(__file__)[0], 'exp_stat.db')
    output = _load(file_address)
    _save(file_address, output)
    return output


def _two_digits(cur: str) -> str:
    return '0' + cur if len(cur) == 1 else cur


def _time():
    now = datetime.datetime.now()
    arr = [now.day, now.hour, now.minute, str(now.second)[0]]
    return ''.join([_two_digits(str(x)) for x in arr])


def _get_exp_name() -> str:
    args = Args.get_instance()
    exp_no = str(get_exp_no())
    args.get_args().exp_no = exp_no
    name_list = [exp_no, _time()]
    exp_name = args.get_args().experiment
    if str(exp_name) != 'None':
        name_list.append(exp_name)
    name_list.append(args.get_name())
    name = '_'.join(name_list)
    return name


def _fancy_print():
    print(vars(Args.get_instance().get_args()))


def _fix_random_seed(seed: int = 6247423):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def exp_starter_pack() -> (str, Namespace, dict):
    args = Args.get_instance().get_args()
    seed = 6147423 if not hasattr(args, 'seed') else args.feature
    _fix_random_seed(seed)
    exp_name = _get_exp_name()
    _fancy_print()
    return exp_name, args, vars(args)
