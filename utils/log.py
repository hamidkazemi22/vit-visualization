import torch


def _convert_form(f):
    f = f.item() if isinstance(f, torch.Tensor) else f
    if hasattr(f, '__round__') and 'bool' not in f.__class__.__name__:
        return round(f, 3)
    return f


def j_print(*args, file=None):
    array = [_convert_form(i) for i in args]
    j_header(*array, file=file)


def j_header(*args, file=None):
    plain = '{}\t' * len(args)
    print(plain.format(*args), flush=True, file=file)
