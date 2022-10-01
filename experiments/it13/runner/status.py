from torch.utils.data import DataLoader
from tqdm import tqdm


class ItemIterator:
    @property
    def iterator_item(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.iterator_item)

    def __getitem__(self, item):
        return self.iterator_item[item]

    def __len__(self):
        return len(self.iterator_item)


class Status(ItemIterator):
    @property
    def iterator_item(self):
        return self.iterable

    def __init__(self, loader: DataLoader, num_elements: int, template: str = None):
        self.iterable = tqdm(loader)
        self.template = template or (', '.join(['{}'] * num_elements))

    def print(self, *values):
        styled = [str(round(v, 2)) if hasattr(v, '__round__') else v for v in values]
        self.iterable.set_description(self.template.format(*styled))


class Meter:
    def __init__(self):
        self.correct = 0
        self.count = 0

    def reset(self):
        self.correct = self.count = 0

    def __add__(self, other: (float, float)):
        self.correct += other[0]
        self.count += other[1]
        return self

    @property
    def value(self) -> float:
        return self.correct / self.count
