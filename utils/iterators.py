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