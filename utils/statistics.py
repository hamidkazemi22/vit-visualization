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
