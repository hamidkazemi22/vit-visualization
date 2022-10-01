from .hook_holder import HookHolder


class TimedHookHolder(HookHolder):
    def get_activations(self):
        all_values = []
        for h in self.hooks:
            all_values += h.activations
        return all_values

    def get_layer(self, item):
        all_values = sorted(self.get_activations())
        return all_values[item][1]

    def set_seed(self, seed: int):
        self._broadcast('set_seed', seed)

    def set_target(self, target: list):
        self._broadcast('set_target', target)

    def reset(self):
        all_values = self.get_activations()
        all_values = sum([v.sum() for _, v in all_values])
        self._broadcast('reset')
        return all_values
