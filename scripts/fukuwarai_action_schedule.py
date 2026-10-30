"""Hold a policy target across bounded control ticks without extra inference."""
import numpy as np


class ActionSchedule:
    def __init__(self, repeat=1):
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 1 <= repeat <= 5:
            raise ValueError('Action repeat must be an integer within [1, 5]')
        self.repeat, self.ticks, self.last = repeat, 0, None

    @property
    def needs_action(self):
        return self.last is None or self.ticks % self.repeat == 0

    def take(self, action=None):
        if self.needs_action:
            value = np.asarray(action, dtype=float)
            if value.shape != (5,) or not np.isfinite(value).all():
                raise ValueError('Expected a new finite five-joint policy target')
            self.last = value.copy()
        elif action is not None:
            raise ValueError('Cannot replace a target between scheduled policy queries')
        self.ticks += 1
        return self.last.copy()
