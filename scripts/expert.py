"""Expert actor for Imitation Learning."""

from __future__ import annotations

import numpy as np


class ExpertActor:
    def __init__(self, seed: int, max_hz: int) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._max_hz = max_hz
        self._time_counter = int(self._max_hz * self._rng.uniform(1.0, 10.0))
        if self._time_counter < self._max_hz:
            self._time_counter = self._max_hz

    def act(self, obs: np.ndarray) -> int:
        """Expert action for imitation learning."""
        action = self._random_timer_act()
        return action

    def _random_timer_act(self) -> int:
        if self._time_counter == 0:
            self._time_counter = int(self._max_hz * self._rng.uniform(1.0, 10.0))
            if self._time_counter < self._max_hz:
                self._time_counter = self._max_hz
            # Replan
            return 1
        else:
            self._time_counter -= 1
            # Not replan
            return 0
