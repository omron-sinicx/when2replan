"""Data structure for the robot command
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np


class RobotCommand(NamedTuple):
    """Robot command"""

    # vx and vy in robot coordinate
    linear_vel: np.ndarray = np.array([0.0, 0.0])

    # vyaw in robot coordinate
    angular_vel: float = 0.0

    def __str__(self):
        return "RobotCommand(linear={}, angular={})".format(
            self.linear_vel, self.angular_vel
        )

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        return RobotCommand(self.linear_vel, self.angular_vel)

    def __deepcopy__(self):
        return RobotCommand(self.linear_vel.copy(), self.angular_vel)
