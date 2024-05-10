"""Data structures for the agent state
"""
from __future__ import annotations

from typing import NamedTuple
import numpy as np


class AgentState(NamedTuple):
    """Agent state"""

    # Agent position (x, y) in global coordinate
    pos: np.ndarray = np.array([0, 0])
    # Agent yaw in global coordinate
    yaw: float = 0.0

    # Agent linear velocity (vx, vy) in robot coordinate
    linear_vel: np.ndarray = np.array([0.0, 0.0])
    # Agent angular velocity (w) in robot coordinate
    angular_vel: float = 0.0

    def __str__(self) -> str:
        return "AgentState((x,y,yaw)={}, (vx,vy,w)={})".format(
            [self.pos[0], self.pos[1], self.yaw],
            [self.linear_vel[0], self.linear_vel[1], self.angular_vel],
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self):
        return AgentState(self.pos, self.yaw, self.linear_vel, self.angular_vel)

    def __deepcopy__(self):
        return AgentState(
            self.pos.copy(), self.yaw, self.linear_vel.copy(), self.angular_vel
        )
