"""Data structure for the moving obstacles
"""
from __future__ import annotations
import math

from typing import Callable, NamedTuple, Tuple
from enum import Enum, auto
import numpy as np
from skimage.draw import disk, rectangle


class Shape(Enum):
    CIRCLE = auto()
    # RECTANGLE = auto()
    # POLYGON = auto()
    # RANDOM = auto() # randomもできそう skimage
    # TODO: impl

    @staticmethod
    def from_str(shape_str: str) -> "Shape":
        if shape_str == "circle":
            return Shape.CIRCLE
        else:
            raise Exception("Unknown shape: {}".format(shape_str))


class MotionModel(Enum):
    POINT_MASS_MODEL = auto()
    SOCIAL_FORCE_MODEL = auto()
    REACTIVE_STOP_MODEL = auto()

    @staticmethod
    def from_str(motion_model_str: str) -> "MotionModel":
        if motion_model_str == "point_mass_model":
            return MotionModel.POINT_MASS_MODEL
        elif motion_model_str == "social_force_model":
            return MotionModel.SOCIAL_FORCE_MODEL
        elif motion_model_str == "reactive_stop_model":
            return MotionModel.REACTIVE_STOP_MODEL
        else:
            raise Exception("Unknown motion model: {}".format(motion_model_str))


class MovingObstacle(NamedTuple):
    # position of the obstacle in global coordinate
    pos: np.ndarray = np.array([0, 0])

    # yaw of the obstacle in global coordinate
    yaw: float = 0.0

    # linear velocity of the obstacle in obstacle coordinate
    linear_vel: np.ndarray = np.array([0.0, 0.0])

    # This is only for social force model
    target_vel: np.ndarray = np.array([0.0, 0.0])

    # angular velocity of the obstacle in obstacle coordinate
    angular_vel: float = 0.0

    # shape
    shape: Shape = Shape.CIRCLE

    # radius of the obstacle
    # if shape is CIRCLE, size[0] is radius
    size: np.ndarray = np.array([0.0, 0.0])

    # Motion
    motion_model: MotionModel = MotionModel.POINT_MASS_MODEL

    def __str__(self) -> str:
        return "MovingObstacle((x,y,yaw)={}, (vx,vy,w)={}, shape={}, size={}, motion_model={})".format(
            [self.pos[0], self.pos[1], self.yaw],
            [self.linear_vel[0], self.linear_vel[1], self.angular_vel],
            self.shape,
            self.size,
            self.motion_model,
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __copy__(self):
        return MovingObstacle(
            self.pos,
            self.yaw,
            self.linear_vel,
            self.target_vel,
            self.angular_vel,
            self.shape,
            self.size,
            self.motion_model,
        )

    def __deepcopy__(self):
        return MovingObstacle(
            self.pos.copy(),
            self.yaw,
            self.linear_vel.copy(),
            self.target_vel,
            self.angular_vel,
            self.shape,
            self.size.copy(),
            self.motion_model,
        )

    def draw_2d(
        self,
        base_map: np.ndarray,
        pose2index: Callable,
        meter2pixel: Callable,
        val: float,
    ) -> np.ndarray:
        """
        Draw the obstacle in 2D map
        Args:
            base_map: 2D map
            pose2index: pose to index mapping functor
            meter2pixel: meter to pixel mapping functor
            val : value to be drawn
        Returns:
            map: 2D map
        """
        map = base_map.copy()
        pose_ij = pose2index(self.pos)
        size_0 = meter2pixel(self.size[0])
        size_1 = meter2pixel(self.size[1])

        if self.shape == Shape.CIRCLE:
            # draw circle
            rr, cc = disk((pose_ij[0], pose_ij[1]), size_0, shape=map.shape)
            map[rr, cc] = val
        else:
            raise NotImplementedError

        return map

    def _point_mass_model(self, dt: float) -> Tuple(np.ndarray, float):
        """
        Point Mass Motion Model
        Args:
            dt: time interval
        Returns:
            new_pos: new position
            new_yaw: new yaw
        """
        rot_mat = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw)],
                [np.sin(self.yaw), np.cos(self.yaw)],
            ]
        )
        new_pos = self.pos + rot_mat.dot(self.linear_vel) * dt
        new_yaw = self.yaw + self.angular_vel * dt
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
        return new_pos, new_yaw

    def _social_force_model(
        self, target_vel: np.ndarray, obs_pos: np.ndarray
    ) -> np.ndarray:
        """
        Social Force Model, ref: https://www.apptec.co.jp/technical_report/pdf/vol27/treport_vol_27-05.pdf
        Args:
            target_vel: target vel (vx,vy)
            obs_pos: obstacle position (x,y)
        Returns:
            dv: linear acceleration
        """
        # Force to target
        tau = 0.5  # seconds
        F_target = (target_vel - self.linear_vel) / tau

        # Force to obstacle
        mass = 20.0  # kg

        radius = self.size[0]  # meter
        relative_dist = np.linalg.norm(obs_pos - self.pos)
        A = 40.0
        B = 1.0
        n_vec = self.pos - obs_pos
        F_repulsive = A * math.exp((radius - relative_dist) / B) * n_vec / mass

        dv = F_target + F_repulsive

        return dv

    def _social_forced_motion(
        self, dt: float, obs_pos: np.ndarray
    ) -> Tuple(np.ndarray, np.ndarray):
        # calculate target vel by point mass model
        rot_mat = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw)],
                [np.sin(self.yaw), np.cos(self.yaw)],
            ]
        )
        target_pos = self.pos + rot_mat.dot(self.target_vel) * dt
        new_yaw = self.yaw + self.angular_vel * dt
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
        target_vel = (target_pos - self.pos) / dt

        # calculate force
        force = self._social_force_model(target_vel, obs_pos)

        # update vel
        new_linear_vel = self.linear_vel + force * dt

        # update pos and yaw
        new_pos = self.pos + new_linear_vel * dt

        return new_pos, new_linear_vel, new_yaw

    def _reactive_stop_motion(
        self, prediction_horizon: int, dt: float, obs_pos: np.ndarray, obs_radius: float
    ) -> Tuple(np.ndarray, float):
        """
        Reactive stop motion when collision on the predictive state
        args:
            prediction_horizon [-]: prediction horizon steps for collision check
            dt [s]: time interval for prediction
        returns:
            new_pos: new position
            new_yaw: new yaw
        """

        # prediction by point mass model
        def point_mass_prediction(pos, yaw, linear_vel, angular_vel, dt):
            rot_mat = np.array(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
            )
            new_pos = pos + rot_mat.dot(linear_vel) * dt
            new_yaw = yaw + angular_vel * dt
            new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
            return new_pos, new_yaw

        predictive_pos = []
        pos = self.pos
        yaw = self.yaw
        linear_vel = self.linear_vel
        angular_vel = self.angular_vel
        for i in range(prediction_horizon):
            new_pos, new_yaw = point_mass_prediction(
                pos, yaw, linear_vel, angular_vel, dt
            )
            predictive_pos.append(new_pos)
            pos = new_pos
            yaw = new_yaw

        # check collision
        is_collision = False
        robot_radius = obs_radius
        obs_radius = self.size[0]
        for pos in predictive_pos:
            if np.linalg.norm(pos - obs_pos) < self.size[0] + robot_radius:
                is_collision = True
                break

        # if collision, reactive stop
        if is_collision:
            new_linear_vel = np.array([0.0, 0.0])
            new_angular_vel = 0.0
        else:
            new_linear_vel = self.linear_vel
            new_angular_vel = self.angular_vel

        # update pos and yaw
        rot_mat = np.array(
            [
                [np.cos(self.yaw), -np.sin(self.yaw)],
                [np.sin(self.yaw), np.cos(self.yaw)],
            ]
        )
        new_pos = self.pos + rot_mat.dot(new_linear_vel) * dt
        new_yaw = self.yaw + new_angular_vel * dt
        new_yaw = np.arctan2(np.sin(new_yaw), np.cos(new_yaw))
        return new_pos, new_yaw

    def step(
        self, dt: float, obs_pos: np.ndarray = None, obs_radius: float = None
    ) -> "MovingObstacle":
        """
        Update the obstacle position and velocity
        Args:
            dt: time interval
            obs_pos: obstacle position (x,y)
        """

        new_pos: np.ndarray = np.array([0.0, 0.0])
        new_yaw: float = 0.0

        if self.motion_model == MotionModel.POINT_MASS_MODEL:
            # point mass model
            new_pos, new_yaw = self._point_mass_model(dt)

            return MovingObstacle(
                pos=new_pos,
                yaw=new_yaw,
                linear_vel=self.linear_vel,
                angular_vel=self.angular_vel,
                size=self.size,
                shape=self.shape,
                motion_model=self.motion_model,
            )

        elif self.motion_model == MotionModel.SOCIAL_FORCE_MODEL:
            if obs_pos is None:
                raise ValueError("obs_pos must be set if you use social force model")

            new_pos, new_linear_vel, new_yaw = self._social_forced_motion(dt, obs_pos)

            return MovingObstacle(
                pos=new_pos,
                yaw=new_yaw,
                linear_vel=new_linear_vel,
                target_vel=self.target_vel,
                angular_vel=self.angular_vel,
                size=self.size,
                shape=self.shape,
                motion_model=self.motion_model,
            )

        elif self.motion_model == MotionModel.REACTIVE_STOP_MODEL:
            if obs_pos is None:
                raise ValueError("obs_pos must be set if you use reactive stop model")
            if obs_radius is None:
                raise ValueError(
                    "obs_radius must be set if you use reactive stop model"
                )

            prediction_horizon = 3  # [steps]
            new_pos, new_yaw = self._reactive_stop_motion(
                prediction_horizon, dt, obs_pos, obs_radius
            )

            return MovingObstacle(
                pos=new_pos,
                yaw=new_yaw,
                linear_vel=self.linear_vel,
                angular_vel=self.angular_vel,
                size=self.size,
                shape=self.shape,
                motion_model=self.motion_model,
            )

        else:
            raise NotImplementedError
