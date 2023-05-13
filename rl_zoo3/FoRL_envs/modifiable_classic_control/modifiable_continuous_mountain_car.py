import math
import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv
from gymnasium import Env
from ..utils import uniform_exclude_inner
from gymnasium.envs.classic_control import utils


class ModifiableContinuousMountainCarEnv(Continuous_MountainCarEnv):
    """A variant of mountain car without hardcoded force/mass."""

    # Rename force to power as the actor applies a force which is then multiplied by the power
    RANDOM_LOWER_POWER = 0.0005
    RANDOM_UPPER_POWER = 0.005
    EXTREME_LOWER_POWER = 0.0001
    EXTREME_UPPER_POWER = 0.01

    RANDOM_LOWER_MASS = 0.001
    RANDOM_UPPER_MASS = 0.005
    EXTREME_LOWER_MASS = 0.0005
    EXTREME_UPPER_MASS = 0.01

    def __init__(self, *args, **kwargs):
        super(ModifiableContinuousMountainCarEnv, self).__init__(*args, **kwargs)

        self.power = 0.0015
        self.mass = 0.0025
        self.nsteps = 0

    def step(self, action: np.ndarray):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], self.min_action), self.max_action)

        velocity += force * self.power - self.mass * math.cos(3 * position)
        if velocity > self.max_speed:
            velocity = self.max_speed
        if velocity < -self.max_speed:
            velocity = -self.max_speed
        position += velocity
        if position > self.max_position:
            position = self.max_position
        if position < self.min_position:
            position = self.min_position
        if position == self.min_position and velocity < 0:
            velocity = 0

        # Convert a possible numpy bool to a Python bool.
        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        reward = 0
        if terminated:
            reward = 100.0
        reward -= math.pow(action[0], 2) * 0.1

        self.nsteps += 1
        target = 110
        if self.nsteps <= target and terminated:
            self.success = True
        else:
            self.success = False

        self.state = np.array([position, velocity], dtype=np.float32)

        if self.render_mode == "human":
            self.render()
        return self.state, reward, terminated, False, {"is_success": self.success}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, new: Optional[bool] = True):
        Env.reset(self, seed=seed, options=options)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.np_random.uniform(low=low, high=high), 0])
        self.nsteps = 0
        self.success = False
        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {"is_success": self.success}

    @property
    def parameters(self):
        parameters = super(ModifiableContinuousMountainCarEnv, self).parameters
        parameters.update(
            {
                "power": self.power,
                "mass": self.mass,
            }
        )
        return parameters


class RandomNormalContinuousMountainCar(ModifiableContinuousMountainCarEnv):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, new: Optional[bool] = True):
        # Additionally call reset of gym.Env to reset the seed
        s = super().reset(seed=seed, options=options)
        if new:
            self.power = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
            self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        return s


class RandomExtremeContinuousMountainCar(ModifiableContinuousMountainCarEnv):
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None, new: Optional[bool] = True):
        # Additionally call reset of gym.Env to reset the seed
        s = super().reset(seed=seed, options=options)
        if new:
            self.force = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_POWER,
                self.EXTREME_UPPER_POWER,
                self.RANDOM_LOWER_POWER,
                self.RANDOM_UPPER_POWER,
            )
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        return s
