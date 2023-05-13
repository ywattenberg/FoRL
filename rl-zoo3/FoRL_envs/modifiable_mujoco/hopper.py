import mujoco
from os import path
import numpy as np
from gymnasium.envs.mujoco.hopper_v4 import HopperEnv
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from ..utils import uniform_exclude_inner
from .utils import MujocoTrackDistSuccessMixIn
from gymnasium import utils
from gymnasium.spaces import Box

# For mojoco model parameter see https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=mjModel#mjmodel


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 3.0,
    "lookat": np.array((0.0, 0.0, 1.15)),
    "elevation": -20.0,
}


class ModifiableHopper(HopperEnv, MujocoTrackDistSuccessMixIn):
    """
    ModifiableHalfCheetah builds upon `half_cheetah_v4` and allows the modification of the totalmass and the power of the actuators

    Warning: the modification of the power is done by directly accessing the underlying mujoco simulation model,
    this is not supported by mujoco.
    If possible refrain from using this and instead modify the xml file defining the model.

    Omitted the friction part of the original Env because as far as I can the `self.friction` is not used even by the original roboschool env
    """

    # These are scaling factors as opposed to the factors used the original environment
    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_FRICTION = 0.75
    RANDOM_UPPER_FRICTION = 1.25
    EXTREME_LOWER_FRICTION = 0.5
    EXTREME_UPPER_FRICTION = 1.5

    RANDOM_LOWER_POWER = 0.75
    RANDOM_UPPER_POWER = 1.25
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.5

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=1e-3,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_state_range=(-100.0, 100.0),
        healthy_z_range=(0.7, float("inf")),
        healthy_angle_range=(-0.2, 0.2),
        reset_noise_scale=5e-3,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_state_range,
            healthy_z_range,
            healthy_angle_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._healthy_state_range = healthy_state_range
        self._healthy_z_range = healthy_z_range
        self._healthy_angle_range = healthy_angle_range

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        if exclude_current_positions_from_observation:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        else:
            observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            path.join(path.dirname(__file__), "assets/hopper.xml"),
            4,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def set_env(self, mass_scaler=None, friction_scaler=None, power_scaler=None):
        if mass_scaler:
            new_mass = int(self.total_mass * mass_scaler)
            mujoco.mj_setTotalmass(self.model, new_mass)
        if friction_scaler:
            friction = np.copy(self.model.geom_friction)
            self.model.geom_friction[:, 0] = friction[:, 0] * friction_scaler
        if power_scaler:
            self.model.actuator_gear = np.copy(self.model.actuator_gear * power_scaler)

    def step(self, action):
        observation, reward, terminated, _, info = super().step(action)
        info["is_success"] = self.is_success()
        return observation, reward, terminated, False, info


class RandomNormalHopper(ModifiableHopper):
    def reset_model(self):
        mass_scaler = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        friction_scaler = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        power_scaler = self.np_random.uniform(self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER)
        self.set_env(mass_scaler, friction_scaler, power_scaler)
        return HopperEnv.reset_model(self)


class RandomExtremeHopper(ModifiableHopper):
    def reset_model(self):
        mass_scaler = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )
        friction_scaler = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FRICTION,
            self.EXTREME_UPPER_FRICTION,
            self.RANDOM_LOWER_FRICTION,
            self.RANDOM_UPPER_FRICTION,
        )
        power_scaler = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_POWER,
            self.EXTREME_UPPER_POWER,
            self.RANDOM_LOWER_POWER,
            self.RANDOM_UPPER_POWER,
        )
        self.set_env(mass_scaler, friction_scaler, power_scaler)
        return HopperEnv.reset_model(self)
