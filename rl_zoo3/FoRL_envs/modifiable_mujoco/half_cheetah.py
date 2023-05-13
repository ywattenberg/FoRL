import mujoco
import os
import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from ..utils import uniform_exclude_inner
from .utils import MujocoTrackDistSuccessMixIn

# For mojoco model parameter see https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=mjModel#mjmodel


class ModifiableHalfCheetah(HalfCheetahEnv, MujocoTrackDistSuccessMixIn):
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

    def __init__(self, **kwargs):
        super(ModifiableHalfCheetah, self).__init__(**kwargs)
        self.total_mass = int(np.sum(self.model.body_mass))

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


class RandomNormalHalfCheetah(ModifiableHalfCheetah):
    def reset_model(self):
        mass_scaler = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )
        friction_scaler = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER
        )
        power_scaler = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER
        )
        self.set_env(mass_scaler, friction_scaler, power_scaler)
        return HalfCheetahEnv.reset_model(self)


class RandomExtremeHalfCheetah(ModifiableHalfCheetah):
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
        return HalfCheetahEnv.reset_model(self)
