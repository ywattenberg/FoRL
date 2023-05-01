import mujoco
import numpy as np
from xml.etree import ElementTree
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from ..utils import uniform_exclude_inner

# For mojoco model parameter see https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=mjModel#mjmodel


class ModifiableHalfCheetah(HalfCheetahEnv):
    """
    ModifiableHalfCheetah builds upon `half_cheetah_v4` and allows the modification of the totalmass and the power of the actuators

    Warning: the modification of the power is done by directly accessing the underlying mujoco simulation model,
    this is not supported by mujoco.
    If possible refrain from using this and instead modify the xml file defining the model.

    Omitted the friction part of the original Env because as far as I can the `self.friction` is not used even by the original roboschool env
    """

    RANDOM_LOWER_MASS = 0.7
    RANDOM_UPPER_MASS = 1.2
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.3

    RANDOM_LOWER_POWER = 0.7
    RANDOM_UPPER_POWER = 1.1
    EXTREME_LOWER_POWER = 0.5
    EXTREME_UPPER_POWER = 1.3

    def __init__(self, **kwargs):
        super(ModifiableHalfCheetah, self).__init__(**kwargs)
        tree = ElementTree.parse(self.fullpath)
        try:
            str_mass = tree.getroot().find("./compiler").attrib["settotalmass"]
            self.default_mass = np.float32(str_mass)
        except (KeyError, AttributeError):
            self.default_mass = np.float32(14)

    def set_actuator_gear(self, new_actuator_gear):
        assert new_actuator_gear.shape == self.model.actuator_gear.shape
        self.model.actuator_gear[:] = np.copy(new_actuator_gear)

    def set_total_mass(self, mass):
        """
        This function automatically scales the mass and with that density of the model
        for more info refer to
        `https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html?highlight=mj_setTotalmass#mj-settotalmass`
        """
        mujoco.mj_setTotalmass(self.model, np.copy(mass))


class RandomNormalHalfCheetah(ModifiableHalfCheetah):
    def __init__(self, **kwargs):
        super(RandomNormalHalfCheetah, self).__init__(**kwargs)

    def reset_model(self):
        obs = HalfCheetahEnv.reset_model(self)
        mass_scaler = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )
        force_scaler = self.np_random.uniform(
            self.RANDOM_LOWER_POWER, self.RANDOM_UPPER_POWER
        )
        new_gear = self.model.actuator_gear * force_scaler

        self.set_total_mass(mass_scaler * self.default_mass)
        self.set_actuator_gear(new_gear)
        return obs
