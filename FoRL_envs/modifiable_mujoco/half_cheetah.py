import mujoco
import os
import numpy as np
from gymnasium.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
from ..utils import uniform_exclude_inner
from .utils import FoRLXMLModifierMixin

# For mojoco model parameter see https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html?highlight=mjModel#mjmodel


class ModifiableHalfCheetah(HalfCheetahEnv, FoRLXMLModifierMixin):
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
        """Render mode currently not supported"""
        assert "render_mode" not in kwargs.keys()

        if "xml_file" in kwargs.keys():
            model_path = kwargs["xml_file"]
        else:
            model_path = "half_cheetah.xml"

        if model_path.startswith("/"):
            self.original_fullpath = model_path
        else:
            self.original_fullpath = os.path.join(
                os.path.dirname(__file__), "assets", model_path
            )
        print(self.original_fullpath)
        if not os.path.exists(self.original_fullpath):
            raise OSError(f"File {self.original_fullpath} does not exist")

        super(ModifiableHalfCheetah, self).__init__(**kwargs)

    def set_env(self, mass_scaler=None, friction_scaler=None, power_scaler=None):
        with self.modify_xml(self.original_fullpath) as tree:
            if mass_scaler:
                for elem in tree.iterfind("compiler"):
                    mass = int(elem.attrib["settotalmass"])
                    elem.set("settotalmass", str(int(mass_scaler * mass)))
            if friction_scaler:
                for elem in tree.iterfind("default/geom"):
                    friction = float(elem.attrib["friction"].split(" ")[0])
                    elem.set("friction", str(friction_scaler * friction) + " .1 .1")
            if power_scaler:
                for elem in tree.iterfind("actuator/motor"):
                    gear = int(elem.attrib["gear"])
                    elem.set("gear", str(int(power_scaler * gear)))
            self._initialize_simulation()
            # self.mujoco_renderer.model = self.model
            # self.mujoco_renderer.data = self.data
            # self.mujoco_renderer.close()


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
