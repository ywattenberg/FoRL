import math
import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.acrobot import AcrobotEnv
from gymnasium import Env
from ..base import EnvBinarySuccessMixin
from ..utils import uniform_exclude_inner


class ModifiableAcrobotEnv(AcrobotEnv):
    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_LENGTH = 0.75
    RANDOM_UPPER_LENGTH = 1.25
    EXTREME_LOWER_LENGTH = 0.5
    EXTREME_UPPER_LENGTH = 1.5

    RANDOM_LOWER_INERTIA = 0.75
    RANDOM_UPPER_INERTIA = 1.25
    EXTREME_LOWER_INERTIA = 0.5
    EXTREME_UPPER_INERTIA = 1.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.nsteps = 0
        self.success = False

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.nsteps = 0
        self.success = False
        return super(ModifiableAcrobotEnv, self).reset(seed=seed, options=options)

    # @property
    # def parameters(self):
    #     return {
    #         "id": self.spec.id,
    #     }

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: swing the end of the second link to the desired height within
        90 time steps
        """
        return self.success

    def step(self, a):
        state, reward, terminated, _, info = super().step(a)
        if self.nsteps <= 90 and terminated:
            self.success = True
        else:
            self.success = False
        info["is_success"] = self.is_success()
        return state, reward, terminated, False, info


class LightAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(LightAcrobot, self).__init__(*args, **kwargs)
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(LightAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(HeavyAcrobot, self).__init__(*args, **kwargs)
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(HeavyAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomHeavyAcrobot, self).__init__(*args, **kwargs)
        self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        return super(RandomHeavyAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(RandomHeavyAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLightAcrobot, self).__init__(*args, **kwargs)
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        return super(RandomLightAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def parameters(self):
        parameters = super(RandomLightAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class ShortAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(ShortAcrobot, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_LOWER_LENGTH

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(ShortAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(LongAcrobot, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_UPPER_LENGTH

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(LongAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLongAcrobot, self).__init__(*args, **kwargs)
        self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        return super(RandomLongAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(RandomLongAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomShortAcrobot, self).__init__(*args, **kwargs)
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
        return super(RandomShortAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def parameters(self):
        parameters = super(RandomShortAcrobot, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LowInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(LowInertiaAcrobot, self).__init__(*args, **kwargs)
        self.inertia = self.EXTREME_LOWER_INERTIA

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(LowInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class HighInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(HighInertiaAcrobot, self).__init__(*args, **kwargs)
        self.inertia = self.EXTREME_UPPER_INERTIA

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(HighInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomHighInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomHighInertiaAcrobot, self).__init__(*args, **kwargs)
        self.inertia = self.np_random.uniform(self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.inertia = self.np_random.uniform(self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)
        return super(RandomHighInertiaAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(RandomHighInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomLowInertiaAcrobot(ModifiableAcrobotEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLowInertiaAcrobot, self).__init__(*args, **kwargs)
        self.inertia = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_INERTIA,
            self.EXTREME_UPPER_INERTIA,
            self.RANDOM_LOWER_INERTIA,
            self.RANDOM_UPPER_INERTIA,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.inertia = self.np_random.uniform(
                self.np_random.uniform,
                self.EXTREME_LOWER_INERTIA,
                self.EXTREME_UPPER_INERTIA,
                self.RANDOM_LOWER_INERTIA,
                self.RANDOM_UPPER_INERTIA,
            )
        return super(RandomLowInertiaAcrobot, self).reset(seed=seed, options=options)

    @property
    def LINK_MOI(self):
        return self.inertia

    @property
    def parameters(self):
        parameters = super(RandomLowInertiaAcrobot, self).parameters
        parameters.update(
            {
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomNormalAcrobot(ModifiableAcrobotEnv):
    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self, *args, **kwargs):
        super(RandomNormalAcrobot, self).__init__(*args, **kwargs)
        self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
        self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
        self.inertia = self.np_random.uniform(self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.mass = self.np_random.uniform(self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS)
            self.length = self.np_random.uniform(self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH)
            self.inertia = self.np_random.uniform(self.RANDOM_LOWER_INERTIA, self.RANDOM_UPPER_INERTIA)
        # reset just resets .state
        return super(RandomNormalAcrobot, self).reset(seed=seed, options=options)

    @property
    def parameters(self):
        parameters = super(RandomNormalAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
                "inertia": self.inertia,
            }
        )
        return parameters


class RandomExtremeAcrobot(ModifiableAcrobotEnv):
    @property
    def LINK_MASS_1(self):
        return self.mass

    @property
    def LINK_MASS_2(self):
        return self.mass

    @property
    def LINK_LENGTH_1(self):
        return self.length

    @property
    def LINK_LENGTH_2(self):
        return self.length

    @property
    def LINK_MOI(self):
        return self.inertia

    def __init__(self, *args, **kwargs):
        super(RandomExtremeAcrobot, self).__init__(*args, **kwargs)
        self.mass = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASS,
            self.EXTREME_UPPER_MASS,
            self.RANDOM_LOWER_MASS,
            self.RANDOM_UPPER_MASS,
        )
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self.inertia = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_INERTIA,
            self.EXTREME_UPPER_INERTIA,
            self.RANDOM_LOWER_INERTIA,
            self.RANDOM_UPPER_INERTIA,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True,
    ):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self.inertia = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_INERTIA,
                self.EXTREME_UPPER_INERTIA,
                self.RANDOM_LOWER_INERTIA,
                self.RANDOM_UPPER_INERTIA,
            )
        # reset just resets .state
        return super(RandomExtremeAcrobot, self).reset(seed=seed, options=options)

    @property
    def parameters(self):
        parameters = super(RandomExtremeAcrobot, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
                "inertia": self.inertia,
            }
        )
        return parameters
