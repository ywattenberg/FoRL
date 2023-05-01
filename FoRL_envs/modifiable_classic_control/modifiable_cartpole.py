import math
import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from gymnasium import Env
from ..base import EnvBinarySuccessMixin
from ..utils import uniform_exclude_inner


class ModifiableCartPoleEnv(CartPoleEnv, EnvBinarySuccessMixin):
    RANDOM_LOWER_FORCE_MAG = 5.0
    RANDOM_UPPER_FORCE_MAG = 15.0
    EXTREME_LOWER_FORCE_MAG = 1.0
    EXTREME_UPPER_FORCE_MAG = 20.0

    RANDOM_LOWER_LENGTH = 0.25
    RANDOM_UPPER_LENGTH = 0.75
    EXTREME_LOWER_LENGTH = 0.05
    EXTREME_UPPER_LENGTH = 1.0

    RANDOM_LOWER_MASSPOLE = 0.05
    RANDOM_UPPER_MASSPOLE = 0.5
    EXTREME_LOWER_MASSPOLE = 0.01
    EXTREME_UPPER_MASSPOLE = 1.0

    def _followup(self):
        """Cascade values of new (variable) parameters"""
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        """new is a boolean variable telling whether to regenerate the environment parameters"""
        """Default is to just ignore it"""
        self.nsteps = 0
        return super(ModifiableCartPoleEnv, self).reset(seed=seed, options=options)

    # @property
    # def parameters(self):
    #     return {
    #         "id": self.spec.id,
    #     }

    def step(self, *args, **kwargs):
        """Wrapper to increment new variable nsteps"""
        self.nsteps += 1
        return super().step(*args, **kwargs)

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        Balance for at least 195 time steps ("definition" of success in Gym:
        https://github.com/openai/gym/wiki/CartPole-v0#solved-requirements)
        """
        target = 195
        if self.nsteps >= target:
            # print("[SUCCESS]: nsteps is {}, reached target {}".format(
            #      self.nsteps, target))
            return True
        else:
            # print("[NO SUCCESS]: nsteps is {}, target {}".format(
            #      self.nsteps, target))
            return False


class StrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(StrongPushCartPole, self).__init__(*args, **kwargs)
        self.force_mag = self.EXTREME_UPPER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(StrongPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class WeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(WeakPushCartPole, self).__init__(*args, **kwargs)
        self.force_mag = self.EXTREME_LOWER_FORCE_MAG

    @property
    def parameters(self):
        parameters = super(WeakPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class RandomStrongPushCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomStrongPushCartPole, self).__init__(*args, **kwargs)
        self.force_mag = self.np_random.uniform(
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
            )
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomStrongPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class RandomWeakPushCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomWeakPushCartPole, self).__init__(*args, **kwargs)
        self.force_mag = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FORCE_MAG,
            self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG,
            self.RANDOM_UPPER_FORCE_MAG,
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE_MAG,
                self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG,
                self.RANDOM_UPPER_FORCE_MAG,
            )
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomWeakPushCartPole, self).parameters
        parameters.update(
            {
                "force": self.force_mag,
            }
        )
        return parameters


class ShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(ShortPoleCartPole, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_LOWER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(ShortPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(LongPoleCartPole, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_UPPER_LENGTH
        self._followup()

    @property
    def parameters(self):
        parameters = super(LongPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLongPoleCartPole, self).__init__(*args, **kwargs)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )
        self._followup()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomLongPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomShortPoleCartPole, self).__init__(*args, **kwargs)
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self._followup()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomShortPoleCartPole, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(LightPoleCartPole, self).__init__(*args, **kwargs)
        self.masspole = self.EXTREME_LOWER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(LightPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class HeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(HeavyPoleCartPole, self).__init__(*args, **kwargs)
        self.masspole = self.EXTREME_UPPER_MASSPOLE
        self._followup()

    @property
    def parameters(self):
        parameters = super(HeavyPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomHeavyPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomHeavyPoleCartPole, self).__init__(*args, **kwargs)
        self.masspole = self.np_random.uniform(
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
        )
        self._followup()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = self.np_random.uniform(
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomHeavyPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomLightPoleCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLightPoleCartPole, self).__init__(*args, **kwargs)
        self.masspole = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASSPOLE,
            self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE,
            self.RANDOM_UPPER_MASSPOLE,
        )
        self._followup()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        Env.reset(self, seed=seed)
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.masspole = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASSPOLE,
                self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE,
                self.RANDOM_UPPER_MASSPOLE,
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomLightPoleCartPole, self).parameters
        parameters.update(
            {
                "mass": self.masspole,
            }
        )
        return parameters


class RandomNormalCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomNormalCartPole, self).__init__(*args, **kwargs)
        self.force_mag = self.np_random.uniform(
            self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
        )
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )
        self.masspole = self.np_random.uniform(
            self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
        )
        self._followup()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        Env.reset(self, seed=seed)
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        if new:
            self.force_mag = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE_MAG, self.RANDOM_UPPER_FORCE_MAG
            )
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
            self.masspole = self.np_random.uniform(
                self.RANDOM_LOWER_MASSPOLE, self.RANDOM_UPPER_MASSPOLE
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomNormalCartPole, self).parameters
        # parameters.update({'force_mag': self.force_mag, 'length': self.length, 'masspole': self.masspole, })
        parameters.update(
            {
                "force_mag": self.force_mag,
                "length": self.length,
                "masspole": self.masspole,
                "total_mass": self.total_mass,
                "polemass_length": self.polemass_length,
            }
        )
        return parameters


class RandomExtremeCartPole(ModifiableCartPoleEnv):
    def __init__(self, *args, **kwargs):
        super(RandomExtremeCartPole, self).__init__(*args, **kwargs)
        """
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        """
        self.force_mag = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_FORCE_MAG,
            self.EXTREME_UPPER_FORCE_MAG,
            self.RANDOM_LOWER_FORCE_MAG,
            self.RANDOM_UPPER_FORCE_MAG,
        )
        self.length = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_LENGTH,
            self.EXTREME_UPPER_LENGTH,
            self.RANDOM_LOWER_LENGTH,
            self.RANDOM_UPPER_LENGTH,
        )
        self.masspole = uniform_exclude_inner(
            self.np_random.uniform,
            self.EXTREME_LOWER_MASSPOLE,
            self.EXTREME_UPPER_MASSPOLE,
            self.RANDOM_LOWER_MASSPOLE,
            self.RANDOM_UPPER_MASSPOLE,
        )

        self._followup()
        # NOTE(cpacker): even though we're just changing the above params,
        # we still need to regen the other var dependencies
        # We need to scan through the other methods to make sure the same
        # mistake isn't being made

        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5 # actually half the pole's length
        # self.polemass_length = (self.masspole * self.length)
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        Env.reset(self, seed=seed)
        self.nsteps = 0  # for super.is_success()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        """
        self.force_mag = self.np_random.uniform(self.LOWER_FORCE_MAG, self.UPPER_FORCE_MAG)
        self.length = self.np_random.uniform(self.LOWER_LENGTH, self.UPPER_LENGTH)
        self.masspole = self.np_random.uniform(self.LOWER_MASSPOLE, self.UPPER_MASSPOLE)
        """
        if new:
            self.force_mag = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE_MAG,
                self.EXTREME_UPPER_FORCE_MAG,
                self.RANDOM_LOWER_FORCE_MAG,
                self.RANDOM_UPPER_FORCE_MAG,
            )
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
            self.masspole = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASSPOLE,
                self.EXTREME_UPPER_MASSPOLE,
                self.RANDOM_LOWER_MASSPOLE,
                self.RANDOM_UPPER_MASSPOLE,
            )
            self._followup()
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomExtremeCartPole, self).parameters
        parameters.update(
            {
                "force_mag": self.force_mag,
                "length": self.length,
                "masspole": self.masspole,
                "total_mass": self.total_mass,
                "polemass_length": self.polemass_length,
            }
        )
        return parameters
