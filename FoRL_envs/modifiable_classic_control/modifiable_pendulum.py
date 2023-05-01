import math
import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from gymnasium import Env
from ..base import EnvBinarySuccessMixin
from ..utils import uniform_exclude_inner


class ModifiablePendulumEnv(PendulumEnv):
    """The pendulum environment without length and mass of object hard-coded."""

    RANDOM_LOWER_MASS = 0.75
    RANDOM_UPPER_MASS = 1.25
    EXTREME_LOWER_MASS = 0.5
    EXTREME_UPPER_MASS = 1.5

    RANDOM_LOWER_LENGTH = 0.75
    RANDOM_UPPER_LENGTH = 1.25
    EXTREME_LOWER_LENGTH = 0.5
    EXTREME_UPPER_LENGTH = 1.5

    def __init__(self, *args, **kwargs):
        super(ModifiablePendulumEnv, self).__init__(*args, **kwargs)

        self.mass = 1.0
        self.length = 1.0

    def step(self, u):
        th, thdot = self.state  # th := theta
        g = self.g
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u
        angle_normalize = ((th + np.pi) % (2 * np.pi)) - np.pi
        costs = angle_normalize**2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = (
            thdot
            + (
                -3 * g / (2 * self.length) * np.sin(th + np.pi)
                + 3.0 / (self.mass * self.length**2) * u
            )
            * dt
        )
        newth = th + newthdot * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        normalized = ((newth + np.pi) % (2 * np.pi)) - np.pi

        self.state = np.array([newth, newthdot])

        # Extra calculations for is_success()
        # TODO(cpacker): be consistent in increment before or after func body
        self.nsteps += 1
        # Track how long angle has been < pi/3
        if -np.pi / 3 <= normalized and normalized <= np.pi / 3:
            self.nsteps_vertical += 1
        else:
            self.nsteps_vertical = 0
        # Success if if angle has been kept at vertical for 100 steps
        target = 100
        if self.nsteps_vertical >= target:
            # print("[SUCCESS]: nsteps is {}, nsteps_vertical is {}, reached target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, nsteps_vertical is {}, target {}".format(
            #      self.nsteps, self.nsteps_vertical, target))
            self.success = False

        # Add render code
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), -costs, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Extra state for is_success()
        self.nsteps = 0
        self.nsteps_vertical = 0
        return super(ModifiablePendulumEnv, self).reset(seed=seed, options=options)

    # @property
    # def parameters(self):
    #     return {
    #         "id": self.spec.id,
    #     }

    def is_success(self):
        """Returns True if current state indicates success, False otherwise

        Success: keep the angle of the pendulum at most pi/3 radians from
        vertical for the last 100 time steps of a trajectory with length 200
        (max_length is set to 200 in sunblaze_envs/__init__.py)
        """
        return self.success


class LightPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(LightPendulum, self).__init__(*args, **kwargs)
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def parameters(self):
        parameters = super(LightPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(HeavyPendulum, self).__init__(*args, **kwargs)
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def parameters(self):
        parameters = super(HeavyPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomHeavyPendulum, self).__init__(*args, **kwargs)
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        return super(RandomHeavyPendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomHeavyPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLightPendulum, self).__init__(*args, **kwargs)
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
        new: Optional[bool] = True
    ):
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        return super(RandomLightPendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomLightPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class ShortPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(ShortPendulum, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_LOWER_LENGTH

    @property
    def parameters(self):
        parameters = super(ShortPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class LongPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(LongPendulum, self).__init__(*args, **kwargs)
        self.length = self.EXTREME_UPPER_LENGTH

    @property
    def parameters(self):
        parameters = super(LongPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomLongPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomLongPendulum, self).__init__(*args, **kwargs)
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        if new:
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
        return super(RandomLongPendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomLongPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomShortPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomShortPendulum, self).__init__(*args, **kwargs)
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
        new: Optional[bool] = True
    ):
        if new:
            self.length = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_LENGTH,
                self.EXTREME_UPPER_LENGTH,
                self.RANDOM_LOWER_LENGTH,
                self.RANDOM_UPPER_LENGTH,
            )
        return super(RandomShortPendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomShortPendulum, self).parameters
        parameters.update(
            {
                "length": self.length,
            }
        )
        return parameters


class RandomNormalPendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomNormalPendulum, self).__init__(*args, **kwargs)
        self.mass = self.np_random.uniform(
            self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
        )
        self.length = self.np_random.uniform(
            self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
            self.length = self.np_random.uniform(
                self.RANDOM_LOWER_LENGTH, self.RANDOM_UPPER_LENGTH
            )
        return super(RandomNormalPendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomNormalPendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
            }
        )
        return parameters


class RandomExtremePendulum(ModifiablePendulumEnv):
    def __init__(self, *args, **kwargs):
        super(RandomExtremePendulum, self).__init__(*args, **kwargs)
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
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
        return super(RandomExtremePendulum, self).reset(
            seed=seed, options=options, new=new
        )

    @property
    def parameters(self):
        parameters = super(RandomExtremePendulum, self).parameters
        parameters.update(
            {
                "mass": self.mass,
                "length": self.length,
            }
        )
        return parameters
