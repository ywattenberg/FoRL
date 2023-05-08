import math
import numpy as np
from typing import Optional
from gymnasium.envs.classic_control.mountain_car import MountainCarEnv
from gymnasium import Env
from ..base import EnvBinarySuccessMixin
from ..utils import uniform_exclude_inner


class ModifiableMountainCarEnv(MountainCarEnv):
    """A variant of mountain car without hardcoded force/mass."""

    RANDOM_LOWER_FORCE = 0.0005
    RANDOM_UPPER_FORCE = 0.005
    EXTREME_LOWER_FORCE = 0.0001
    EXTREME_UPPER_FORCE = 0.01

    RANDOM_LOWER_MASS = 0.001
    RANDOM_UPPER_MASS = 0.005
    EXTREME_LOWER_MASS = 0.0005
    EXTREME_UPPER_MASS = 0.01

    def __init__(self, *args, **kwargs):
        super(ModifiableMountainCarEnv, self).__init__(*args, **kwargs)

        self.force = 0.001
        self.mass = 0.0025

    def step(self, action):
        """Rewritten to remove hard-coding of values in original code"""
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        position, velocity = self.state
        velocity += (action - 1) * self.force + math.cos(3 * position) * (-self.mass)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(
            position >= self.goal_position and velocity >= self.goal_velocity
        )
        reward = -1.0

        # New additions to support is_success()
        self.nsteps += 1
        target = 110
        if self.nsteps <= target and terminated:
            # print("[SUCCESS]: nsteps is {}, done before target {}".format(
            #      self.nsteps, target))
            self.success = True
        else:
            # print("[NO SUCCESS]: nsteps is {}, not done before target {}".format(
            #      self.nsteps, target))
            self.success = False
        ###

        self.state = (position, velocity)

        # Add suppot for rendering in human mode
        if self.render_mode == "human":
            self.render()  # Calling render() will call super().render()

        return (
            np.array(self.state, dtype=np.float32),
            reward,
            terminated,
            False,
            {"is_success": self.success},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        self.nsteps = 0
        return super(ModifiableMountainCarEnv, self).reset(seed=seed, options=options)

    # @property
    # def parameters(self):
    #     return {
    #         "id": self.spec.id,
    #     }

    def is_success(self):
        """Returns True is current state indicates success, False otherwise
        get to the top of the hill within 110 time steps (definition of success in Gym)

        MountainCar sets done=True once the car reaches the "top of the hill",
        so we can just check if done=True and nsteps<=110. See:
        https://github.com/openai/gym/blob/0ccb08dfa1535624b45645e141af9398e2eba416/gym/envs/classic_control/mountain_car.py#L49
        """
        # NOTE: Moved logic to step()
        return self.success


class WeakForceMountainCar(ModifiableMountainCarEnv):
    def __init__(self, *args, **kwargs):
        super(WeakForceMountainCar, self).__init__(*args, **kwargs)
        self.force = self.EXTREME_LOWER_FORCE

    @property
    def parameters(self):
        parameters = super(WeakForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class StrongForceMountainCar(ModifiableMountainCarEnv):
    def __init__(self, *args, **kwargs):
        super(StrongForceMountainCar, self).__init__(*args, **kwargs)
        self.force = self.EXTREME_UPPER_FORCE

    @property
    def parameters(self):
        parameters = super(StrongForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class RandomStrongForceMountainCar(ModifiableMountainCarEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        if new:
            self.force = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomStrongForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class RandomWeakForceMountainCar(ModifiableMountainCarEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        if new:
            self.force = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE,
                self.EXTREME_UPPER_FORCE,
                self.RANDOM_LOWER_FORCE,
                self.RANDOM_UPPER_FORCE,
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomWeakForceMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
            }
        )
        return parameters


class LightCarMountainCar(ModifiableMountainCarEnv):
    def __init__(self, *args, **kwargs):
        super(LightCarMountainCar, self).__init__(*args, **kwargs)
        self.mass = self.EXTREME_LOWER_MASS

    @property
    def parameters(self):
        parameters = super(LightCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class HeavyCarMountainCar(ModifiableMountainCarEnv):
    def __init__(self):
        super(HeavyCarMountainCar, self).__init__()
        self.mass = self.EXTREME_UPPER_MASS

    @property
    def parameters(self):
        parameters = super(HeavyCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomHeavyCarMountainCar(ModifiableMountainCarEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        if new:
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomHeavyCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomLightCarMountainCar(ModifiableMountainCarEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        if new:
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomLightCarMountainCar, self).parameters
        parameters.update(
            {
                "mass": self.mass,
            }
        )
        return parameters


class RandomNormalMountainCar(ModifiableMountainCarEnv):
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.nsteps = 0  # for is_success()
        if new:
            self.force = self.np_random.uniform(
                self.RANDOM_LOWER_FORCE, self.RANDOM_UPPER_FORCE
            )
            self.mass = self.np_random.uniform(
                self.RANDOM_LOWER_MASS, self.RANDOM_UPPER_MASS
            )
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomNormalMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
                "mass": self.mass,
            }
        )
        return parameters


class RandomExtremeMountainCar(ModifiableMountainCarEnv):
    # TODO(cpacker): Is there any reason to not have an __init__?
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        new: Optional[bool] = True
    ):
        # Additionally call reset of gym.Env to reset the seed
        Env.reset(self, seed=seed)
        self.nsteps = 0  # for is_success()
        if new:
            self.force = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_FORCE,
                self.EXTREME_UPPER_FORCE,
                self.RANDOM_LOWER_FORCE,
                self.RANDOM_UPPER_FORCE,
            )
            self.mass = uniform_exclude_inner(
                self.np_random.uniform,
                self.EXTREME_LOWER_MASS,
                self.EXTREME_UPPER_MASS,
                self.RANDOM_LOWER_MASS,
                self.RANDOM_UPPER_MASS,
            )

        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0])
        return np.array(self.state, dtype=np.float32), {}

    @property
    def parameters(self):
        parameters = super(RandomExtremeMountainCar, self).parameters
        parameters.update(
            {
                "force": self.force,
                "mass": self.mass,
            }
        )
        return parameters
