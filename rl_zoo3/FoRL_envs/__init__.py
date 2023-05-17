# from .registration import register, registry, make
# from .monitor import MonitorParameters
from gymnasium.envs.registration import register

# Classic control environments.

register(
    id="FoRLCartPole-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_cartpole:ModifiableCartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="FoRLCartPoleRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_cartpole:RandomNormalCartPole",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="FoRLCartPoleRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_cartpole:RandomExtremeCartPole",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="FoRLMountainCar-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_mountain_car:ModifiableMountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLMountainCarRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_mountain_car:RandomNormalMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLContinuousMountainCar-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_continuous_mountain_car:ModifiableContinuousMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLContinuousMountainCarRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_mountain_car:RandomNormalContinuousMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLContinuousMountainCarRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_mountain_car:RandomExtremeContinuousMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLMountainCarRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_mountain_car:RandomExtremeMountainCar",
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id="FoRLPendulum-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_pendulum:ModifiablePendulumEnv",
    max_episode_steps=200,
)

register(
    id="FoRLPendulumRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_pendulum:RandomNormalPendulum",
    max_episode_steps=200,
)

register(
    id="FoRLPendulumRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_pendulum:RandomExtremePendulum",
    max_episode_steps=200,
)

register(
    id="FoRLAcrobot-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_acrobot:ModifiableAcrobotEnv",
    max_episode_steps=500,
)

register(
    id="FoRLAcrobotRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_acrobot:RandomNormalAcrobot",
    max_episode_steps=500,
)

register(
    id="FoRLAcrobotRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_classic_control.modifiable_acrobot:RandomExtremeAcrobot",
    max_episode_steps=500,
)

# # Mujoco environments

register(
    id="FoRLHopper-v0",
    entry_point="FoRL_envs.modifiable_mujoco.hopper:ModifiableHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="FoRLHopperRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_mujoco.hopper:RandomNormalHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="FoRLHopperRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_mujoco.hopper:RandomExtremeHopper",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id="FoRLHalfCheetah-v0",
    entry_point="FoRL_envs.modifiable_mujoco.half_cheetah:ModifiableHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="FoRLHalfCheetahRandomNormal-v0",
    entry_point="FoRL_envs.modifiable_mujoco.half_cheetah:RandomNormalHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id="FoRLHalfCheetahRandomExtreme-v0",
    entry_point="FoRL_envs.modifiable_mujoco.half_cheetah:RandomExtremeHalfCheetah",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

# print(registration.registry)
# print(registration.registry.__dict__)

"""Commented-out code for registering unused environment combinations
CLASSIC_CONTROL = {
    'CartPole': [
        'StrongPush',
        'WeakPush',
        'ShortPole',
        'LongPole',
        'LightPole',
        'HeavyPole',
    ],
    'MountainCar': [
        'LowStart',
        'HighStart',
        'WeakForce',
        'StrongForce',
        'LightCar',
        'HeavyCar',
    ],
    'Pendulum': [
        'Light',
        'Heavy',
        'Short',
        'Long',
    ],
    'Acrobot': [
        'Light',
        'Heavy',
        'Short',
        'Long',
        'LowInertia',
        'HighInertia',
    ]
}

for baseline, variants in CLASSIC_CONTROL.items():
    for variant in variants:
        if baseline == 'CartPole':
            max_length = 200
            goal_achieved = 195.0
        elif baseline == 'MountainCar':
            max_length = 200
            goal_achieved = -110.0
        elif baseline == 'Pendulum':
            max_length = 200
            goal_achieved = None
        elif baseline == 'Acrobot':
            max_length = 500
            goal_achieved = None

        register(
            id='Sunblaze{}{}-v0'.format(baseline, variant),
            entry_point='FoRL_envs.modifiable_classic_control.modifiable_mountain_car:{}{}'.format(variant, baseline),
            max_episode_steps=max_length,
            reward_threshold=goal_achieved,
        )

        register(
            id='Sunblaze{}Random{}-v0'.format(baseline, variant),
            entry_point='FoRL_envs.modifiable_classic_control.modifiable_mountain_car:Random{}{}'.format(variant, baseline),
            max_episode_steps=max_length,
            reward_threshold=goal_achieved,
        )
"""

"""Commented-out code for registering unused environment combinations
MUJOCO = {
    'Hopper': [
        'Strong',
        'Weak',
        'HeavyTorso',
        'LightTorso',
        'SlipperyJoints',
        'RoughJoints',
    ],
    'HalfCheetah': [
        'Strong',
        'Weak',
        'HeavyTorso',
        'LightTorso',
        'SlipperyJoints',
        'RoughJoints',
    ]
}

for baseline, variants in MUJOCO.items():
    for variant in variants:
        if baseline == 'Hopper':
            goal_achieved = 3800.0
        elif baseline == 'HalfCheetah':
            goal_achieved = 4800.0
        # elif baseline == 'Ant':
            # goal_achieved = 6000.0

        register(
            id='Sunblaze{}{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.mujoco:{}{}'.format(variant, baseline),
            max_episode_steps=1000,
            reward_threshold=goal_achieved,
        )

        register(
            id='Sunblaze{}Random{}-v0'.format(baseline, variant),
            entry_point='sunblaze_envs.mujoco:Random{}{}'.format(variant, baseline),
            max_episode_steps=1000,
            reward_threshold=goal_achieved,
        )
"""


"""Commented-out Ant (not used at the moment)

register(
    id='SunblazeAnt-v0',
    entry_point='sunblaze_envs.mujoco:ModifiableRoboschoolAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='SunblazeAntRandomNormal-v0',
    entry_point='sunblaze_envs.mujoco:RandomNormalAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='SunblazeAntRandomExtreme-v0',
    entry_point='sunblaze_envs.mujoco:RandomExtremeAnt',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
"""

'''Commented-out registration code for Atari and Doom envs

from .breakout import Breakout
from .space_invaders import SpaceInvaders
from .vizdoom import VizDoomEnvironment

# Maximum number of episode steps with frameskip of 4.
MAX_EPISODE_STEPS = 10000

def register_delayed_actions(env_id, entry_point, set_a, set_b, kwargs=None):
    """Helper for registering environment with delayed actions."""
    if kwargs is None:
        kwargs = {}

    for set_name, set_range in [('A', set_a), ('B', set_b)]:
        kwargs.update({
            'wrapped_class': entry_point,
            'wrappers': [
                ('sunblaze_envs.wrappers:ActionDelayWrapper', {
                    'delay_range_start': set_range[0],
                    'delay_range_end': set_range[1],
                }),
            ],
        })

        register(
            id='Sunblaze{}DelayedActionsSet{}-v0'.format(env_id, set_name),
            entry_point='sunblaze_envs.wrappers:wrap_environment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs=kwargs,
        )


# Physical 2D world environments.
for game in [Breakout, SpaceInvaders]:
    worlds = game.worlds.keys()
    game = game.__name__

    for world in worlds:
        if world == 'baseline':
            name = ''
        else:
            name = ''.join([w.capitalize() for w in world.split('_')])

        # Default frameskip (4) environment.
        register(
            id='Sunblaze{}{}-v0'.format(game, name),
            entry_point='sunblaze_envs:{}'.format(game),
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'world': world
            }
        )

        # No frameskip environment.
        register(
            id='Sunblaze{}{}NoFrameskip-v0'.format(game, name),
            entry_point='sunblaze_envs:{}'.format(game),
            max_episode_steps=4 * MAX_EPISODE_STEPS,
            kwargs={
                'world': world,
                'frameskip': 1,
            }
        )

        # Delayed action modification of the environment.
        if world == 'baseline':
            register_delayed_actions(
                env_id=game,
                entry_point='sunblaze_envs:{}'.format(game),
                set_a=(0, 3),
                set_b=(1, 5),
                kwargs={
                    'world': world
                }
            )

# VizDoom environments.
for scenario, variants in VizDoomEnvironment.scenarios.items():
    for name, variant in variants.items():
        scenario_name = ''.join([w.capitalize() for w in scenario.split('_')])

        if name == 'baseline':
            variant_name = ''
        else:
            variant_name = ''.join([w.capitalize() for w in name.split('_')])

        # Default frameskip (4) environment.
        register(
            id='SunblazeVizDoom{}{}-v0'.format(scenario_name, variant_name),
            entry_point='sunblaze_envs:VizDoomEnvironment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'scenario': scenario,
                'variant': name,
            }
        )

        # No frameskip environment.
        register(
            id='SunblazeVizDoom{}{}NoFrameskip-v0'.format(scenario_name, variant_name),
            entry_point='sunblaze_envs:VizDoomEnvironment',
            max_episode_steps=MAX_EPISODE_STEPS,
            kwargs={
                'scenario': scenario,
                'variant': name,
                'frameskip': 1,
            }
        )

        # Delayed action modification of the environment.
        if name == 'baseline':
            register_delayed_actions(
                env_id='VizDoom{}'.format(scenario_name),
                entry_point='sunblaze_envs:VizDoomEnvironment',
                set_a=(0, 5),
                set_b=(5, 10),
                kwargs={
                    'scenario': scenario,
                    'variant': name,
                }
            )
'''
