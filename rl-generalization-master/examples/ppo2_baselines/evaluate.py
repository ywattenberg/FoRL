import argparse
import json
import multiprocessing
import os
import pickle
import random

from baselines.common import set_global_seeds, tf_util as U
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from chainerrl import misc
import gym
from gym import wrappers
from gym.utils.seeding import create_seed
import numpy as np
import tensorflow as tf

from . import base
from ..util import NumpyEncoder


def main():
    parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('load', type=str)
    parser.add_argument('--normalize', type=str)
    parser.add_argument('--env', type=str, default='SunblazeCartPole-v0')
    #parser.add_argument('--policy', help='Policy architecture', choices=['mlp', 'cnn'], default='mlp')
    parser.add_argument('--seed', type=int, help='RNG seed, defaults to random')
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--max-episode-len', type=int, default=10000)
    parser.add_argument('--eval-n-trials', type=int, default=100)
    parser.add_argument('--episodes-per-trial', type=int, default=1)
    parser.add_argument('--eval-n-parallel', type=int, default=1)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('load', type=str, nargs='*')
    args = parser.parse_args()

    # Fixes problem of eval script being run with ".../checkpoints/*"
    if len(args.load) > 1:
        import natsort
        print("Detected multiple model file args, sorting and choosing last..")
        # Fixes issue of 'normalize' file inside checkpoint folder
        args.load = [f for f in args.load if 'normalize' not in f]
        args.load = natsort.natsorted(args.load, reverse=True)[0]
        print("Using {}".format(args.load))
    else:
        args.load = args.load[0]

    # ppo2 is trained on "total episodes" only but is evaluated via "trials"
    total_episodes = args.eval_n_trials * args.episodes_per_trial

    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)

    # If seed is unspecified, generate a pseudorandom one
    if not args.seed:
        # "Seed must be between 0 and 2**32 - 1"
        seed = create_seed(args.seed, max_bytes=4)
    else:
        seed = args.seed

    # Log it for reference
    with open(os.path.join(args.outdir, 'seed.txt'), 'w') as fout:
        fout.write("%d\n" % seed)

    # Set the tf, numpy and random seeds (in addition to the env seed)
    set_global_seeds(seed)

    output_lock = multiprocessing.Lock()

    def evaluator(process_idx):
        def make_env():
            env = base.make_env(args.env, process_idx)
            # This ensures that the env seeds are not the same between
            # parallel evaluators
            # Note that the tf/numpy/python.random seeds are seeded once prior
            env.seed(seed + process_idx)
            if args.record:
                env = gym.wrappers.Monitor(env, args.outdir, video_callable=lambda _: True)
            return env

        env = DummyVecEnv([make_env])
        obs_space = env.observation_space
        act_space = env.action_space

        # TODO(cpacker): this should really be in the top-level dir
        norm_path = args.normalize if args.normalize else os.path.join(os.path.dirname(args.load), 'normalize')
        with open(norm_path, 'rb') as f:
            obs_norms = pickle.load(f)
        clipob = obs_norms['clipob']
        mean = obs_norms['mean']
        var = obs_norms['var']

        # Load model
        with U.make_session(num_cpu=1) as sess:

            if 'SpaceInvaders' in args.env or 'Breakout' in args.env:
                raise NotImplementedError
            else:
                # '.../checkpoint/XXXX' -> '.../make_model.pkl'
                pkl_path = os.path.join(
                    os.path.dirname(os.path.dirname(args.load)),
                    'make_model.pkl')
                # from: https://github.com/openai/baselines/issues/115
                print("Constructing model from " + pkl_path)
                with open(pkl_path, 'rb') as fh:
                    import cloudpickle
                    make_model = cloudpickle.load(fh)
                model = make_model()
                print("Loading saved model from " + args.load)
                model.load(args.load)
                '''
                # alternate method
                policy_fn = base.mlp_policy
                model = Model(
                    policy_fn,
                    obs_space,
                    act_space,
                    nbatch_act=,
                    nbatch_train=,
                    nsteps=,
                    ent_coef=,
                    vf_coef=,
                    max_grad_norm=,

                    nenvs=1,
                    nsteps=args.max_episode_len
                )
                model.load(args.load)
                '''
            # Unwrap DummyVecEnv to access mujoco.py object
            env_base = env.envs[0].unwrapped

            # Record a binary success measure if the env supports it
            if hasattr(env_base, 'is_success') and callable(getattr(env_base, 'is_success')):
                success_support = True
            else:
                print("Warning: env does not support binary success, ignoring.")
                success_support = False

            rate = 0.0
            for _ in range(total_episodes):
                obs, state, done = env.reset(), model.initial_state, False
                episode_rew = 0
                success = False
                for step in range(args.max_episode_len):
                    obs = np.clip((obs-mean) / np.sqrt(var), -clipob, clipob)  # normalize
                    action, value, state, _ = model.step(obs, state, np.reshape(np.asarray([done]), (1,)))
                    obs, rew, done, _ = env.step(action)
                    episode_rew += rew
                    if success_support and env_base.is_success():
                        #print("success at step {} w/ reward {}".format(step,episode_rew))
                        success = True
                    if done:
                        #print("done at step {} w/ reward {}".format(step,episode_rew))
                        break
                rate += success/total_episodes

                with output_lock:
                    with open(os.path.join(args.outdir, 'evaluation.json'), 'a') as results_file:
                        results_file.write(json.dumps({
                            'reward': episode_rew,
                            'success': success if success_support else 'N/A',
                            'environment': env_base.parameters,
                            'model': args.load,
                        }, cls=NumpyEncoder))
                        results_file.write('\n')
            #print(rate)

    misc.async.run_async(args.eval_n_parallel, evaluator)


if __name__ == '__main__':
    main()

