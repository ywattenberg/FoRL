import argparse
import subprocess
# Pendulum, CartPole, Mountaincar, Acrobot, Hopper, HalfCheetah
envs = [
    "FoRLHalfCheetah-v0",
]

def get_random_name(deterministic_name):
    return deterministic_name[:-3] + "RandomNormal-v0"

Models = [
    "ppo_lstm",
    ]

eps = [
    0.0
]

parser = argparse.ArgumentParser()
parser.add_argument('--schedule', action='store_true')
args = parser.parse_args()
with open('to_run.txt', 'w') as f:
    for model in Models:
        for env in envs:
            for ep in eps:
                f.write(model + ' ' + env + ' ' + str(ep/100.0) + '\n')
                f.write(model + ' ' + get_random_name(env) + ' ' + str(ep/100.0) + '\n')
#subprocess.Popen(["sbatch", "train_eval_template.sh"])
if args.schedule:
    subprocess.Popen(["sbatch", "train_eval_template.sh"])
