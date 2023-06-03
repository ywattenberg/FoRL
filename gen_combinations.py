import argparse
import subprocess
# Pendulum, CartPole, Mountaincar, Acrobot, Hopper, HalfCheetah
envs = [
    "FoRLCartPole-v0",
    "FoRLMountainCar-v0",
    "FoRLPendulum-v0",
    "FoRLAcrobot-v0",
    "FoRLHopper-v0",
    "FoRLHalfCheetah-v0",
]

def get_random_name(deterministic_name):
    return deterministic_name[:-3] + "RandomNormal-v0"

Models = [    
    "a2c"
    ]

eps = [
    #0.0
    # 0.1,
    # 0.5,
    1,
    5,
    10
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
