import argparse
import subprocess
envs = [
    "FoRLHalfCheetah-v0",
]

Models = [
    "ppo_lstm",
    ]

eps = [
    0.0
]

parser = argparse.ArgumentParser()
parser.add_argument('--schedule', action='store_true')
args = parser.parse_args()
with open('FoRL/to_run.txt', 'w') as f:
    for model in Models:
        for env in envs:
            for ep in eps:
                f.write(model + ' ' + env + ' ' + str(ep/100.0) + '\n')
#subprocess.Popen(["sbatch", "train_eval_template.sh"])
if args.schedule:
    subprocess.Popen(["sbatch", "train_eval_template.sh"])
