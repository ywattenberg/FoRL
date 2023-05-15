import subprocess
import time


Deterministic_envs = [
    "FoRLCartPole-v0",
    "FoRLMountainCar-v0",
    "FoRLPendulum-v0",
    "FoRLAcrobot-v0",
    "FoRLHopper-v0",
    "FoRLHalfCheetah-v0",
]

def get_random_name(deterministic_name):
    return deterministic_name[:-3] + "RandomNormal-v0"

def get_extreme_name(deterministic_name):
    return deterministic_name[:-3] + "RandomExtreme-v0"

Models = ["ppo", "a2c", "ppo_lstm"]

for Model in Models:
    for env in Deterministic_envs:
        # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
        cmdRR = (
            "python rl-zoo3/enjoy.py --algo "
            + Model
            + " --train_env "
            + get_random_name(env)
            + " --eval_env "
            + get_random_name(env)
            + " -f "
            + "rl-zoo3/rl-trained-agents"
            + " --device cuda --progress --no-render --no-hub"
        )
        res = subprocess.Popen(cmdRR.split(" "))
        res.wait()
