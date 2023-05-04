import subprocess
import time

FoRL_envs = [
    "FoRLCartPoleRandomNormal-v0",
    "FoRLMountainCarRandomNormal-v0",
    "FoRLPendulumRandomNormal-v0",
    "FoRLAcrobotRandomNormal-v0",
    "FoRLHopperRandomNormal-v0",
    "FoRLHalfCheetahRandomNormal-v0",
]

Deterministic_envs = [
    "FoRLCartPole-v0",
    "FoRLMountainCar-v0",
    "FoRLPendulum-v0",
    "FoRLAcrobot-v0",
    "FoRLHopper-v0",
    "FoRLHalfCheetah-v0",
]

Models = ["ppo", "a2c", "ppo_lstm"]

for Model in Models:
    for env in FoRL_envs + Deterministic_envs:
        # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
        cmd = (
            "python rl-zoo3\\train.py --algo "
            + Model
            + " --env "
            + env
            + " --device cuda --vec-env subproc --progress -conf FoRL_conf\\"
            + Model
            + ".yml"
        )
        res = subprocess.Popen(cmd.split(" "))
        res.wait()
