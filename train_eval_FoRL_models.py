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

def prog_print(msg):
    print("=================================")
    print(msg)
    print("=================================")
Models = ["ppo", "a2c", "ppo_lstm"]

for Model in Models:
    for env in Deterministic_envs:
        # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
        prog_print(f"Training deterministic env for {Model} in {env}")
        cmd = (
            "python rl-zoo3\\train.py --algo "
            + Model
            + " --env "
            + env
            + " -f "
            + "rl-zoo3\\rl-trained-agents"
            + " --verbose "
            + "0"
            + " --device cuda --vec-env subproc --progress -conf FoRL_conf\\"
            + Model
            + ".yml"
        )
        res = subprocess.Popen(cmd.split(" "))
        res.wait()

        prog_print(f"Training random env for {Model} in {env}")
        cmd = (
            "python rl-zoo3\\train.py --algo "
            + Model
            + " --env "
            + get_random_name(env)
            + " -f "
            + "rl-zoo3\\rl-trained-agents"
            + " --verbose "
            + "0"
            + " --device cuda --vec-env subproc --progress -conf FoRL_conf\\"
            + Model
            + ".yml"
        )
        res = subprocess.Popen(cmd.split(" "))
        res.wait()
        
        prog_print(f"Eval RR for {Model} in {env}")
        cmdRR = (
            "python rl-zoo3\\enjoy.py --algo "
            + Model
            + " --train_env "
            + get_random_name(env)
            + " --eval_env "
            + get_random_name(env)
            + " -f "
            + "rl-zoo3\\rl-trained-agents"
            + " --eval_folder "
            + "eval_res.txt"
            + " --verbose "
            + "1"
            + " --device cuda --progress --no-render --no-hub"
        )
        res = subprocess.Popen(cmdRR.split(" "))
        res.wait()
        
        prog_print(f"Eval DR for {Model} in {env}")
        cmdDR = (
            "python rl-zoo3\\enjoy.py --algo "
            + Model
            + " --train_env "
            + env
            + " --eval_env "
            + get_random_name(env)
            + " -f "
            + "rl-zoo3\\rl-trained-agents"
            + " --eval_folder "
            + "eval_res.txt"
            + " --verbose "
            + "1"
            + " --device cuda --progress --no-render --no-hub"
        )
        res = subprocess.Popen(cmdDR.split(" "))
        res.wait()
        
        prog_print(f"Eval RE for {Model} in {env}")
        cmdRE = (
            "python rl-zoo3\\enjoy.py --algo "
            + Model
            + " --train_env "
            + get_random_name(env)
            + " --eval_env "
            + get_extreme_name(env)
            + " -f "
            + "rl-zoo3\\rl-trained-agents"
            + " --eval_folder "
            + "eval_res.txt"
            + " --verbose "
            + "1"
            + " --device cuda --progress --no-render --no-hub"
        )
        res = subprocess.Popen(cmdRE.split(" "))
        res.wait()
