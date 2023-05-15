import subprocess
import time

Model_path = "FoRL_logs"

Deterministic_envs = [
    "FoRLCartPole-v0",
    # "FoRLMountainCar-v0",
    # "FoRLPendulum-v0",
    # "FoRLAcrobot-v0",
    # "FoRLHopper-v0",
    # "FoRLHalfCheetah-v0",
]


def get_random_name(deterministic_name):
    return deterministic_name[:-3] + "RandomNormal-v0"


def get_extreme_name(deterministic_name):
    return deterministic_name[:-3] + "RandomExtreme-v0"


def prog_print(msg):
    print("=================================")
    print(msg)
    print("=================================")


Models = ["ddpg"]  # , "ppo_lstm", "a2c", "dqn"]


def get_train_cmd(Model, env):
    return [
        "python",
        "rl_zoo3/train.py",
        "--algo",
        Model,
        "--env",
        env,
        "-f",
        Model_path,
        "--verbose",
        "0",
        "--device",
        "cuda",
        "--progress",
        "-conf",
        f"FoRL_conf//{Model}.yml",
    ]


def get_eval_cmd(Model, train_env, test_env):
    return [
        "python",
        "rl_zoo3/enjoy.py",
        "--algo",
        Model,
        "--train_env",
        train_env,
        "--eval_env",
        test_env,
        "-f",
        Model_path,
        "--eval_folder",
        "eval_res.txt",
        "--device",
        "cuda",
        "--progress",
        "--no-render",
        "--no-hub",
        "-n",
        "50000",
    ]


for Model in Models:
    for env in Deterministic_envs:
        # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
        prog_print(f"Training deterministic env for {Model} in {env}")
        res = subprocess.Popen(get_train_cmd(Model, env))
        res.wait()

        prog_print(f"Training random env for {Model} in {env}")
        res = subprocess.Popen(get_train_cmd(Model, get_random_name(env)))
        res.wait()

        prog_print(f"Eval DD for {Model} in {env}")
        res = subprocess.Popen(get_eval_cmd(Model, env, env))
        res.wait()

        prog_print(f"Eval DR for {Model} in {env}")
        res = subprocess.Popen(get_eval_cmd(Model, env, get_random_name(env)))
        res.wait()

        prog_print(f"Eval RR for {Model} in {env}")
        res = subprocess.Popen(
            get_eval_cmd(Model, get_random_name(env), get_random_name(env))
        )
        res.wait()

        prog_print(f"Eval DE for {Model} in {env}")
        res = subprocess.Popen(get_eval_cmd(Model, env, get_extreme_name(env)))
        res.wait()

        prog_print(f"Eval RE for {Model} in {env}")
        res = subprocess.Popen(
            get_eval_cmd(Model, get_random_name(env), get_extreme_name(env))
        )
        res.wait()
