import subprocess
import time
import sys
import os

def get_random_name(deterministic_name):
    return deterministic_name[:-3] + "RandomNormal-v0"


def get_extreme_name(deterministic_name):
    return deterministic_name[:-3] + "RandomExtreme-v0"


def prog_print(msg):
    print("=================================")
    print(msg)
    print("=================================")


def get_eval_cmd(python_path, Model, env, Model_path, eps):

    return [
        python_path,
        "FoRL/rl_zoo3/train.py",
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
        f"FoRL/FoRL_conf/{Model}.yml",
        "--env-kwargs",
        f"epsilon: {eps}",
        "--n-timesteps",
        "100000",
        "--optimize",
        "--n-trials",
        "10",
        "--n-jobs",
        "2",
        "--study-name",
        f"{Model}_{env}_{eps}",
        f"--storage",
        f"sqlite:///{Model_path}results.db"
    ]

def main(args):
    Model, env, eps = args
    Model_path = "FoRL/results/optimized/"
    python_path = os.environ["PYTHONPATH"]

    prog_print(f"START ==>  Model: {Model}, env: {env}, Model_path: {Model_path}, python_path: {python_path}")
    sys.stdout.flush()
    # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
    prog_print(f"Optimize {Model} in {env}")
    res = subprocess.Popen(get_eval_cmd(python_path, Model, env, Model_path, eps))
    res.wait()
    sys.stdout.flush()
    
    with open('FoRL/to_run.txt', 'r') as f:
        lines = f.readlines()
    # print(lines[0].startswith(f"{Model} {env} {eps}"))
    with open('FoRL/to_run.txt', 'w') as f:
        for line in lines:
            if line.strip("\n") != f"{Model} {env} {eps}":
                f.write(line)
    prog_print(f"DONE for {Model} in {env}")
    sys.stdout.flush()


if __name__ == "__main__":
    print(sys.argv[0])
    Model = sys.argv[1]
    env = sys.argv[2]
    eps = sys.argv[3]
    main([Model, env, eps]) 
