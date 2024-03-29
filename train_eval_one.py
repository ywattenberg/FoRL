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


def get_train_cmd(python_path, Model, env, Model_path, eps):
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
    ]


def get_eval_cmd(python_path, Model, train_env, test_env, Model_path, eval_folder):
    return [
        python_path,
        "FoRL/rl_zoo3/enjoy.py",
        "--algo",
        Model,
        "--train_env",
        train_env,
        "--eval_env",
        test_env,
        "-f",
        Model_path,
        "--eval_folder",
        eval_folder,
        "--device",
        "cuda",
        "--progress",
        "--no-render",
        "--no-hub",
        "-n",
        "100000",
    ]


def main(args):
    Model, env, eps = args
    Model_path = os.environ["TMPDIR"]
    eval_folder = "FoRL/results/" + Model + "_" + env + "_" + str(eps) + '_'+ "result.txt"
    python_path = os.environ["PYTHONPATH"]

    prog_print(f"START ==>  Model: {Model}, env: {env}, Model_path: {Model_path}, eval_folder: {eval_folder}, python_path: {python_path}")
    sys.stdout.flush()
    # python .\train.py --algo a2c --env FoRLMountainCarRandomNormal-v0 --device cuda --vec-env subproc --progress -conf ..\FoRL_conf\a2c.yml
    prog_print(f"Training deterministic env for {Model} in {env}")
    res = subprocess.Popen(get_train_cmd(python_path, Model, env, Model_path, eps))
    res.wait()
    sys.stdout.flush()
    prog_print(f"Training random env for {Model} in {env}")
    res = subprocess.Popen(get_train_cmd(python_path, Model, get_random_name(env), Model_path, eps))
    res.wait()
    sys.stdout.flush()
    prog_print(f"Eval DD for {Model} in {env}")
    res = subprocess.Popen(get_eval_cmd(python_path, Model, env, env, Model_path, eval_folder))
    res.wait()
    sys.stdout.flush()
    prog_print(f"Eval DR for {Model} in {env}")
    res = subprocess.Popen(get_eval_cmd(python_path, Model, env, get_random_name(env), Model_path, eval_folder))
    res.wait()
    sys.stdout.flush()
    prog_print(f"Eval RR for {Model} in {env}")
    res = subprocess.Popen(
        get_eval_cmd(python_path, Model, get_random_name(env), get_random_name(env), Model_path, eval_folder)
    )
    res.wait()

    prog_print(f"Eval DE for {Model} in {env}")
    res = subprocess.Popen(get_eval_cmd(python_path, Model, env, get_extreme_name(env), Model_path, eval_folder))
    res.wait()

    prog_print(f"Eval RE for {Model} in {env}")
    res = subprocess.Popen(
        get_eval_cmd(python_path, Model, get_random_name(env), get_extreme_name(env), Model_path, eval_folder)
        )
    res.wait()
    
    with open('FoRL/to_run.txt', 'r') as f:
        lines = f.readlines()
    print(lines[0].startswith(f"{Model} {env} {eps}"))
    with open('FoRL/to_run.txt', 'w') as f:
        for line in lines:
            if line.strip("\n") != f"{Model} {env} {eps}":
                print(line)
                f.write(line)
    prog_print(f"DONE for {Model} in {env}")


if __name__ == "__main__":
    print(sys.argv[0])
    Model = sys.argv[1]
    env = sys.argv[2]
    eps = sys.argv[3]
    main([Model, env, eps]) 
