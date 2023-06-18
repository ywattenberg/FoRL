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
        os.path.join(os.environ["HOME"], "FoRL/rl_zoo3/train.py"),
        #"rl_zoo3/train.py",
        "--algo",
        Model,
        "--env",
        env,
        "-f",
        Model_path,
        "--progress",
        "--verbose",
        "0",
        "--device",
        "cuda",
        "-conf",
        os.path.join(os.environ["HOME"], f"FoRL/FoRL_conf/{Model}.yml"),
        #"FoRL_conf/" + Model + ".yml",
        "--env-kwargs",
        f"epsilon: {eps}",
        "--n-timesteps",
        "50000",
    ]


def get_eval_cmd(python_path, Model, train_env, test_env, Model_path, eval_folder):
    return [
        python_path,
        os.path.join(os.environ["HOME"], "FoRL/rl_zoo3/enjoy.py"),
        #"rl_zoo3/enjoy.py",
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
        "10000",
    ]


def main(args):
    steps = 1e6
    Model, env_d, eps = args
    env = get_random_name(env_d)
    Model_path = os.path.join("os.environ["HOME"]", f"FoRL/results/models/evo/{str(eps)}")
    eval_folder = os.path.join(
        os.environ["HOME"],
        "FoRL/results/evo/" + Model + "_" + env + "_" + str(eps) + "_" + "result.txt",
    )
    python_path = os.environ["PYTHONPATH"]
    # Model_path = f"results/models/evo/{str(eps)}"
    # eval_folder = f"results/evo/{Model}_{env}_{str(eps)}_result.txt"
    # python_path = "python"

    train_cmd = get_train_cmd(python_path, Model, env, Model_path, eps)
    res = subprocess.run(train_cmd)

    for i in range(int(steps // 10000)):
        test_iid = get_eval_cmd(python_path, Model, env, env, Model_path, eval_folder)
        test_ood = get_eval_cmd(
            python_path,
            Model,
            env,
            get_extreme_name(env_d),
            Model_path,
            os.path.join(
                os.environ["HOME"],
                f"FoRL/results/evo/{Model}_{get_extreme_name(env_d)}_{str(eps)}_result.txt",
            ),
        )
            
        res = subprocess.run(test_iid)
        res = subprocess.run(test_ood)
        new_train = train_cmd + [
            "-i",
            Model_path + f"/{Model}/{env}_{i+1}" + env + ".zip",
        ]
        res = subprocess.run(train_cmd)
    res = subprocess.run(test_iid)
    res = subprocess.run(test_ood)


if __name__ == "__main__":
    print(sys.argv[0])
    Model = sys.argv[1]
    env = sys.argv[2]
    eps = sys.argv[3]
    main([Model, env, eps])
