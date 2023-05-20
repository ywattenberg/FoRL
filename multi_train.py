from multiprocessing import Pool
from train_eval_one import main

if __name__ == '__main__':
    path = "FoRL/to_run.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
    runs = []
    for line in lines:
        runs.append(line.strip().split(' '))
    with Pool(processes=12) as pool:
        pool.map(main, runs)


