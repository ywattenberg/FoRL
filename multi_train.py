from multiprocessing import Pool
from optimize_one import main
import os

if __name__ == '__main__':
    path = os.path.join(os.environ["HOME"], "FoRL/to_run.txt")
    with open(path, 'r') as f:
        lines = f.readlines()
    runs = []
    for line in lines:
        runs = runs + [line.strip().split(' ')]*10
    with Pool(processes=10) as pool:
        pool.map(main, runs)
    #    runs.append(line.strip().split(' '))
    # with Pool(processes=1) as pool:
    #     pool.map(main, runs)



