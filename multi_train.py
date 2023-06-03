from multiprocessing import Pool
from optimize_one import main

if __name__ == '__main__':
    path = "FoRL/to_run.txt"
    with open(path, 'r') as f:
        lines = f.readlines()
    runs = []
    for line in lines:
        with Pool(processes=10) as pool:
            pool.map(main, [line.strip().split(' ')]*10)
    #    runs.append(line.strip().split(' '))
    # with Pool(processes=1) as pool:
    #     pool.map(main, runs)



