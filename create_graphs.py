import os
import math
import numpy as np 
import matplotlib.pyplot as plt
FILE_NAME = 'eval_res.txt'
RESULTS_DIR = 'results'

envs = [
    "FoRLCartPole",
    "FoRLMountainCar",
    "FoRLPendulum",
    "FoRLAcrobot",
    "FoRLHopper",
    "FoRLHalfCheetah",
]
def short_name(name):
    if 'RandomNormal' in name:
        return 'R'
    if 'RandomExtreme' in name:
        return 'E'

    return 'D'

def load_data(fn):
    f =  open(fn, 'r')
    lines = f.read().splitlines()
    data = {}
    for l in lines:
        elements = l.split(', ')
        train_name = elements[0][:-3]
        eval_name = elements[1][:-3]
        algo_name = elements[2]
        succ_rate = elements[3]
        succ_std = elements[4]
        
        for env_name in envs:
            if env_name not in train_name:
                continue
            
            if algo_name not in data.keys():
                data[algo_name] = {}
            if env_name not in data[algo_name].keys():
                data[algo_name][env_name] = {}
            
                
            key = short_name(train_name) + short_name(eval_name)
            if key in data[algo_name][env_name]:
                print("Key", key, "already in data")
            data[algo_name][env_name][key] = (float(succ_rate),  float(succ_std))
    return data


def geom_mean(values):
    return math.pow(math.prod(values), 1./len(values))

def geom_std(values):
    values = [v for v in values if v != 0]
    return np.exp(np.std(np.log(values)))


            
def plot_graphs(data):
    for algo in data.keys():
        xs = []
        ys = []
        gms = []
        gstds = []
        errs = []
        for env in data[algo].keys():
            if env == 'FoRLMountainCar':
                continue
            try:
                dd = data[algo][env]['DD']
                dr = data[algo][env]['DR']
                de = data[algo][env]['DE']
                rr = data[algo][env]['RR']
                re = data[algo][env]['RE']
                gm =  geom_mean([dr[0], de[0], re[0]])
                gstd = geom_std([dr[0], de[0], re[0]])
                gms.append(gm)
                gstds.append(gstd)
            except:
                print("Failed for", algo, env, data[algo][env])

        gmm = np.mean(gms)
        gstdm = np.mean(gstds)
        xs.append(0)
        ys.append(gmm)
        errs.append(gstdm)
        for eps in sorted([0.01,0.05,0.1,0.001,0.005]):
            gms = []
            gstds = []
            for env in data[algo].keys(): 
                if env == 'FoRLMountainCar':
                    continue
                try:
                    data2 = load_data(f'results\{algo}_{env}-v0_{eps}_result.txt')
                except:
                    print("No file for ", f'results\{algo}_{env}-v0_{eps}_result.txt')
                    continue
                try:
                    dd = data2[algo][env]['DD']
                    dr = data2[algo][env]['DR']
                    de = data2[algo][env]['DE']
                    rr = data2[algo][env]['RR']
                    re = data2[algo][env]['RE']
                    gm =  geom_mean([dr[0], de[0], re[0]])
                    gstd = geom_std([dr[0], de[0], re[0]])
                    gms.append(gm)
                    gstds.append(gstd)
                except:
                    print("Failed for", algo, env, data2[algo][env], eps)
                    continue
            if(len(gms)==0):
                continue
            gmm = np.mean(gms)
            gstdm = np.mean(gstds)
            xs.append(eps)
            ys.append(gmm)
            errs.append(np.std(gms))
        print(xs)
        print(ys)
        print(errs)
        plt.title(f"Generalization of {algo}")
        plt.bar(range(6),ys, yerr=errs)
        plt.xticks(range(6),labels=sorted([0,0.01,0.05,0.1,0.001,0.005]))
        plt.show()
def main():
    data = load_data(FILE_NAME)
    plot_graphs(data)
            
if __name__ == '__main__':
    main()