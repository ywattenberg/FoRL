import os
import math

FILE_NAME = 'eval_res.txt'
LATEX_FILE = 'table.txt'

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

def load_data():
    f =  open(FILE_NAME, 'r')
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

def geom_mean(v1, v2, v3):
    return (v1*v2*v3)**(1./3)

def geom_std(v1, v2, v3, gm):
    v1 = gm if v1 == 0 else v1
    v2 = gm if v2 == 0 else v2
    v3 = gm if v3 == 0 else v3
    if gm == 0:
        return 0
    sqr_sum = math.log(v1/gm)**2 + math.log(v2/gm)**2 + math.log(v3/gm)**2
    sqr_sum = sqr_sum/3
    return math.exp(math.sqrt(sqr_sum))

def write_latex(data):
    f = open(LATEX_FILE, 'w')
    f.write('\\begin{center}\n')
    f.write('\\begin{tabular}{lcccc}\n')
    f.write('\\toprule\n')
    f.write('Algorithm&Environment&Default&Interpolation&Extrapolation\\\\\n')
    
    for algo in data.keys():
        f.write('\\midrule\n')
        for env in data[algo].keys():
            
            try:
                dr = data[algo][env]['DR']
                rr = data[algo][env]['RR']
                re = data[algo][env]['RE']
            except:
                print("Failed for", algo, env, data[algo][env])

            gm =  geom_mean(dr[0], rr[0], re[0])
            gstd = geom_std(dr[1], rr[1], re[1], gm)
            f.write(f"{algo}&{env}&nan&{rr[0]:.2f}+/-{rr[1]:.2f}&{gm:.2f}+/-{gstd:.2f}\\\\\n")   
    
    
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n') 
    f.write('\\end{center}\n')

def main():
    data = load_data()
    write_latex(data)
            
if __name__ == '__main__':
    main()