import os
import math
import numpy as np 

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

def geom_mean(values):
    return math.pow(math.prod(values), 1./len(values))

def geom_std(values):
    return np.exp(np.std(np.log(values)))

def table_start(f, header):
    col_centered = 'l' + (len(header)-1) * 'c'
    f.write('\\begin{center}\n')
    f.write('\\begin{tabular}{'+col_centered+'}\n')
    f.write('\\toprule\n')
    f.write(header[0])
    for i in range(len(header)-1):
        f.write(f'&{header[i+1]}')
    f.write('\\\\\n')

def table_end(f):
    f.write('\\bottomrule\n')
    f.write('\\end{tabular}\n') 
    f.write('\\end{center}\n')

def table_write_line(f, values):
    f.write(values[0])
    for i in range(len(values)-1):
        f.write(f'&{values[i+1]}')
    f.write('\\\\\n')

def format_succ(tuple):
    return f"{100*tuple[0]:.2f}+/-{tuple[1]:.2f}"

def write_latex(data):
    f = open(LATEX_FILE, 'w')
    table_start(f, ['Algorithm', 'Environment', 'Default', 'Interpolation', 'Extrapolation'])
 
    for algo in data.keys():
        f.write('\\midrule\n')
        for env in data[algo].keys():
            
            try:
                dd = data[algo][env]['DD']
                dr = data[algo][env]['DR']
                de = data[algo][env]['DE']
                rr = data[algo][env]['RR']
                re = data[algo][env]['RE']
                gm =  geom_mean([dr[0], de[0], re[0]])
                gstd = geom_std([dr[0], de[0], re[0]])
            except:
                print("Failed for", algo, env, data[algo][env])

            table_write_line(f, [algo, env, format_succ(dd), format_succ(rr), format_succ((gm,gstd))])
    
    table_end(f)

def write_latex_full(data):
    f = open(LATEX_FILE, 'w')
    table_start(f, ['Algorithm', 'Environment', 'DD', 'DR', 'DE', 'RR', 'RE'])
 
    for algo in data.keys():
        f.write('\\midrule\n')
        for env in data[algo].keys():
            
            try:
                dd = data[algo][env]['DD']
                dr = data[algo][env]['DR']
                de = data[algo][env]['DE']
                rr = data[algo][env]['RR']
                re = data[algo][env]['RE']
            except:
                print("Failed for", algo, env, data[algo][env])

            table_write_line(f, [algo, env, format_succ(dd), format_succ(dr), format_succ(de), format_succ(rr), format_succ(re)])
    
    table_end(f)

def main():
    data = load_data()
    write_latex(data)
            
if __name__ == '__main__':
    main()