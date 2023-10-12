# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:51:27 2021

@author: jureb
"""


import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

#sys.path.append("../../nltk/")

sys.path.append(os.getcwd()+"/source")
import ProGED as pg
from ProGED.generators.base_generator import ProGEDMaxAttemptError
from ProGED.generators.grammar_construction import string_to_unit, unit_to_string

#warnings.filterwarnings("ignore")

if __name__ == "__main__":
    datadir = ""

    N = 5000
    time_limit = 60 #2*24*3600
    
    grammar_name = "universal"
    units = True
    
    eqN = int(sys.argv[1])
    name = sys.argv[2]
    processN = int(sys.argv[3])
    np.random.seed(0)
    
    eqfile = "source/FeynmanEquations.csv"
    reference = pd.read_csv(eqfile)
    
    #print("---Reading grammar")
    #with open("grammar.pg", "rb") as f:
    #    grammar = pickle.load(f)
    
    #print("--Loading models")
    #with open(modelsfile, "rb") as file:
    #    models = pickle.load(file)
    
    """------------- create grammar ---------------"""
    funits = {}
    with open("source/feynman-units.csv", "r") as file:
        file.readline()
        for line in file:
            a = line.split(",")
            unit = unit_to_string([int(a[i]) for i in range(2,7)])
            funits[a[0]] = unit
    ufunits = list(set(list(funits.values())))
    
    var_number = int(reference["# variables"][eqN])
    var_names = [reference["v"+str(n)+"_name"][eqN] for n in range(1, var_number + 1)]# + [reference["Output"][eqN]]
    var_probs = [1/var_number]*var_number #+ [0.1/var_number]*var_number# + [0.1/var_number]*var_number
                                                             
    symbols = {"start":"S", "const":"C", "x":["'"+v+"'" for v in var_names]}
    functions1 = ["sin", "cos", "tan", "sqrt", "exp"]
    functions2 = ["asin", "acos", "atan", "sinh", "cosh", "tanh"]
    p_fun = [5]*len(functions1) + [1]*len(functions2)
    p_fun = 0.4 * np.array(p_fun) / np.sum(p_fun)
    
    if units:
        units = [string_to_unit(funits[var]) for var in var_names]
        target_var = reference["Output"][eqN]
        units += [string_to_unit(funits[target_var])]
        grammar = pg.grammar_from_template(grammar_name+"-dim", 
                                           {"variables": ["'"+v+"'" for v in var_names], 
                                            "functions": functions1+functions2, "p_functs": [0.6]+list(p_fun),
                                            "units": units, "extended_units": True})


    else:
        grammar= pg.grammar_from_template(grammar_name, 
                        {"variables": var_names, "p_vars": var_probs, 
                         "functions": functions1+functions2, "p_functs": [0.6]+list(p_fun)}
                        )
        
    with open("grammar.txt", "w") as f:
        f.write(str(grammar.grammar))
        
    """ --------------- generate models --------------"""
    models = pg.model_box.ModelBox()
    time_start = time.time()
    time_now = time_start
    expected_time = 0
    n = 0
    n_valid = 0
    
    while n_valid < N and 2*expected_time < time_limit - (time_now - time_start):
        try:
            sample, p, code = grammar.generate_one()
            expr_str = "".join(sample)
            
            #print("-> ", expr_str, p, code)
                
            valid, expr = models.add_model(expr_str, symbols = symbols, grammar = None, code=code, p=p)
            if valid:
                #print(expr)
                n_valid += 1
            
            #print("---> ", valid, expr)

        except ProGEDMaxAttemptError:
            pass
            #print("--- max attempts reached")
        
        n += 1
        time_now = time.time()
        expected_time = (time_now - time_start)/n
        if n % 1000 == 0:
            print("n: ", n, ", n valid: ", n_valid, ", n unique:", len(models), ", mean t: ", np.round(expected_time, 3))
    
    print("--Exporting generated models")
    with open("models.pg", "wb") as file:
        pickle.dump(models, file)
    
    #with open("models.txt", "w") as file:
    #    file.write(str(models))
    
    """------------------- create the xrsl for estimation -------------------"""
    datafile = reference["Filename"][eqN]
    name_est = name + "Estimation"
    txt = ''
    txt += '&\n'
    txt += '(executable = "run_'+name_est+ '.sh")\n'
    txt += '(inputFiles =\n'
    txt += '  ("run_'+name_est+'.sh" "run_'+name_est+'.sh")\n'
    txt += '  ("feynman_env.sif" "/media/sf_shared/cluster_jobs/feynman_env.sif")\n'
    txt += '  ("models.pg" "models.pg")\n'
    txt += '  ("source.tar" "/media/sf_shared/cluster_jobs/feynman_universal_2/source.tar")\n'
    #txt += '  ("Feynman_with_units_p'+str(i)+'.npz" "gsiftp://dcache.arnes.si/data/arnes.si/gen.vo.sling.si/brence/feynman/Feynman_with_units_p'+str(i)+'.npz")\n'
    txt += '  ("' + datafile + '" "/media/sf_shared/cluster_jobs/Feynman_with_units/' + datafile +'")\n'
    txt += ')\n'
    txt += '(outputFiles =\n'
    txt += '("results.tar" "results.tar")\n'
    txt += '("feynman.log" "feynman.log")\n'
    txt += ')\n'
    txt += '(walltime = "2 days")\n'
    txt += '(memory=2000)\n'
    txt += '(stdout="standard.log")\n'
    txt += '(count='+str(processN)+')\n'
    txt += '(countpernode='+str(processN)+')\n' 
    #txt += '(queue = "grid")\n'
    txt += '(jobname = "'+name+'")'
    with open(name_est + ".xrsl", "w", newline='\n') as file:
        file.write(txt)
        
    txt2 = ''
    txt2 += 'tar -xf source.tar\n'
    txt2 += 'mkdir -p results\n'
    txt2 += 'singularity exec feynman_env.sif python3.7 source/utils/estimation_for_grid_Feynman_database.py '+str(eqN)+' '+'models.pg ' + str(processN) + ' >> feynman.log\n'
    txt2 += 'wait\n'
    txt2 += 'tar -cf results.tar results'
    with open('run_'+name_est+'.sh', "w", newline='\n') as file:
        file.write(txt2)