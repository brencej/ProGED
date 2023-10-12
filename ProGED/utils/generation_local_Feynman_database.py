# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 15:51:27 2021

@author: jureb
"""


import os
import pickle

import numpy as np
import pandas as pd

import ProGED as pg
from ProGED.generators.grammar_construction import string_to_unit, unit_to_string

# sys.path.append("../../nltk/")

#warnings.filterwarnings("ignore")

np.random.seed(0)


if __name__ == "__main__":

    datadir = "E:/Feynman_with_units/"
    eqfile = "../source/FeynmanEquations.csv"
    reference = pd.read_csv(eqfile)
    
    modelsN = 10000
    
    batch_size = None
    n_batches = 1
    
    name = "progedDimTest2"
    workers = 8
    
    grammar_name = "universal"
    units = True
    
    output_folder = name + "/" 
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    
    funits = {}
    with open("../source/feynman-units.csv", "r") as file:
        file.readline()
        for line in file:
            a = line.split(",")
            unit = unit_to_string([int(a[i]) for i in range(2,7)])
            funits[a[0]] = unit
    ufunits = list(set(list(funits.values())))
    
    print(name)
    for eqN in range(8, 9):
        datafile = reference["Filename"][eqN]
        print("eqN: " + str(eqN) + ", file: " + datafile)
        data = np.loadtxt(datadir + datafile)
        #%%
        var_number = int(reference["# variables"][eqN])
        var_names = [reference["v"+str(n)+"_name"][eqN] for n in range(1, var_number + 1)] + [reference["Output"][eqN]]
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
            ED = pg.EqDisco(data = data, variable_names = var_names, generator_template_name = grammar_name+"-dim", 
                            variable_probabilities = var_probs,
                            generator_settings = {"functions": functions1+functions2, "p_functs": [0.6]+list(p_fun),
                                                  "units": units, "extended_units": True},
                            strategy_settings = {"N": modelsN})
        
        else:
            ED = pg.EqDisco(data = data, variable_names = var_names, generator_template_name = grammar_name, 
                            variable_probabilities = var_probs,
                            generator_settings = {"functions": functions1+functions2, "p_functs": [0.6]+list(p_fun)},
                            strategy_settings = {"N": modelsN})
        
        print("Generating models")
        models = ED.generate_models()
        model_batches = models.split(batch_size = batch_size, n_batches = n_batches)
        
        for i in range(len(model_batches)):            
            batch_name = name + "_eq" + str(eqN) + "_batch" + str(i)
            batch_folder = output_folder + batch_name + "/"
            os.makedirs(os.path.dirname(batch_folder), exist_ok = True)
            
            with open(batch_folder + batch_name + ".models", "wb") as file:
                pickle.dump(model_batches[i], file)
                
            txt = ''
            txt += '&\n'
            txt += '(executable = "run_'+name+ '.sh")\n'
            txt += '(inputFiles =\n'
            txt += '  ("run_'+name+'.sh" "run_'+name+'.sh")\n'
            txt += '  ("feynman_env.sif" "/media/sf_shared/cluster_jobs/feynman_env.sif")\n'
            txt += '  ("' + batch_name + '.models" "' + batch_name + '.models")\n'
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
            txt += '(count='+str(workers)+')\n'
            txt += '(countpernode='+str(workers)+')\n' 
            #txt += '(queue = "grid")\n'
            txt += '(jobname = "'+batch_name+'")'
            with open(batch_folder + name + ".xrsl", "w", newline='\n') as file:
                file.write(txt)
                
            txt2 = ''
            txt2 += 'tar -xf source.tar\n'
            txt2 += 'mkdir -p results\n'
            txt2 += 'singularity exec feynman_env.sif python3.7 source/utils/estimation_for_grid_Feynman_database.py '+str(eqN)+' '+batch_name+'.models ' + str(workers) + ' >> feynman.log\n'
            txt2 += 'wait\n'
            txt2 += 'tar -cf results.tar results'
            with open(batch_folder + 'run_'+name+'.sh', "w", newline='\n') as file:
                file.write(txt2)
                
