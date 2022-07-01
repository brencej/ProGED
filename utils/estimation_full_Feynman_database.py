# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 14:52:13 2021

@author: jureb
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd()+"/source")
import ProGED as pg
from ProGED.generators.grammar_construction import string_to_unit, unit_to_string

#warnings.filterwarnings("ignore")

np.random.seed(0)

if __name__ == "__main__":

    grammar_name = "universal"
    units = True
    
    max_N = 1000 # maximum number of unique equations we want to end up with
    max_repeats = 10**7 # maximum total number of generate_one calls we allow
    max_attempts = 1 # maximum number of generation attempts per generate_one call
    max_depth = 1000 # maximum recursion depth allowed in each generate_one call
    
    eqN = int(sys.argv[1])
    workers = int(sys.argv[2])

    """-------------------------------DATA LOADING-----------------------------------------"""
    print("--Loading data")
    eqfile = "source/FeynmanEquations.csv"
    reference = pd.read_csv(eqfile)
    
    print("eqN: " + str(eqN) + ", file: " + reference["Filename"][eqN])
    data = np.loadtxt(reference["Filename"][eqN])
    sampleind = np.random.randint(0,10**6,1000)
    
    datafile = reference["Filename"][eqN]
    data = np.loadtxt(datafile)
    
    var_number = int(reference["# variables"][eqN])
    var_names = [reference["v"+str(n)+"_name"][eqN] for n in range(1, var_number + 1)] + [reference["Output"][eqN]]
    var_probs = [1/var_number]*var_number #+ [0.1/var_number]*var_number# + [0.1/var_number]*var_number
                                                     
    """-----------------------------GRAMMAR SETTINGS----------------------------------------"""        
    symbols = {"start":"S", "const":"C", "x":["'"+v+"'" for v in var_names]}
    functions1 = ["sin", "cos", "tan", "sqrt", "exp"]
    functions2 = ["asin", "acos", "atan", "sinh", "cosh", "tanh"]
    p_fun = [5]*len(functions1) + [1]*len(functions2)
    p_fun = 0.4 * np.array(p_fun) / np.sum(p_fun)
    
    if units:
        print("--Preparing units")
        funits = {}
        with open("source/feynman-units.csv", "r") as file:
            file.readline()
            for line in file:
                a = line.split(",")
                unit = unit_to_string([int(a[i]) for i in range(2,7)])
                funits[a[0]] = unit
                ufunits = list(set(list(funits.values())))
        print("--Preparing dimensional EqDisco")
        units = [string_to_unit(funits[var]) for var in var_names]
        target_var = reference["Output"][eqN]
        units += [string_to_unit(funits[target_var])]
        ED = pg.EqDisco(data = data, variable_names = var_names, generator_template_name = grammar_name+"-dim", 
                        variable_probabilities = var_probs,
                        generator_settings = {"functions": functions1+functions2, "p_functs": [0.6]+list(p_fun),
                                              "units": units, "extended_units": True},
                        strategy_settings = {"N": max_N, "max_total_repeats": max_repeats},
                        repeat_limit = max_attempts,
                        depth_limit = max_depth)
    
    else:
        print("--Preparing EqDisco")
        ED = pg.EqDisco(data = data, variable_names = var_names, generator_template_name = grammar_name, 
                        variable_probabilities = var_probs,
                        generator_settings = {"functions": functions1+functions2, "p_functs": [0.6]+list(p_fun)},
                        strategy_settings = {"N": max_N, "max_total_repeats": max_repeats},
                        repeat_limit = max_attempts,
                        depth_limit = max_depth)
    
    print("--Generating models")
    models = ED.generate_models()
    
    print("--Exporting generated models to /results")
    with open("results/eq" + str(eqN) + ".models", "wb") as file:
        pickle.dump(models, file)
    
    print("--Fitting models")
    fit_models = ED.fit_models()
    
    print("--Exporting fit models to /results")
    with open("results/eq" + str(eqN) + "_fit.models", "wb") as file:
        pickle.dump(fit_models, file)
        