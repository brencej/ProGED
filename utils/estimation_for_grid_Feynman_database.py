# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 11:22:45 2021

@author: jureb
"""

import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.append(os.getcwd()+"/source")

import ProGED as pg

#warnings.filterwarnings("ignore")

np.random.seed(0)

if __name__ == "__main__":
    datadir = ""
    
    eqN = int(sys.argv[1])
    modelsfile = sys.argv[2]
    processN = int(sys.argv[3])
    
    eqfile = "source/FeynmanEquations.csv"
    reference = pd.read_csv(eqfile)
    
    print("eqN: " + str(eqN) + ", file: " + reference["Filename"][eqN])
    data = np.loadtxt(datadir + reference["Filename"][eqN])
    sampleind = np.random.randint(0,10**6,1000)
    
    print("--Loading models")
    with open(modelsfile, "rb") as file:
        models = pickle.load(file)
        
    pool = mp.Pool(processN)
    print("--Fitting models")
    models = pg.fit_models(models, data[sampleind], target_variable_index=-1, pool_map = pool.map, verbosity = 1)
    
    print("--Exporting results")
    with open("results/" + modelsfile.split(".")[0] + "_fit.models", "wb") as file:
        pickle.dump(models, file)
    
    