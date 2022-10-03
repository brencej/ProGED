# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 14:32:17 2021

@author: jureb
"""

import glob
import sys

import ProGED as pg

if __name__ == "__main__":
    files = glob.glob("*fit_models.pg")
    N = int(sys.argv[1])
    
    outfile = sys.argv[2]
    successes = 0
    
    for file in files:
        print("---" + file)
        models = pg.ModelBox()
        models.load(file)
            
        best_models = models.retrieve_best_models(N)
        if len(best_models) > 0:
            if best_models[0].get_error() < 1e-9:
                successes += 1
            
        with open(outfile, "a") as f:
            f.write("\n---" + file + ", unique models: " + str(len(models))+"\n")
            f.write(str(best_models))
        
    with open(outfile, "a") as f:
        f.write("-------- successes: " + str(successes) + " ----------------")
    print("Successes: ", successes)
            
        
    