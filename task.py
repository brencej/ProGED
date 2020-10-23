# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:09:35 2020

@author: Jure
"""

import numpy as np
# import sympy as sp
# from nltk import PCFG

# from model import Model
# from model_box import ModelBox
# from generate import generate_models
# from parameter_estimation import fit_models
#from grammar_construction import grammar_from_template

class EDTask:
    def __init__(self, dataX, dataY, variable_names, output_variable, success_threshold = 1e-8, task_type = "algebraic"):
        self.task_type = task_type
        self.dataX = dataX
        self.dataY = dataY
        self.var_names = variable_names
        self.output_var = output_variable
        self.success_thr = success_threshold
        
        self.symbols = {"start":"E", "const": "C", "x": ["'" + v + "'" for v in self.var_names]}
        
if __name__ == "__main__":
    print("--- task.py test ---")
    
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1, 5])
    
    task = EDTask(X, y, ["x", "y"], "f")
    
    print(task.symbols)
     
     
