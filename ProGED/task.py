# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:09:35 2020

@author: Jure
"""

# import numpy as np
# import sympy as sp
# from nltk import PCFG

# from model import Model
# from model_box import ModelBox
# from generate import generate_models
# from parameter_estimation import fit_models
#from grammar_construction import grammar_from_template

"""Class for defining the equation discovery task.

TODO:
    Consider whether this could be just a dictionary.
"""

class EDTask:
    def __init__(self, dataX, dataY, variable_names, output_variables, success_threshold = 1e-8, task_type = "algebraic"):
        """Initialize an equation discovery task specification.
        
        Arguments:
            dataX (numpy array): Input data of shape N x M, where N is the number of samples 
                and M is the number of variables.
            dataY (numpy array): Output data of shape N x D, where N is the number of samples
                and D is the number of output variables.
            variable_names (list of strings): Names of input variables.
            output_variables (list of strings): Names of output variables.
            success_threshold (float): Maximum allowed error for considering a model to be correct.
            task_type (string): Type of ED task. Currently implemented:
                algebraic
        """
        
        self.task_type = task_type
        self.dataX = dataX
        self.dataY = dataY
        self.var_names = variable_names
        self.output_var = output_variables
        self.success_thr = success_threshold
        
        self.symbols = {"start":"E", "const": "C", "x": ["'" + v + "'" for v in self.var_names]}
        
if __name__ == "__main__":
    print("--- task.py test ---")
    import numpy as np
    
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1, 5])
    
    task = EDTask(X, y, ["x", "y"], "f")
    
    print(task.symbols)
     
     
