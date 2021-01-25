# -*- coding: utf-8 -*-
"""Class for defining the equation discovery task.

TODO:
    Consider whether this could be just a dictionary.
"""

import numpy as np

class EDTask:
    def __init__(self, 
                 data = None, 
                 target_variable_index = None, 
                 time_index = None,
                 variable_names = None,
                 success_threshold = 1e-8, 
                 task_type = "algebraic"):
        """Initialize an equation discovery task specification.
        
        Arguments:
            data (numpy array): Input data of shape N x M, where N is the number of samples 
                and M is the number of variables.
            target_variable_index (int): Index of column in data that belongs to the target variable.
            time_index (int): Index of column in data that belongs to measurement of time. 
                Required for differential equations, None otherwise.
            variable_names (list of strings): Names of input variables.
            success_threshold (float): Maximum allowed error for considering a model to be correct.
            task_type (string): Type of ED task. Currently implemented:
                algebraic
                differential
        """
        
        self.variable_mask = np.ones(data.shape[-1], bool)
        
        if task_type == "differential":
            if not time_index:
                raise TypeError ("Missing temporal data. Temporal data is required for differential equation task type."\
                                 "Specify index of temporal data column as time_index.")
            self.variable_mask[time_index] = False
        self.time_index = time_index
        
        self.variable_mask[target_variable_index] = False
        
        if not variable_names:
            self.var_names = np.array([chr(97+i) for i in range(data.shape[-1])])
        else:
            self.var_names = np.array(variable_names)

        self.task_type = task_type
        self.target_variable_index = target_variable_index
        self.data = data
        self.success_thr = success_threshold
        
        self.symbols = {"start":"E", "const": "C", "x": ["'" + v + "'" for v in self.var_names[self.variable_mask]]}
        
if __name__ == "__main__":
    print("--- task.py test ---")
    import numpy as np
    
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1, 5]).reshape(2,1)
    data = np.hstack((X,y))
    
    task = EDTask(data, -1, ["x", "y", "f"])
    
    print(task.symbols)
     
     
