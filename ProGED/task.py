# -*- coding: utf-8 -*-
"""
Class for defining the equation discovery task.
"""

import numpy as np
import pandas as pd

TASK_TYPES = ("algebraic", "differential")

class EDTask:
    def __init__(self, 
                 data = None, 
                 rhs_vars = None,
                 lhs_vars = None,
                 constant_symbol = None,
                 symbols = None,
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
            constant_symbol (string): String to be used as the symbol for a free constant.
            symbols (dict): Dictionary of symbols. If None, the dictionary is constructed using 
                the variable_names and constant_symbol arguments. If all are None, default values are used. 
                Elements:
                    "const": String to be used as the symbol for a free constant.
                    "x": List of strings, representing variable names, each encased in "''".
            success_threshold (float): Maximum allowed error for considering a model to be correct.
            task_type (string): Type of ED task. Currently implemented:
                algebraic
                differential
        """

        self.data = data
        self.task_type = task_type
        self.lhs_vars = lhs_vars
        self.rhs_vars = rhs_vars
        self.success_thr = success_threshold
        
        if not symbols:
            if not constant_symbol:
                self.constant_symbol = "C"
            else:
                self.constant_symbol = constant_symbol

            self.symbols = {"start":"E", 
                            "const": self.constant_symbol, 
                            "x": ["'" + v + "'" for v in self.rhs_vars]}
        else:
            self.symbols = symbols

    def verify_task(self):
        if self.data is None:
            raise TypeError("Missing inputs. Either task object or data required.")
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("The data should be in the form of Pandas DataFrame.")
        if self.task_type == "differential" and 't' not in self.data.columns:
            raise TypeError ("Missing temporal data. Temporal data is required for differential equation task type."
                             "Specify temporal data column with column name 't'.")
        
        
if __name__ == "__main__":
    print("--- task.py test ---")
    import numpy as np
    
    X = np.array([[0, 0], [1, 1]])
    y = np.array([1, 5]).reshape(2,1)
    data = pd.DataFrame(np.hstack((X,y)), columns=["x", "y"])
    
    task = EDTask(data, -1, ["x", "y", "f"])
    
    print(task.symbols)
     
     
