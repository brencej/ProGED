# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:43:01 2020

@author: Jure
"""

import numpy as np
# import sympy as sp
# from nltk import PCFG

# from model import Model
# from model_box import ModelBox
from generate import generate_models
from parameter_estimation import fit_models
from generators.base_generator import BaseExpressionGenerator
# from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template
from task import EDTask

GENERATOR_LIBRARY = {"grammar": grammar_from_template}

class EqDisco:
    def __init__ (self, task = None,  dataX = None, dataY = None, variable_names = None, output_variable = None, 
                  success_threshold = 1e-8, task_type = "algebraic",
                  generator = "grammar", generator_template_name = "universal", generator_parameters = {},
                  strategy = "monte-carlo", strategy_parameters = {"N":5},
                  verbosity = 1):        
        
        if not task:
            if isinstance(dataX, type(None)) or isinstance(dataY, type(None)):
                raise TypeError ("Missing inputs. Either task object or data required.")
            else:
                if not variable_names:
                    variable_names = [chr(97+i) for i in range(dataX.shape[-1])]
                if not output_variable:
                    output_variable = "f"
                self.task = EDTask(dataX, dataY, variable_names, output_variable, success_threshold, task_type)
        elif isinstance(task, EDTask):
            self.task = task
        else:
            raise TypeError ("Missing task information!")
            
        generator_parameters.update({"variables":self.task.symbols["x"]})
        if isinstance(generator, BaseExpressionGenerator):
            self.generator = generator
        elif isinstance(generator, str):
            if generator in GENERATOR_LIBRARY:
                self.generator = GENERATOR_LIBRARY[generator](generator_template_name, 
                                                              generator_parameters)
            else:
                raise KeyError("Generator name not found. Supported generators:\n" + str(list(GENERATOR_LIBRARY.keys())))
        else:
            raise TypeError ("Invalid generator specification. Expected: class that inherits from "\
                             "generators.base_generator.BaseExpressionGenerator or string, corresponding to template name.\n"\
                             "Input: " + str(type(generator)))
            
        self.strategy = strategy
        self.strategy_parameters = strategy_parameters
        
        self.models = None
        self.solution = None
        
        self.verbosity = verbosity
        
        
    def generate_models (self, strategy_parameters = None):
        if not strategy_parameters:
            strategy_parameters = self.strategy_parameters
        self.models = generate_models(self.generator, self.task.symbols, self.strategy, strategy_parameters, verbosity=self.verbosity)
        return self.models
    
    def fit_models (self, pool_map = map):
        self.models = fit_models(self.models, self.task.dataX, self.task.dataY, pool_map = pool_map, verbosity=self.verbosity)
        return self.models
        
    
if __name__ == "__main__":
    print("--- equation_discoverer.py test --- ")
    np.random.seed(2)
    
    from pyDOE import lhs
    
    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5
    
    X = lhs(2, 10)*5
    y = testf(X)
    
    N = 2
    disco = EqDisco(dataX = X, dataY = y, strategy_parameters = {"N":10}, verbosity = 0,
                    generator="grammar", generator_template_name="universal")
    
    print(disco.generate_models())
    print(disco.fit_models())
