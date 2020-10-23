# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 17:34:31 2020

@author: Jure
"""

import numpy as np

from model import Model
from model_box import ModelBox

from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template


def generate_models(model_generator, symbols, strategy = "monte-carlo", strategy_parameters = {"N":5}, verbosity=0):
    if isinstance(strategy, str):
        if strategy in STRATEGY_LIBRARY:
            return STRATEGY_LIBRARY[strategy](model_generator, symbols, verbosity=verbosity,  **strategy_parameters)
        else:
            raise KeyError ("Strategy name not found in library.\n"\
                            "Input: " + strategy)
                
    elif isinstance(strategy, lambda x: x):
        return strategy(model_generator, symbols, **strategy_parameters)
    
    else:
        raise TypeError ("Unknown strategy type. Expecting: string or callable.\n"\
                         "Input: " + str(type(strategy)))

def monte_carlo_sampling (model_generator, symbols, N=5, max_repeat = 10, verbosity=0):
    x = [s.strip("'") for s in symbols["x"]]
    models = ModelBox()
    
    for n in range(N):
        good = False
        n = 0
        while not good and n < max_repeat:
            sample, p, code = model_generator.generate_one()
            expr_str = "".join(np.array(sample))
            
            if verbosity > 1:
                print("-> ", expr_str, p, code)
                
            valid, expr = models.new_model(expr_str, symbols, model_generator, code=code, p=p)
            
            if verbosity > 1:
                print("---> ", valid, expr)
                
            if valid:
                good = True
            n += 1
        if verbosity > 0 and len(models) > 0:
            print(models[-1])
            
    return models

STRATEGY_LIBRARY = {"monte-carlo": monte_carlo_sampling}


if __name__ == "__main__":
    print("--- generate.py test ---")
    np.random.seed(0)
    generator = grammar_from_template("polynomial", {"variables":["'x'", "'y'"], "p_vars":[0.3,0.7]})
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10
    
    models = generate_models(generator, symbols, strategy_parameters = {"N":10})
    
    print(models)
                        