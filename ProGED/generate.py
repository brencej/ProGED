# -*- coding: utf-8 -*-

from ProGED.model_box import ModelBox
from ProGED.generators.base_generator import ProGEDMaxAttemptError


"""Functions for generating models using a given generator. 

Attributes:
    STRATEGY_LIBRARY (dict): Dictionary defining the implemented strategies. Items:
        key (string): strategy name
        value (function): function implementing the generation strategy with arguments:
            model_generator (BaseExpressionGenerator): generator object to be used,
            symbols (dict): dictionary containing:
                "start": start symbol (str)
                "const": unique symbol representing free constants, to be later enumerated (str)
                "x": list of strings, representing variable symbols (list of strings)
            kwargs: a dictionary of keyword arguments to be passed to the strategy function.
                See specific strategy for details.

Methods:
    generate_models: High-level interface to the generation strategies. Intended to be called from outside.
    monte_carlo_sampling: Monte-Carlo strategy to sampling models from a model generator.
    
"""


def generate_models(model_generator, symbols, strategy = "monte-carlo", system_size=1,
                    strategy_settings = {"N": 5}, verbosity=0, **kwargs):
    """Generate models using given generator and specified strategy.
    
    generate_models is intended as an interface to the generation methods defined in the module.
    
    Arguments:
        model_generator (BaseExpressionGenerator): Model generator instance. Should inherit from 
            BaseExpressionGenerator and implement generate_one.
        symbols (dict): Dictionary containing:
                "start": start symbol (str)
                "const": unique symbol representing free constants, to be later enumerated (str)
                "x": list of strings, representing variable symbols (list of strings)
        strategy (str): Name of strategy, as defined in STRATEGY_LIBRARY. 
            Currently only the Monte-Carlo method is implemented.
        strategy_settings (dict): Dictionary of keywords to be passed to the generator function.
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
        
    Returns:
        ModelBox instance, containing the generated models.
    """
    if isinstance(strategy, str):
        if strategy in STRATEGY_LIBRARY:
            return STRATEGY_LIBRARY[strategy](model_generator, symbols, system_size=system_size,
                                              verbosity=verbosity, **strategy_settings, **kwargs)
        else:
            raise KeyError ("Strategy name not found in library.\n"\
                            "Input: " + strategy)
                
    elif isinstance(strategy, lambda x: x):
        return strategy(model_generator, symbols, **strategy_settings, **kwargs)
    
    else:
        raise TypeError ("Unknown strategy type. Expecting: string or callable.\n"\
                         "Input: " + str(type(strategy)))

def monte_carlo_sampling (model_generator, symbols, N=5, system_size=1, max_repeat=10,
                          max_total_repeats=None, verbosity=0, **kwargs):
    """Generate models using the Monte-Carlo approach to sampling.
    
    Randomly sample the stochastic generator until N models have been generated. 
    Models that ModelBox considers invalid, according to ModelBox.add_model, are rejected.
    The sampling is repeated until a valid model is generated or max_repeat attempts are made.
    
    Arguments:
        model_generator (BaseExpressionGenerator): Model generator instance. Should inherit from 
            BaseExpressionGenerator and implement generate_one.
        symbols (dict): Dictionary containing:
                "start": start symbol (str)
                "const": unique symbol representing free constants, to be later enumerated (str)
                "x": list of strings, representing variable symbols (list of strings)
        N (int): Number of (valid) models to sample.
        max_repeat (int): Number of allowed repeated attempts at sampling a single model.
        max_total_repeats (int): Total allowed number of calls to generate_one. The default None is interpreted by setting limit, higher than N*max_repeat.
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
    
    Returns:
        ModelBox instance containing the generated models.
    """
    models = ModelBox()
    
    if not max_total_repeats:
        max_total_repeats = N*max_repeat
    
    current_total = 0
    while len(models) < N and current_total < max_total_repeats:
        good = False
        n = 0
        try:
            while not good and n < max_repeat:
                exprs = []; codes = []; p = 1

                for _ in range(system_size):
                    sample, pi, code = model_generator.generate_one()
                    exprs += ["".join(sample)]
                    codes += [code]
                    p *= pi
                codes = ",".join(codes)

                if verbosity > 1:
                    print("-> ", exprs, p, codes)
                    
                valid, expr = models.add_model(exprs, symbols, p=p,
                                               info={"grammar": model_generator, "code": codes},
                                               **kwargs)
                
                if verbosity > 1:
                    print("---> ", valid, expr)
                    
                if valid:
                    good = True
                n += 1
    
            if verbosity > 0 and len(models) > 0:
                print(models[-1])
        except ProGEDMaxAttemptError:
            n += 1
        
        current_total += n
        
    return models

def monte_carlo_sampling_2 (model_generator, symbols, N=5, max_repeat = 10, max_total_repeats = None,
                            verbosity=0, **kwargs):
    """Generate models using the Monte-Carlo approach to sampling.
    
    Randomly sample the stochastic generator until N models have been generated. 
    Models that ModelBox considers invalid, according to ModelBox.add_model, are rejected.
    The sampling is repeated until a valid model is generated or max_repeat attempts are made.
    
    Arguments:
        model_generator (BaseExpressionGenerator): Model generator instance. Should inherit from 
            BaseExpressionGenerator and implement generate_one.
        symbols (dict): Dictionary containing:
                "start": start symbol (str)
                "const": unique symbol representing free constants, to be later enumerated (str)
                "x": list of strings, representing variable symbols (list of strings)
        N (int): Number of (valid) models to sample.
        max_repeat (int): Number of allowed repeated attempts at sampling a single model.
        max_total_repeats (int): Total allowed number of calls to generate_one. The default None is interpreted by setting limit, higher than N*max_repeat.
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
    
    Returns:
        ModelBox instance containing the generated models.
    """
    models = ModelBox()
    
    if not max_total_repeats:
        max_total_repeats = N*max_repeat
    
    current_total = 0
    while len(models) < N and current_total < max_total_repeats:
        good = False
        n = 0
        try:
            while not good and n < max_repeat:
                sample, p, code = model_generator.generate_one()
                expr_str = "".join(sample)
                
                if verbosity > 1:
                    print("-> ", expr_str, p, code)
                    
                valid, expr = models.add_model(expr_str, symbols, model_generator, code=code, p=p, **kwargs)
                
                if verbosity > 1:
                    print("---> ", valid, expr)
                    
                if valid:
                    good = True
                n += 1
    
            if verbosity > 0 and len(models) > 0:
                print(models[-1])
        except ProGEDMaxAttemptError:
            n += 1
        
        current_total += n
        
    return models

STRATEGY_LIBRARY = {"monte-carlo": monte_carlo_sampling}


if __name__ == "__main__":
    print("--- generate.py test ---")
    import numpy as np
    from generators.grammar_construction import grammar_from_template
    np.random.seed(0)
    generator = grammar_from_template("polynomial", {"variables":["'x'", "'y'"], "p_vars":[0.3,0.7]})
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10
    
    models = generate_models(generator, symbols, strategy_settings = {"N":10})
    
    print(models)
                        
