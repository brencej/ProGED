# -*- coding: utf-8 -*-
# todo: __main__ update to dataFrames
import json

import numpy as np
from copy import deepcopy

from ProGED.generate import generate_models
from ProGED.parameter_estimation import fit_models
from ProGED.generators.base_generator import BaseExpressionGenerator
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.task import EDTask
from ProGED.postprocessing import models_statistics
from ProGED.configs import settings

"""
User-facing module for straightforward equation discovery tasks.

Modules:
    EqDisco: Interface for streamlined probabilistic grammar-based equation discovery.
    
Attributes:
    GENERATOR_LIBRARY (dict): Library of available types of generators.
        Currently, ProGED supports only probabilistic context-free grammars as generators.
"""

GENERATOR_LIBRARY = {"grammar": grammar_from_template}


class EqDisco:
    """
    EqDisco provides a modular interface to the ProGED system that simplifies the workflow for common problems in equation discovery.
    
    An instance of this class represents the equation discovery algorithm for a single problem. A typical workflow using EqDisco would look like:
            ED = EqDisco (inputs)
            ED.generate_models()
            ED.fit_models()
    
    Arguments:
        task (ProGED.EDTask): Instance of EDTask, containing specifications of the equation discovery problem. 
            If not provided, created by EqDisco based on other arguments.
        data (numpy.array): Input data of shape N x M, where N is the number of samples 
            and M is the number of variables. Not required if 'task' is provided.
        target_variable_index (int):  Index of column in data that belongs to the target variable.
            Not required if 'task' is provided. Default -1 (last column).
        time_index (int): Index of column in data that belongs to measurement of time. 
            Required for differential equations, None otherwise. Not required if 'task' is provided.
        variable_names (list of strings): Names of input variables. If not provided, names will be auto-generated. 
            Not required if 'task' is provided.
        constant_symbol (string): String to be used as symbol for constant parameter. Default: "C".
        task_type (string): Specifies type of equation being solved. See ProGED.task.TASK_TYPES for supported equation types. 
            Default: algebraic. Not required if 'task' is provided.
        system_size (int): if discovering system of equations, system_size defines the number of equations. Default: 1.
        success_threshold (float): Relative root mean squared error (RRMSE), 
            below which a model is considered to be correct. Default: 1e-8.
        generator (ProGED.generators.BaseExpressionGenerator or string): Instance of generator, deriving from BaseExpressionGenerator 
            or a string matching a geenrator type from GENERATOR_LIBRARY. Default: 'grammar'.
            If string, the instance will be created by EqDisco based on other arguments.
        generate_template_name (string): If constructing a grammar from the library, use this to specify
            the template name. Not required if a generator instance is provided. Default: 'universal'.
        variable_probabilities (list of floats): Prior probability distribution over variable symbols. If not provided, a uniform
            distribution is assumed. Not required if a generator instance is provided.
        generator_settings (dict): Arguments to be passed to the generator constructor. See documentation
            of the specific generator for possible settings. Has no effect if a generator instance is provided.
        strategy (string): Name of sampling strategy from STRATEGY_LIBRARY. Default: 'monte-carlo'.
        strategy_settings (dict): Arguments to be passed to the chosen sampling strategy function.
            See documentation for the specific strategy for available options.
            For Monte-Carlo sampling, the available options are:
                N (int): Total number of candidate equations to generate.
                max_repeat (int): Sometimes generation encounters an error. In such cases, the generator will
                    attempt to repeat the generation at most max_repeat times.
        sample_size (int): Total number of candidate equations to sample when using Monte-Carlo. 
            Irrelevant when strategy_settings is provided. Default: 10.
        estimation_settings (dict): Arguments to be passed to the system for parameter estimation.
            See documentation for ProGED.fit_models for details and available options. Optional.
        verbosity (int): Level of printout desired. 0: none, 1:info, 2+: debug. 
        
    Methods:
        generate_models (strategy_settings = None): Generates candidate equations according to the specifications
            in task, generator and strategy settings. Constructs a ProGED.ModelBox, containing a ProGED.Model 
            for each unique generated equation. The ModelBox is stored within the EqDisco instance.
            
            Arguments:
                strategy_settings (dict): Arguments to be passed to the chosen sampling strategy function.
                    Optional. If None, strategy_settings, stored in the EqDisco instance are passed to generate_models.
                    See documentation of sampling strategy functions for available options.
            Returns:
                ProGED.ModelBox of generated models
                
        fit_models (estimation_settings = {}, pool_map = map): Estimates parameters for each of the 
            generated models. 
            
            Arguments:
                estimation_settings (dict): Arguments to be passed to the system for parameter estimation.
                    See documentation for ProGED.fit_models for details and available options. Optional.
                    If None, estimation_settings, stored in the EqDisco instance are passed to fit_models.
                pool_map (funct): Map function to be used when calling parameter estimation on the
                    array of candidate equations. Allows for simple parallelization using multiprocessing
                    By default, the python map function is used, which provides no paralellization.
                    Example usage with eight workers:
                        from multiprocessing import Pool
                        pool = Pool(8)
                        
                        ED = EqDisco(...)
                        ED.generate_models()
                        ED.fit_models (pool_map = pool.map)
            Returns:
                ProGED.ModelBox of models with fitted parameters.
    """

    def __init__(self,
                 task=None,
                 data=None,
                 rhs_vars=None,
                 lhs_vars=None,
                 constant_symbol="C",
                 task_type="algebraic",  # deprecated: this should be inferred from configs.py/settings['parameter_estimation']['task_type'] or estimation_settings['par_esti']['task_type'].
                 system_size=1,  # deprecated: this should be inferred from lhs_vars (system_size = len(lhs_vars)).
                 generator="grammar",
                 generator_template_name="universal",
                 variable_probabilities=None,
                 generator_settings={},
                 strategy="monte-carlo",
                 strategy_settings={},
                 sample_size=10,
                 max_attempts=1,
                 repeat_limit=100,
                 depth_limit=1000,
                 estimation_settings={},
                 success_threshold=1e-8,
                 verbosity=1):
        
        if not task:
            if isinstance(data, type(None)):
                raise TypeError("Missing inputs. Either task object or data required.")
            else:
                self.task = EDTask(data=data,
                                   lhs_vars=lhs_vars,
                                   rhs_vars=rhs_vars,
                                   constant_symbol=constant_symbol,
                                   success_threshold=success_threshold,
                                   task_type=task_type)
                
        elif isinstance(task, EDTask):
            self.task = task
        else:
            raise TypeError("Missing task information!")
        
        if not variable_probabilities:
            # variable_probabilities = [1/len(self.task.var_names)]*np.sum(self.task.variable_mask)
            variable_probabilities = [1/len(self.task.rhs_vars)]*len(self.task.rhs_vars)
        generator_settings.update({"variables":self.task.symbols["x"], "p_vars": variable_probabilities})
        if isinstance(generator, BaseExpressionGenerator):
            self.generator = generator
        elif isinstance(generator, str):
            if generator in GENERATOR_LIBRARY:
                self.generator = GENERATOR_LIBRARY[generator](generator_template_name, 
                                                              generator_settings,
                                                              repeat_limit=repeat_limit,
                                                              depth_limit=depth_limit)
            else:
                raise KeyError("Generator name not found. Supported generators:\n" + str(list(GENERATOR_LIBRARY.keys())))
        else:
            raise TypeError ("Invalid generator specification. Expected: class that inherits from "\
                             "generators.base_generator.BaseExpressionGenerator or string, corresponding to template name.\n"\
                             "Input: " + str(type(generator)))

        self.system_size = system_size
            
        self.strategy = strategy
        self.strategy_settings = {"N": sample_size, "max_repeat": max_attempts}
        self.strategy_settings.update(strategy_settings)

        self.estimation_settings = deepcopy(settings)
        self.estimation_settings.update(
            {"lhs_vars": self.task.lhs_vars,
             "verbosity": verbosity})
        self.estimation_settings.update(estimation_settings)
        
        self.models = None
        self.solution = None
        
        self.verbosity = verbosity

    def generate_models(self, strategy_settings={}):
        strategy_settings_preset = deepcopy(self.strategy_settings)
        strategy_settings_preset.update(strategy_settings)
        self.models = generate_models(self.generator, self.task.symbols,
                                      lhs_vars=self.task.lhs_vars,
                                      rhs_vars=self.task.rhs_vars,
                                      system_size=self.system_size,
                                      strategy=self.strategy,
                                      strategy_settings=strategy_settings_preset,
                                      verbosity=self.verbosity)
        return self.models
    
    def fit_models(self, settings={}, pool_map=map):

        estimation_settings_preset = deepcopy(self.estimation_settings)
        estimation_settings_preset.update(settings)

        self.models = fit_models(self.models, self.task.data,
                                 pool_map=pool_map,
                                 settings=estimation_settings_preset)
        return self.models
    
    def get_results(self, N=3):
        return self.models.retrieve_best_models(N)
    
    def get_stats (self):
        return models_statistics(self.models, 
                                 self.task.data, 
                                 self.task.lhs_vars[0],
                                 self.task.success_thr)

    def write_results(self, filename=None, dummy=10**8):
        res = []
        models = self.models.models_dict
        for eq in models.keys():
            model = models[eq]
            for tree in model.trees.keys():
                try:
                    res.append({"eq": eq, "error": model.get_error(dummy=dummy), "p": model.p, "code": tree})
                except Exception as error:
                    print(f"Failed to write results for equation {eq}.")
        res = sorted(res, key=lambda x: x["error"])
        if filename is not None:
            with open(filename, 'w') as file:
                json.dump(res, file)
        else:
            return res


if __name__ == "__main__":
    print("--- equation_discoverer.py test --- ")
    import pandas as pd
    np.random.seed(0)

    # model: 2 (x + 0.3)
    def f(x):
        return 2.0 * (x + 0.3)
    X = np.linspace(-1, 1, 20)
    Y = f(X)
    X = X.reshape(-1, 1)
    Y = Y.reshape(-1, 1)
    data = pd.DataFrame(np.hstack((X, Y)), columns=["x", "y"])
        
    ED = EqDisco(task=None, data=data, rhs_vars=["x"], lhs_vars=["y"], sample_size=1, verbosity=1)
    ED.generate_models()
    ED.fit_models()
    print(ED.get_results())
    print("-----------------------\n")
    print(ED.get_stats())
