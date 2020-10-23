# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:12:29 2020

@author: Jure
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from nltk import PCFG

from model import Model
from model_box import ModelBox
from generate import generate_models

"""Methods for estimating model parameters. Currently implemented: differential evolution.

Methods:
    fit_models: Performs parameter estimation on given models. Main interface to the module.
"""

def model_error (model, params, X, Y):
    """Defines mean squared error as the error metric."""
    testY = model.evaluate(X, *params)
    res = np.mean((Y-testY)**2)
    if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#        print(model.expr, model.params, model.sym_params, model.sym_vars)
        return 10**9
    return res

def model_constant_error (model, params, X, Y):
    """Alternative to model_error, intended to allow the discovery of physical constants.
    Work in progress."""
    
    testY = model.evaluate(X, *params)
    return np.std(testY)#/np.linalg.norm(params)

def optimization_wrapper (x, *args):
    """Calls the appropriate error function. The choice of error function is made here.
    
    TODO:
        We need to pass information on the choice of error function from fit_models all the way to here,
            and implement a library framework, similarly to grammars and generation strategies."""
    
    return model_error (args[0], x, args[1], args[2])
    
def DE_fit (model, X, Y, p0, **kwargs):
    """Calls scipy.optimize.differential_evolution. 
    Exists to make passing arguments to the objective function easier."""
    
    bounds = [[-10**1, 10**1] for i in range(len(p0))]
    return differential_evolution(optimization_wrapper, bounds, args = [model, X, Y],
                                  maxiter=10**2, popsize=10)
    
def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def find_parameters (model, X, Y):
    """Calls the appropriate fitting function. 
    
    TODO: 
        add method name input, matching to a dictionary of fitting methods.
    """
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov
    
    res = DE_fit (model, X, Y, p0=model.params)
    
#    res = min_fit (model, X, Y)
#    opt_params = res.x; othr = res
    
    return res

class ParameterEstimator:
    """Wraps the entire parameter estimation, so that we can pass the map function in fit_models
        a callable with only a single argument.
        Also checks some basic requirements, suich as minimum and maximum number of parameters.
        
        TODO:
            add inputs to make requirements flexible
            add verbosity input
    """
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def fit_one (self, model):
        print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > 5:
                model.set_estimated({}, valid=False)
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun":model_error(model, [], self.X, self.Y)})
            else:
                res = find_parameters(model, self.X, self.Y)
                model.set_estimated(res)
        except:
            model.set_estimated({}, valid=False)
        return model
    
def fit_models (models, X, Y, pool_map = map, verbosity=0):
    """Performs parameter estimation on given models. Main interface to the module.
    
    Supports parallelization by passing it a pooled map callable.
    
    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted. 
        X (numpy.array): Input data of shape N x M, where N is the number of samples 
            and M is the number of variables.
        Y (numpy.array): Output data of shape N x D, where N is the number of samples
            and D is the number of output variables.
        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, X, Y, pool_map = pool.map)
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
    """
    estimator = ParameterEstimator(X, Y)
    return ModelBox(dict(zip(models.keys(), list(pool_map(estimator.fit_one, models.values())))))



if __name__ == "__main__":
    print("--- parameter_estimation.py test --- ")
    np.random.seed(2)
    
    from pyDOE import lhs
    
    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5
    
    X = lhs(2, 10)*5
    y = testf(X)
    
    grammar = PCFG.fromstring("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10
    
    models = generate_models(N, grammar, symbols)
    
    models = fit_models(models, X, y)    
    print(models)
    