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

def model_error (model, params, X, Y):
    testY = model.evaluate(X, *params)
    res = np.mean((Y-testY)**2)
    if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#        print(model.expr, model.params, model.sym_params, model.sym_vars)
        return 10**9
    return res

def model_constant_error (model, params, X, Y):
    testY = model.evaluate(X, *params)
    return np.std(testY)#/np.linalg.norm(params)

def optimization_wrapper (x, *args):
    return model_error (args[0], x, args[1], args[2])
    
def DE_fit (model, X, Y, p0, **kwargs):
    bounds = [[-10**1, 10**1] for i in range(len(p0))]
    return differential_evolution(optimization_wrapper, bounds, args = [model, X, Y],
                                  maxiter=10**2, popsize=10)
    
def min_fit (model, X, Y):
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def find_parameters (model, X, Y):
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov
    
    res = DE_fit (model, X, Y, p0=model.params)
    
#    res = min_fit (model, X, Y)
#    opt_params = res.x; othr = res
    
    return res

class EstimationWrapper:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    def __call__(self, model):
        return find_parameters(model, self.X, self.Y)

class ParameterEstimator:
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
    