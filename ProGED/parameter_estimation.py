# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:12:29 2020

@author: Jure
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import sympy as sp
# import sympy.core as sp
# from nltk import PCFG

# from model import Model
from model_box import ModelBox
# from generate import generate_models
# from generators.grammar import GeneratorGrammar

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

def ode (models_list, params_matrix, T, X_data, y0):
    """Solve system of ODEs defined by equations in models_list.

    Raise error if input is incompatible.
        Input:
    models_list -- list (not dictionary) of models that e.g.
    generate_models() generates.
    params_matrix -- list of lists or ndarrays of parameters for
    corresponding models.
    y0 -- array (1-dim) of initial value of vector function y(t)
    i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...].
    X_data -- 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
    T -- (1-dim) array, i.e. of shape (N,)
        Output:
    Solution of ODE evaluated at times T.
    """
    if not (isinstance(models_list, list)
            and (isinstance(params_matrix, list)
                and len(params_matrix)>0
                and isinstance(params_matrix[0], (list, np.ndarray)))
            and X_data.ndim == 2
            and y0.ndim == 1):
        print(type(params_matrix[0]))
        print(isinstance(models_list, list),
            isinstance(params_matrix, list),
            len(params_matrix)>0,
            isinstance(params_matrix[0], (list, np.ndarray)),
            X_data.ndim == 2,
            y0.ndim == 1 )
        print("Programmer's defined error: Input arguments are not"
                        +" in required form!")
        raise TypeError("Programmer's defined error: Input arguments are not"
                        +" in required form!")
    elif not T.shape[0] == X_data.shape[0]:
        print("Number of samples in T and X does not match.")
        raise IndexError("Number of samples in T and X does not match.")
    elif not (y0.shape[0] == len(models_list)  #len(equations)=len(models used)
            and len(models_list[0].sym_vars) == X_data.shape[1] + y0.shape[0]):
        print("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")
        raise IndexError("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")
    X = interp1d(T, X_data, axis=0, kind='cubic', fill_value="extrapolate")  # N-D
    lamb_exprs = [
        sp.lambdify(model.sym_vars, model.full_expr(*params), "numpy")
        for model, params in zip(models_list, params_matrix)
    ]
    def dy_dt(t, y):  # \frac{dy}{dt} ; # y = [y1,y2,y3,...] # ( shape= (n,) )
        # N-D:
        b = np.concatenate((y, X(t))) # =[y,X(t)] =[y,X1(t),X2(t),...]
        return np.array([lamb_expr(*b) for lamb_expr in lamb_exprs])  # older version with *b.T
    Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)
    return Yode.y

def model_ode_error (model, params, T, X, Y):
    """Defines mean squared error of solution to differential equation
    as the error metric.

        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns without features that are derived.
        - Y are columns of features that are derived via ode fitting.
    """
    model_list = [model]; params_matrix = [params] # 12multi conversion (temporary)
    try:
        odeY = ode(model_list, params_matrix, T, X, y0=Y[0]) # spremeni v Y[:1]
    except Exception as error:
        print("error inside ode() of model_ode_error.")
        print("params at error:", params, "Error message:", error)
        odeY = ode(model_list, params_matrix, T, X, y0=Y[0]) # spremeni v Y[:1]
    odeY = odeY.T  # solve_ivp() returns in oposite (DxN) shape.
    res = np.mean((Y-odeY)**2)
    if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#        print(model.expr, model.params, model.sym_params, model.sym_vars)
        return 10**9
    return res

def model_error_general (model, params, X, Y, T="algebraic"):
    """Calculate error of model with given parameters in general with
    type of error given.

        Input = TODO:
    - T is column of times at which samples in X and Y happen.
    - X are columns without features that are derived.
    - Y are columns of features that are derived via ode fitting.
    """
    if isinstance(T, str):
        return model_error(model, params, X, Y)
    else:
        return model_ode_error(model, params, T, X, Y)

def optimization_wrapper (x, *args):
    """Calls the appropriate error function. The choice of error function is made here.
    
    TODO:
        We need to pass information on the choice of error function from fit_models all the way to here,
            and implement a library framework, similarly to grammars and generation strategies."""
    
    return model_error (args[0], x, args[1], args[2])

def optimization_wrapper_ODE (x, *args):
    """Calls the appropriate error function. The choice of error function is made here.
    
    TODO:
        We need to pass information on the choice of error function from fit_models all the way to here,
            and implement a library framework, similarly to grammars and generation strategies."""
    return model_ode_error(args[0], x, args[3], args[1], args[2])
    
def DE_fit (model, X, Y, p0, T="algebraic", **kwargs):
    """Calls scipy.optimize.differential_evolution. 
    Exists to make passing arguments to the objective function easier."""
    
    bounds = [[-10**1, 10**1] for i in range(len(p0))]
    if isinstance(T, str):
        return differential_evolution(optimization_wrapper, bounds, args = [model, X, Y],
                                    maxiter=10**2, popsize=10)
    else:
        return differential_evolution(optimization_wrapper_ODE, bounds, args = [model, X, Y, T],
                                    maxiter=10**2, popsize=10)

def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def find_parameters (model, X, Y, T="algebraic"):
    """Calls the appropriate fitting function. 
    
    TODO: 
        add method name input, matching to a dictionary of fitting methods.
    """
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov

    res = DE_fit(model, X, Y, p0=model.params, T=T)

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
    def __init__(self, X, Y, T="algebraic"):
        self.X = X
        self.Y = Y
        self.T = T
        
    def fit_one (self, model):
        print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > 5:
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun":model_error_general(model, [], self.X, self.Y, self.T)})
            else:
                res = find_parameters(model, self.X, self.Y, self.T)
                model.set_estimated(res)
        except Exception as error:
            print(f"Excepted an error: {error}!! \nModel:", model)
            model.set_estimated({}, valid=False)
        return model
    
def fit_models (models, X, Y, T="algebraic", pool_map = map, verbosity=0):
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
    estimator = ParameterEstimator(X, Y, T)
    return ModelBox(dict(zip(models.keys(), list(pool_map(estimator.fit_one, models.values())))))



if __name__ == "__main__":
    print("--- parameter_estimation.py test --- ")
    np.random.seed(2)
    
    from pyDOE import lhs
    from generators.grammar import GeneratorGrammar
    from generate import generate_models

    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5
    
    X = lhs(2, 10)*5
    y = testf(X)
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10
    
    models = generate_models(grammar, symbols, strategy_parameters = {"N":10})
    
    models = fit_models(models, X, y)    
    print(models)

