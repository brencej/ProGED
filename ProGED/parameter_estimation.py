# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:12:29 2020

@author: Jure
"""

import os
import sys
import time

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint
import sympy as sp
# import sympy.core as sp
# from nltk import PCFG

import examples.mute_so as mt
from _io import TextIOWrapper as stdout_type
# from model import Model
from model_box import ModelBox
# from generate import generate_models
# from generators.grammar import GeneratorGrammar

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="overflow encountered in square")


"""Methods for estimating model parameters. Currently implemented: differential evolution.

Methods:
    fit_models: Performs parameter estimation on given models. Main interface to the module.
"""

def model_error (params, model, X, Y, *residue):
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

EQUATION_TYPES = ("algebraic", "differential")

def model_error_general (params, model, X, Y, T, **estimation_settings):
    """Calculate error of model with given parameters in general with
    type of error given.

        Input = TODO:
    - X are columns without features that are derived.
    - Y are columns of features that are derived via ode fitting.
    - T is column of times at which samples in X and Y happen.
    - estimation_settings: look description of fit_models()
    """
    equation_type = estimation_settings["equation_type"]
    if equation_type == "algebraic":
        return model_error(params, model, X, Y)
    elif equation_type == "differential":
        # Model_ode_error might use estimation[verbosity] agrument for
        # ode solver's settings and suppresing its warnnings:
        return model_ode_error(params, model, X, Y, T, estimation_settings)
    else:
        types_string = "\", \"".join(EQUATION_TYPES)
        raise ValueError("Variable equation_type has unsupported value "
                f"\"{equation_type}\", while list of possible values: "
                f"\"{types_string}\".")

def ode (models_list, params_matrix, T, X_data, y0, **estimation_settings):
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
        # todo: model.lambdify(params=params, args="numpy")
        for model, params in zip(models_list, params_matrix)
    ]
    def dy_dt(t, y):
        """Represents  \frac{dy}{dt}.

        y -- [y1,y2,y3,...] i.e. ( shape= (n,) ) """

        # N-D:
        b = np.concatenate((y, X(t))) # =[y,X(t)] =[y,X1(t),X2(t),...]
        # Older version with *b.T:
        return np.array([lamb_expr(*b) for lamb_expr in lamb_exprs])
    # Older (default RK45) method:
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, atol=0)  
    # Set min_step via prescribing maximum number of steps:
    if "max_ode_steps" in estimation_settings:
        max_steps = estimation_settings["max_ode_steps"]
    else:
        # max_steps = 10**6  # On laptop, this would need less than 3 seconds.
        max_steps = T.shape[0]*10**3  # Set to |timepoints|*1000.
    # Convert max_steps to min_step:
    min_step_from_max_steps = abs(T[-1] - T[0])/max_steps
    # The minimal min_step to avoid min step error in LSODA:
    min_step_error = 10**(-15)
    min_step = max(min_step_from_max_steps, min_step_error)  # Force them both.
    rtol = 10**(-4)
    atol = 10**(-6)
    # Yode = solve_ivp(dy_dt, (T[0], T[-1]), y0, t_eval=T, method="LSODA", rtol=rtol, atol=atol, min_step=min_step).y
    # Alternative LSODA using odeint (may be faster?):
    Yode = odeint(dy_dt, y0, T, rtol=rtol, atol=atol, tfirst=True, hmin=min_step).T 
    return Yode

def model_ode_error (params, model, X, Y, T, estimation_settings):
    """Defines mean squared error of solution to differential equation
    as the error metric.

        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns without features that are derived.
        - Y are columns of features that are derived via ode fitting.
    """
    model_list = [model]; params_matrix = [params] # 12multi conversion (temporary)
    dummy = 10**9
    try:
        # Next few lines strongly suppress any warnning messages 
        # produced by LSODA solver, called by ode() function.
        # Suppression further complicates if making log files (Tee):
        change_std2tee = False  # Normaly no need for this mess.
        if not isinstance(sys.stdout, stdout_type):
            # In this case the real standard output (sys.stdout) is not
            # saved in original location sys.stdout. We have to obtain
            # it inside of Tee object (look module tee_so).
            tee_object = sys.stdout  # obtain Tee object that has sys.stdout
            std_output = tee_object.stdout  # Obtain sys.stdout.
            sys.stdout = std_output  # Change fake stdout to real stdout.
            change_std2tee = True  # Remember to change it back.
        # Next line works only when sys.stdout is real. Thats why above.
        with open(os.devnull, 'w') as f, mt.stdout_redirected(f):
            odeY = ode(model_list, params_matrix, T, X, y0=Y[0],
                        **estimation_settings)  # change to Y[:1]
        if change_std2tee: 
            sys.stdout = tee_object  # Change it back to fake stdout (tee).

        odeY = odeY.T  # solve_ivp() returns in oposite (DxN) shape.
        if not odeY.shape == Y.shape:
            # print("The ODE solver did not found ys at all times -> returning dummy error.")
            return dummy
        try:
            res = np.mean((Y-odeY)**2)
            if np.isnan(res) or np.isinf(res) or not np.isreal(res):
#                print(model.expr, model.params, model.sym_params, model.sym_vars)
                return dummy
            return res
        except Exception as error:
            print("Programmer1: Params at error:", params, f"and {type(error)} with message:", error)
            return dummy

    except Exception as error:
        print("Programmer: Excerpted an error inside ode() of model_ode_error.")
        print("Programmer: Params at error:", params, f"and {type(error)} with message:", error)
        print("Returning dummy error. All is well.")
        return dummy

def DE_fit (model, X, Y, T, p0, **estimation_settings):
    """Calls scipy.optimize.differential_evolution. 
    Exists to make passing arguments to the objective function easier."""
    
    # bounds = [[-3*10**1, 3*10**1] for i in range(len(p0))]
    lower_bound, upper_bound = (estimation_settings["lower_upper_bounds"][i] for i in (0, 1))
    bounds = [[lower_bound, upper_bound] for i in range(len(p0))]

    start = time.perf_counter()
    def diff_evol_timeout(x=0, convergence=0):
        now = time.perf_counter()
        if (now-start) > estimation_settings["timeout"]:
            print("Time out!!!")
            return True
        else:
            return False
    
    return differential_evolution(
        estimation_settings["objective_function"], bounds,
        args=[model, X, Y, T, estimation_settings],
        callback=diff_evol_timeout, maxiter=10**2, popsize=10)

def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    
    return minimize(optimization_wrapper, model.params, args = (model, X, Y))

def find_parameters (model, X, Y, T, **estimation_settings):
    """Calls the appropriate fitting function. 
    
    TODO: 
        add method name input, matching to a dictionary of fitting methods.
    """
#    try:
#        popt, pcov = curve_fit(model.evaluate, X, Y, p0=model.params, check_finite=True)
#    except RuntimeError:
#        popt, pcov = model.params, 0
#    opt_params = popt; othr = pcov

    equation_type = estimation_settings["equation_type"]
    if equation_type == "algebraic":
        estimation_settings["objective_function"] = model_error
    elif equation_type == "differential":
        estimation_settings["objective_function"] = model_ode_error
    else:
        types_string = "\", \"".join(EQUATION_TYPES)
        raise ValueError("Variable equation_type has unsupported value "
                f"\"{equation_type}\", while list of possible values: "
                f"\"{types_string}\".")

    res = DE_fit(model, X, Y, T, p0=model.params, **estimation_settings)


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
        Input:
            estimation_settings: Dictionary with multiple parameters
                that determine estimation process more specifically.
    """
    def __init__(self, X, Y, T, **estimation_settings):
        self.X = X
        self.Y = Y
        self.T = T
        self.estimation_settings = estimation_settings
        
    def fit_one (self, model):
        print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > 5:
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun":model_error_general(
                    [], model, self.X, self.Y, self.T,
                    **self.estimation_settings)})
            else:
                res = find_parameters(model, self.X, self.Y, self.T,
                                     **self.estimation_settings)
                model.set_estimated(res)
        except Exception as error:
            print((f"Excepted an error: Of type {type(error)} and message:"
                    f"{error}!! \nModel:"), model)
            model.set_estimated({}, valid=False)
        # todo: optional kwargs: verbosity>1: print next line:
        print(f"model: {str(model.get_full_expr()):<70}; "
                + f"p: {model.p:<23}; "
                + f"error: {model.get_error()}")

        return model
    
def fit_models (models, X, Y, T=None, pool_map=map, verbosity=0,
                equation_type="algebraic", timeout=np.inf, 
                lower_upper_bounds=(-30, 30), **additional):
    """Performs parameter estimation on given models. Main interface to the module.
    
    Supports parallelization by passing it a pooled map callable.
    
    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted. 
        X (numpy.array): Input data of shape N x M, where N is the number of samples 
            and M is the number of variables.
        Y (numpy.array): Output data of shape N x D, where N is the number of samples
            and D is the number of output variables.
        T (numpy.array): Times array, used for solving differential equations, where
            form required is noted inside the definition of ode() function.
        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, X, Y, pool_map = pool.map)
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
        equation_type: Type of equations, e.g. "algebraic" or "differential", that
            equation discovery algorithm tries to discover.
        timeout: Maximal time consumed for whole minimization optimization process,
            e.g. for differential evolution, that is performed for each model.
        lower_upper_bounds: Pair, i.e. tuple of lower and upper bound used to
            specify the boundaries of optimization, e.g. of differential evolution.
        additional: Other parameters used in lower level parts of discovery,
            e.g. in solving ODEs.
        max_ode_steps: As an example of above, it can be passed through **additional
            argument. Maximal number of steps used in one run of LSODA solver.
        estimation_settings: Dictionary where majority of optional arguments is stored.
    """
    if not isinstance(T, type(None)):
        equation_type = "differential"
    estimation_settings = {
        "verbosity": verbosity, "equation_type": equation_type,
        "timeout": timeout, "lower_upper_bounds": lower_upper_bounds}
    estimation_settings = {**estimation_settings, **additional}
    estimator = ParameterEstimator(X, Y, T, **estimation_settings)
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
    
    models = fit_models(models, X, y, equation_type="algebraic")
    print(models)

