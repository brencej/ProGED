# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sympy as sp

"""Module implementing the SystemModel class that represents a system of equations 
defined by a vector of canonical expression strings.
    
An object of SystemModel acts as a container for various representations of the model,
including its vector of expressions, symbols, parameters, the parse trees that simplify to it,
and associated information and references. 
Class methods serve as an interfance to interact with the model.
The class is intended to be used as part of an equation discovery algorithm."""

class SystemModel:
    def __init__(self, expr, p=0, params=[], sym_params=[], sym_vars=[], info={}, observed=[]):
        """Initialize a SystemModel.
        
        Arguments:
            TODO
            """
        
        if not info:
            self.info = {}
        else:
            self.info = info
        
        if isinstance(expr, str):
            self.expr = [sp.sympify(ex) for ex in expr.strip(" ()").split(",")]
        else:
            self.expr = expr
     
        """sym_params should be tuple of tuples of sympy symbols"""
        self.sym_params = sym_params
        self.params = params
        self.n_params = [len(par) for par in sym_params]
        self.total_eq_params = sum(self.n_params)

        """sym_vars should be tuple of tuples of sympy symbols"""
        self.sym_vars = sp.symbols(sym_vars)

        self.p = 0
        self.info = info

        """TODO: figure out better structure for trees"""
        self.trees = {} #trees has form {"code":[p,n]}"
        if "code" in info:
            code = info["code"]
        else:
            code = ""
        self.add_tree(code, p)

        self.estimated = {}
        self.valid = False

        # list of variables that are (un)observed - used for partially-observed systems
        self.observed = observed

        # observability mask, to know which initial values are observable and which not
        self.obs_mask = np.full(len(self.sym_vars), False, dtype=bool)

        # extra parameters, i.e. initial values
        self.initials = np.random.random(len(self.expr))*5

        # number of successful persistent homology comparisons vs. pure rmse
        self.all_iters = 0
        self.ph_used = 0


    def set_estimated(self, result, valid=True):
        """Store results of parameter estimation and set validity of model according to input.
        
        Arguments:
            result (dict): Results of parameter estimation. 
                Designed for use with methods, implemented in scipy.optimize, but works with any method.
                Required items:
                    "x": solution of optimization, i.e. optimal parameter values (list of floats)
                    "fun": value of optimization function, i.e. error of model (float).
            valid: True if the parameter estimation succeeded.
                Set as False if the optimization was unsuccessfull or the model was found to not fit 
                the requirements. For example, we might want to limit ED to models with 5 or fewer parameters
                due to computational time concerns. In this case the parameter estimator would refuse
                to fit the parameters and set valid = False. 
                Invalid models are typically excluded from post-analysis."""
        
        self.estimated = result
        self.valid = valid
        if valid:
            #self.params = np.split(result["x"], np.cumsum(self.n_params))
            self.set_params(result["x"], split=True)

    def get_error(self, dummy=10**8):
        """Return model error if the model is valid, or dummy if the model is not valid.
        
        Arguments:
            dummy: Value to be returned if the parameter have not been estimated successfully.
            
        Returns:
            error of the model, as reported by set_estimated, or the dummy value.
        """
        if self.valid:
            return self.estimated["fun"]
        else:
            return dummy

    def get_params(self, flatten=False):
        if flatten:
            return [par for eq_params in self.params for par in eq_params]
        else:
            return self.params

    def set_params(self, params, split=True):
        if split:
            expr_params = params[:self.total_eq_params]
            # self.initials = params[self.total_eq_params:] # removed
            self.params = np.split(expr_params, np.cumsum(self.n_params))[:-1]
            # self.params = list(filter(None, self.params))
        else:
            self.params=params

    def add_tree (self, code, p):
        """Add a new parse tree to the model.
        
        Arguments:
            code (str): The parse tree code, expressed as a string of integers.
            p (float): Probability of parse tree.
        """
        if code in self.trees:
            self.trees[code][1] += 1
        else:
            self.trees[code] = [p,1]
            self.p += p

    def full_expr (self, params=None):
        """Substitutes parameter symbols in the symbolic expression with given parameter values.
        
        Arguments:
            params (list of floats): Parameter values.
            
        Returns:
            sympy expression."""
        if not params:
            params = self.params

        fullexprs = []
        for i, ex in enumerate(self.expr):
            if type(self.sym_params[i]) != type((1,2)):
                fullexprs += [ex.subs([[self.sym_params[i], params[i]]])]
            else:
                fullexprs += [ex.subs(list(zip(self.sym_params[i], params[i])))]
                
        return fullexprs

    def get_jacobian(self):
        return sp.Matrix(self.full_expr()).jacobian(self.sym_vars)
    
    def lambdify_jacobian(self):
        J = self.get_jacobian()
        Jf = sp.lambdify(self.sym_vars, J)
        #return lambda x: Jf(*x.T)
        return Jf

    def lambdify (self, params=None, list=False, matrix_input = True, arg="numpy"):
        """Produce a callable function from the symbolic expression and the parameter values.
        
        This function is required for the evaluate function. It relies on sympy.lambdify, which in turn 
            relies on eval. This makes the function somewhat problematic and can sometimes produce unexpected
            results. Syntactic errors in variable or parameter names will likely produce an error here.
        
        Arguments:
            arg (string): Passed on to sympy.lambdify. Defines the engine for the mathematical operations,
                that the symbolic operations are transformed into. Default: numpy.
                See sympy documentation for details.
                
        Returns:
            callable function that takes variable values as inputs and return the model value.
        """
        if not params:
            params = self.params
        fullexprs = self.full_expr(params)
        if matrix_input:
            lambdas = [sp.lambdify(self.sym_vars, full_expr, arg) for full_expr in fullexprs]
            if list:
                return lambdas
            else:
                return lambda x: np.transpose([lam(*x.T) for lam in lambdas])
        else:
            system = sp.Matrix(fullexprs)
            return sp.lambdify((sp.symbols("t"), self.sym_vars), system)

    def get_time(self):
        if "time" in self.estimated:
            return self.estimated["time"]
        else:
            return 0

    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self.expr)
    