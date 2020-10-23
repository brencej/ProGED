# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 11:25:27 2020

@author: Jure
"""

from nltk.grammar import Nonterminal
from nltk import PCFG
import numpy as np
import sympy as sp

class Model:
    """Class that represents a single model, defined by its canonical expression string.
    
    An object of Model acts as a container for various representations of the model,
    including its expression, symbols, parameters, the parse trees that simplify to it,
    and associated information and references. 
    Class methods serve as an interfance to interact with the model.
    The class is intended to be used as part of an equation discovery algorithm.
    
    TODO
    
    list the member variables and methods
    document the methods
    """
    
    def __init__(self, expr = None, grammar=None, params=[], sym_params=[], sym_vars = [], code="", p=0):
        
        self.grammar = grammar
        self.params = params
        if isinstance(expr, type("")):
            self.expr = sp.sympify(expr)
        else:
            self.expr = expr
     
        try:
            self.sym_params = sp.symbols(sym_params)
            if type(self.sym_params) != type((1,2)):
                if isinstance(sym_params, list):
                    self.sym_params = tuple(sym_params)
                elif isinstance(sym_params, (int, float, str)):
                    self.sym_params = (self.sym_params, )
                else:
                    print("Unknown type passed as sym_params input of Model."\
                          "Valid types: tuple or list of strings."\
                          "Example: ('C1', 'C2', 'C3').")
        except ValueError:
            print(expr, params, sym_params, sym_vars)
        self.sym_vars = sp.symbols(sym_vars)
        self.p = 0
        """self.trees has form {"code":[p,n]}"""
        self.trees = {}
        if len(code)>0:
            self.add_tree(code, p)
        self.estimated = {}
        self.valid = False
        
    def add_tree (self, code, p):
        if code in self.trees:
            self.trees[code][1] += 1
        else:
            self.trees[code] = [p,1]
            self.p += p
        
    def set_estimated(self, result, valid=True):
        self.estimated = result
        self.valid = valid
        if valid:
            self.params = result["x"]
        
    def get_error(self, dummy=10**8):
        if self.valid:
            return self.estimated["fun"]
        else:
            return dummy
        
    def set_params(self, params):
        self.params=params
        
    def lambdify (self, arg="numpy"):
        self.lamb_expr = sp.lambdify(self.sym_vars, self.expr.subs(list(zip(self.sym_params, self.params))), arg)
        test = self.lamb_expr(np.array([1,2,3, 4]))
        if type(test) != type(np.array([])):
            self.lamb_expr = lambda inp: [test for i in range(len(inp))]
        return self.lamb_expr

    def evaluate (self, points, *args):
        lamb_expr = sp.lambdify(self.sym_vars, self.full_expr(*args), "numpy")
        
        if type(points[0]) != type(np.array([1])):
            if type(lamb_expr(np.array([1,2,3]))) != type(np.array([1,2,3])):
                return np.ones(len(points))*lamb_expr(1)
            return lamb_expr(points)
        else:
#            if type(lamb_expr(np.array([np.array([1,2,3])]).T)) != type(np.array([1,2,3])):
            if len(str(self.expr)) < 0:
                return np.ones(len(points))*lamb_expr(*np.array([np.array([1,2,3])]).T)
            return lamb_expr(*points.T)
    
    def full_expr (self, *params):
        if type(self.sym_params) != type((1,2)):
            return self.expr.subs([[self.sym_params, params]])
        else:
            return self.expr.subs(list(zip(self.sym_params, params)))
        
    def get_full_expr(self):
        return self.full_expr(*self.params)
    
    def __str__(self):
        return str(self.expr)
    
    def __repr__(self):
        return str(self.expr)
    
    
    
if __name__ == '__main__':
    print("--- model.py test ---")
    grammar_str = "S -> 'c' '*' 'x' [1.0]"
    grammar = PCFG.fromstring(grammar_str)
    parse_tree_code = "0"
    expression_str = "c*x"
    probability = 1.0
    symbols_params = ["c"]
    symbols_variables = ["x"]
    
    print("Create the model instance and print the model.")
    model = Model(expr = expression_str, 
                  grammar = grammar, 
                  code = parse_tree_code, 
                  p = probability,
                  sym_params = symbols_params,
                  sym_vars = symbols_variables)
    print(model)
    assert str(model) == expression_str
    
    print("Try to print the model error before it thas been estimated."\
          "The model returns the dummy value for an invalid model.")
    print(model.get_error())
    assert model.get_error() == 10**8
    
    print("Perform parameter estimation and add the results to the model."\
          "Then, print the model with the parameter values substituted.")
    result = {"x":[1.2], "fun":0.001}
    model.set_estimated(result)
    
    print(model.full_expr(*model.params))
    assert str(model.full_expr(*model.params)) == "1.2*x"
    
    print("Evaluate the model at points X.")
    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, *model.params)
    print(y)
    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.array([0, 6.0]))**2) < 1e-15
    
    