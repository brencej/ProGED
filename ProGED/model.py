# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sympy as sp

class Model:

    """
    Class model represents a single model, defined by its canonical expression string.
    An object of Model acts as a container for various representations of the model,
    including its expression, symbols, parameters, the parse trees that simplify to it,
    and associated information and references.

    Attributes:
        expr        (SymPy expression)       The canonical expression defining the model.
        sym_vars    (list of Sympy symbols)  The symbols appearing in expr that are to be interpreted as variables.
        sym_params  (list of strings)        Symbols appearing in expr that are to be interpreted as free constants.
        params      (list of floats)         The values for the parameters, initial or estimated.
        estimated   (dict)                   Results of optimization. Required items:
                                              "x" (list of floats) solution of optimization, i.e. optimal parameter values
                                              "fun" (float) value of optimization function, i.e. error of model
        valid       (boolean)                True if parameters successfully estimated. False if estimation has not
                                              been performed yet or if it was unsuccessful.
        trees       (dict)                   Tracks parse trees that simplify to expr. Keys are codes of parse trees,
                                              values are a list with:
                                              * probability of parse tree (float)
                                              * number of occurences during sampling (int)
        p           (float)                  Total probability of model. Computed as sum of probabilities of parse trees.
        grammar     (GeneratorGrammar)       Grammar the produced the model. In the future will likely be generalized
                                              to BaseExpressionGenerator and tracked for each parse tree.

    Methods:
        add_tree:       Add a new parse tree to the parse tree dict and update the probabilities.
        set_estimated:  Save results of parameter estimation and set model validity according to input.
        get_error:      Return the model error if model valid or a dummy value if model not valid.
        lambdify:       Produce callable function from symbolic expression and parameter values.
        evaluate:       Compute the value of the expression for given variable values and parameter values.
        full_expr:      Produce symbolic expression with parameters substituted by their values.
    """
    
    def __init__(self, expr, sym_vars, lhs_vars, sym_params=[], params={}, code="", p=1, grammar=None, **kwargs):
        """Initialize a Model with the initial parse tree and information on the task."""

        # transform string to sympy expression if not alraedy
        if isinstance(expr, str):
            self.expr = list([sp.sympify(ex) for ex in expr.split(",")]) # strip(" ()")
        elif isinstance(expr, list):
            self.expr = [sp.sympify(ex) for ex in expr]
        else:
            self.expr = expr

        # set system and target variables
        self.lhs_vars = lhs_vars
        self.sym_vars = sp.symbols(sym_vars)

        # transform sym_params to sympy symbols
        try:
            sym_params = list(sym_params) if isinstance(sym_params, str) else sym_params
            self.sym_params = tuple(sp.symbols(sym_params))
        except ValueError:
            print("Unknown type passed as sym_params input of Model."
                  "Valid types: string, tuple or list of strings. Example: ('C1', 'C2', 'C3').")

        # create dictionary of parameters (keys->names, values->values of parameters)
        if not params:
            param_values = np.random.uniform(low=-5, high=5, size=len(self.sym_params))
            self.params = dict(zip(self.sym_params, param_values))
        else:
            if len(params) != len(self.sym_params):
                raise ValueError('The number of parameters symbols (sym_params) must match the number of parameter values (params).')
            if isinstance(params, int):
                self.params = dict(zip(self.sym_params, [params]))
            if isinstance(params, (list, tuple)):
                self.params = dict(zip(self.sym_params, params))
            elif isinstance(params, dict):
                self.params = params
            else:
                print("Unknown type passed as params input of Model."
                      "Valid types: int, dict or list/tuple of floats. Examples: {'C1': 5} or [5, 2.5] or 1.")

        # add info whether the parameters are valid or not.
        self.estimated = {}
        self.valid = kwargs.get('valid', False)

        # set random initial values (should be limited to ODEs?)
        self.initials = dict(zip(self.lhs_vars, np.random.uniform(low=-5, high=5, size=len(self.lhs_vars))))

        # observability info (whether data for all state variables is present)
        self.observed_vars = kwargs.get('observed_vars', [s.strip("'") for s in sym_vars])
        self.unobserved_vars = kwargs.get('unobserved_vars', [])

        # grammar info
        self.info = kwargs.get('info', {})
        self.grammar = grammar
        self.p = 0
        self.trees = {} #trees has form {"code":[p,n]}"
        self.add_tree(code, p)

        # number of successful persistent homology comparisons vs. pure rmse (should be limited to ODEs?)
        self.ph_all_iters = 0
        self.ph_used = 0
        self.ph_zerovszero = 0

    def add_tree(self, code, p):
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
        
    def set_estimated(self, result, valid=True):
        """Store results of parameter estimation and set validity of model according to input.
        
        Arguments:
            result (dict): Results of parameter estimation. Includes:
                                "x": solution of optimization, i.e. optimal parameter values (list of floats)
                                "fun": value of optimization function, i.e. error of model (float).
                                "duration": duration of parameter estimation
                                "complete_results" (optional): complete output of optimization algorithm
            valid (boolean): True if the parameter estimation succeeded.
                        Set as False if the optimization was unsuccessfull or the model was found to not fit
                        the requirements. For example, we might want to limit ED to models with 5 or fewer
                        parameters due to computational time concerns. In this case the parameter estimator
                        would refuse to fit the parameters and set valid = False.
                        Invalid models are typically excluded from post-analysis.
        """
        
        self.estimated = result
        self.valid = valid
        if valid:
            self.params = dict(zip(self.params.keys(), result["x"]))

    def get_error(self, dummy=10**9):
        """Return model error if the model is valid, or dummy if the model is not valid.
        
        Arguments:
            dummy (float)    Value to be returned if the parameter have not been estimated successfully.
            
        Returns:
            error (float)    Error of the model, as reported by set_estimated, or the dummy value.
        """
        if self.valid:
            return self.estimated["fun"]
        else:
            return dummy

    def set_params(self, params, with_initials=False):

        if with_initials:
            true_params = params[:-len(self.unobserved_vars)] if self.unobserved_vars else params

            if isinstance(params, (list, np.ndarray)):
                self.params = dict(zip(self.params.keys(), true_params))
            else:
                self.params=true_params
        else:
            if isinstance(params, (list, np.ndarray)):
                self.params = dict(zip(self.params.keys(), params))
            else:
                self.params=params

    def set_initials(self, params, data_inits):

        i = 0
        for ilhs in self.lhs_vars:
            if ilhs in list(data_inits.keys()):
                self.initials[ilhs] = data_inits[ilhs]
            else:
                if self.unobserved_vars:
                    self.initials[ilhs] =  params[-len(self.unobserved_vars):][i]
                    i =+ 1
                else:
                    raise ValueError(f'The variable "{ilhs}" is declared as observed but not present in the data.')

    def get_params(self):
        return self.params

    def lambdify(self, params=None, arg="numpy", matrix_input=True, list=False, add_time=False):

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
        # if not params:
        #     params = self.params
        # return sp.lambdify(self.sym_vars, self.full_expr(*params), "numpy")

        if not params:
            params = self.params

        fullexprs = self.full_expr(params)
        if matrix_input:
            if add_time:
                lambdas = [sp.lambdify([sp.symbols("t")] + self.sym_vars, full_expr, arg) for full_expr in fullexprs]
            else:
                lambdas = [sp.lambdify(self.sym_vars, full_expr, arg) for full_expr in fullexprs]

            if list:
                return lambdas
            else:
                return lambda x: np.transpose([lam(*x.T) for lam in lambdas])
        else:
            system = sp.Matrix(fullexprs)
            if add_time:
                return sp.lambdify([sp.symbols("t")] + self.sym_vars, system)
            else:
                return sp.lambdify(self.sym_vars, system)


    def evaluate(self, points, params=None):
        """Evaluate the model for given variable and parameter values.
        
        If possible, use this function when you want to do computations with the model.
        It relies on lambdify so it shares the same issues, but includes some safety checks.
        Example of use with stored parameter values:
            predictions = model.evaluate(X, model.params)
        
        Arguments:
            points (numpy array): Input data, shaped N x M, where N is the number of samples and
                                  M the number of variables.
            params (dict): Parameter values.
            
        Returns:
            Numpy array of shape N x D, where N is the number of samples and D the number of output variables.
        """

        if not params:
            params = self.params

        lamb_expr = self.lambdify(params)

        if isinstance(points, pd.DataFrame):
            points = np.array(points)

        if type(points[0]) != type(np.array([1])):
            if type(lamb_expr(np.array([1,2,3]))) != type(np.array([1,2,3])):
                return np.ones(len(points))*lamb_expr(1)
            return lamb_expr(points)
        else:
            return lamb_expr(points)

    def full_expr(self, params=None):
        """Substitutes parameter symbols in the symbolic expression with given parameter values."""

        if not params:
            params = self.params

        fullexprs = []
        for i, ex in enumerate(self.expr):
            fullexprs += [ex.subs(list(zip(params.keys(), params.values())))]

        return fullexprs

    def get_full_expr(self):
        return self.full_expr(self.params)

    def get_time(self):
        if "duration" in self.estimated:
            return self.estimated["duration"]
        else:
            return 0
    
    def __str__(self):
        if self.valid:
            return str(self.full_expr(self.params))
        else:
            return str(self.expr)
    
    def __repr__(self):
        return self.__str__()
    
    
    
if __name__ == '__main__':

    print("--- model.py examples are included in the folder 'tests' ---")

    model0 = Model(expr="C0 * sin(x-y) - sin(x)",
                  sym_vars=["x", "y"],
                  lhs_vars=["x"],
                  sym_params=['C0'])
    print("--- model.py test 0 finished ---")

    model1 = Model(expr=["C0 * sin(x-y) - sin(x)", "C1 * sin(x-y) - sin(y)"],
                  sym_vars=["x", "y"],
                  lhs_vars=["x", "y"],
                  sym_params=['C0', 'C1'])
    print("--- model.py test 1 finished ---")

    from ProGED import ModelBox
    model2 = ModelBox()
    model2.add_model(["C * sin(x-y) - sin(x)", "C * sin(x-y) - sin(y)"],
                     symbols={"x": ["x", "y"], "const": "C"})
    print("--- model.py test 2 finished ---")

    model3 = ModelBox()
    model3.add_model("C * sin(x-y) - sin(x)",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=['x'])
    print("--- model.py test 3 finished ---")

    print("--- model.py test 4 ---")
    from nltk import PCFG
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
                  sym_vars = symbols_variables,
                  lhs_vars = symbols_variables)
    print(model)
    assert str(model.expr[0]) == expression_str
    
    print("Try to print the model error before it thas been estimated."\
          "The model returns the dummy value for an invalid model.")
    print(model.get_error())
    assert model.get_error() == 10**9
    
    print("Perform parameter estimation and add the results to the model."\
          "Then, print the model with the parameter values substituted.")
    result = {"x":[1.2], "fun":0.001}
    model.set_estimated(result)
    
    print(model.full_expr(model.params))
    assert str(model.full_expr(model.params)) == "[1.2*x]"
    
    print("Evaluate the model at points X.")
    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, model.params)
    print(y)
    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.array([0, 6.0]).reshape(2, 1))**2) < 1e-15
    
    print("--- model.py test 3 finished ---")

