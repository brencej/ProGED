# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import sympy as sp

np.random.seed(0)

class Model:

    """
    Class model represents a single model, defined by its canonical expression string.
    An object of Model acts as a container for various representations of the model,
    including its expression, symbols, parameters, the parse trees that simplify to it,
    and associated information and references.

    Attributes:
        expr        (SymPy expression)       The canonical expression defining the model.
        sym_vars    (list of Sympy symbols)  The symbols appearing in expr that are to be interpreted as variables.
        lhs_vars    (list of strings)        The variables appearing on the left hand side of the equations in the model.
        rhs_vars    (list of strings)        The variables appearing on the right hand side of the equations in the model.
        extra_vars  (list of strings)        The subset of observed variables that are not lhs variables.
        sym_params  (list of strings)        Symbols appearing in expr that are to be interpreted as free constants.
        params      (list of floats)         The values for the parameters, initial or estimated.
        estimated   (dict)                   Results of optimization. Items:
                                                "x" (list of floats) solution of optimization, i.e. optimal parameter values
                                                "fun" (float) value of optimization function, i.e. error of model
                                                "duration" (float) the duration of optimization
        valid       (boolean)                True if parameters successfully estimated. False if estimation has not
                                                been performed yet or if it was unsuccessful.
        trees       (dict)                   Tracks parse trees that simplify to expr. Keys are codes of parse trees,
                                                values are a list with:
                                                * probability of parse tree (float)
                                                * number of occurences during sampling (int)
        code        (string)                 Parse tree code, expressed as string of integers, corresponding to the choice of
                                                production rules when generating the expression. Allows the generator to replicate
                                                the generation. Requires the originating grammar to be useful.
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
    
    def __init__(self, expr, sym_vars, lhs_vars, sym_params=[], params={}, estimated={}, valid=False,
                 trees={}, code="", p=1, grammar=None, **kwargs):
        """Initialize a Model with the initial parse tree and information on the task."""

        expr, sym_vars, lhs_vars, sym_params = self.check_initialization_input(expr, sym_vars, lhs_vars, sym_params)

        # set system variables (rhs vars and extra vars) and target variables (lhs vars)
        self.expr = expr
        self.sym_vars = sym_vars
        self.lhs_vars = lhs_vars
        expr_symbols = [iexpr.free_symbols for iexpr in self.expr]
        self.rhs_vars = [item for item in self.sym_vars if item in list(set.union(*expr_symbols))]
        self.sym_params = sym_params

        # create dictionary of parameters (keys->parameter names, values->values of parameters)
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
        self.estimated = estimated
        self.valid = valid

        # set random initial values (should be limited to ODEs?)
        self.initials = dict(zip(self.lhs_vars, np.zeros(len(self.lhs_vars))))

        # observability info (whether data for all state variables is present)
        self.observed_vars = kwargs.get('observed_vars', [str(i) for i in self.rhs_vars])
        self.unobserved_vars = kwargs.get('unobserved_vars', [])

        self.extra_vars = [str(item) for item in self.observed_vars if str(item) not in self.lhs_vars]

        # grammar info
        self.info = kwargs.get('info', {})
        self.grammar = grammar
        self.p = p                             # TODO: CHECK IF CORRECT (BEFORE IT WAS 0 BUT MODELBOX TEST FAILED)
        self.trees = trees  #trees has form {"code":[p,n]}"
        self.add_tree(code, p)

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

    def set_params(self, params, extra_params=False):
        """ Sets self.params based on new argument "params".

         Arguments:
            params (list of floats)    parameter values that are returned by optimization algorithm, e.g. [1, -2.33, 1.45]
            extra_params (bool)        if True, the function checks the number of model params versus extra params that
                                           could also be in the list (e.g. parameter estimation also estimates unknown
                                           initial values) and only sets a subset of params as model params. Extra
                                           parameters are determined based on number of unobserved variables.

        Returns:
            nothing, it only updates self.params
        """

        if extra_params:
            true_params = params[:-len(self.unobserved_vars)] if self.unobserved_vars else params
        else:
            true_params = params

        if isinstance(params, (list, np.ndarray)):
            self.params = dict(zip(self.params.keys(), true_params))
        else:
            self.params = true_params

    def set_initials(self, params, data_inits):
        """ Sets self.initials based on arguments "params" and "data_inits". First, it checks which left hand side
        variables can get initial values from the data. If the initial value is not in data, then it takes the initial
        value from the argument params.

         Arguments:
            params (list of floats)     parameter values that are returned by optimization algorithm, e.g. [1, -2.33, 1.45]
            data_inits (dict)           initial values represented as a dictonary, where keys are column names of the
                                        dataframe and values are the initial values in each column. e.g. {'y': 1.1}

        Returns:
            nothing, it only updates self.initials
        """

        i = 0
        for ilhs in self.lhs_vars:
            if ilhs in list(data_inits.keys()):
                self.initials[ilhs] = data_inits[ilhs]
            else:
                if self.unobserved_vars:
                    self.initials[ilhs] = params[-len(self.unobserved_vars):][i]
                    i =+ 1
                else:
                    raise ValueError(f'The variable "{ilhs}" is declared as observed but not present in the data.')

    def get_params(self):
        return self.params

    def lambdify(self, params=None, arg="numpy", add_time=False, list=False):
        """Produce a callable function from the symbolic expression and the parameter values. This function is required
        for the evaluate function. It relies on sympy.lambdify. Syntactic errors in variable or parameter names
        will likely produce an error here.
        
        Arguments:
            params (list of floats): parameter values that are returned by optimization algorithm, e.g. [1, -2.33, 1.45]
            arg (string): Passed on to sympy.lambdify. Defines the engine for the mathematical operations,
                          that the symbolic operations are transformed into. Default: numpy. See sympy
                          documentation for details.
            add_time(bool): if True, time ("t") is added as a first argument of newly created callable function.
            list (bool): if True, each expression is return as an element of a list. If False, expressions are joined
                         in one function.

        Returns:
            callable function that takes variable values as inputs and return the model value.
        """

        if not params:
            params = self.params
        fullexprs = self.full_expr(params)

        if add_time:
            # lambdas = [sp.lambdify([sp.symbols("t")] + self.rhs_vars, full_expr, arg) for full_expr in fullexprs]  # currently testing with sym_vars
            # lambdas = [sp.lambdify([sp.symbols("t")] + self.sym_vars, full_expr, arg) for full_expr in fullexprs]
            lambdas = [sp.lambdify(["t"] + self.lhs_vars + self.extra_vars, full_expr, arg) for full_expr in fullexprs]
        else:
            lambdas = [sp.lambdify(self.rhs_vars, full_expr, arg) for full_expr in fullexprs]

        if list:
            return lambdas
        else:
            return lambda x: np.transpose([lam(*x.T) for lam in lambdas])


    def evaluate(self, data_points, params=None):
        """Evaluate the model for given variable and parameter values.
        
        If possible, use this function when you want to do computations with the model.
        It relies on lambdify so it shares the same issues, but includes some safety checks.
        Example of use with stored parameter values: predictions = model.evaluate(X, model.params)
        
        Arguments:
            data_points (numpy array): Input data, shaped N x M, where N is the number of samples and
                                           M the number of variables. e.g. = np.array( [[0.], [5.]])
            params (dict): Parameter values, stored in a dict as in the model.params (e.g. {C1: 2.22}).
            
        Returns:
            Numpy array of shape N x D, where N is the number of samples and D the number of output variables.
        """

        if not params:
            params = self.params

        lamb_expr = self.lambdify(params)

        # tranform data_points to numpy array if it is in dataframe
        if isinstance(data_points, pd.DataFrame):
            data_points = np.array(data_points)

        # TODO: This was changed, the return of the first sentence should be checked!
        if type(lamb_expr(np.array([[1], [1]]))) != type(np.array([[1], [1]])):
            return np.ones(len(data_points))*lamb_expr(1)
        else:
            return lamb_expr(data_points)

    def full_expr(self, params=None):
        """
        Substitutes parameter symbols in the symbolic expression with given parameter values.
        Returns sympy expression with parameter values.
        """

        if not params:
            params = self.params

        fullexprs = []
        for i, ex in enumerate(self.expr):
            fullexprs += [ex.subs(list(zip(params.keys(), params.values())))]

        return fullexprs

    def get_full_expr(self):
        return self.full_expr(self.params)

    def split(self):
        model_splits = []
        for ilhs, iexpr in enumerate(self.expr):
            sym_params_split = [item for item in self.sym_params if item in list(iexpr.free_symbols)]
            model_splits.append(Model(expr=iexpr,
                                      grammar=self.grammar,
                                      sym_vars=self.rhs_vars,
                                      lhs_vars=self.lhs_vars[ilhs],
                                      sym_params=sym_params_split))
        return model_splits

    def nice_print(self, return_string=False, round_params=2):
        """
        Prints the model in a form of equation.

        Arguments:
            return_string (bool): if True, the string_to_print is returned. Default: False.
            round_params (int): the number of decimal places the floats should be rounded up in the printed string. Default=2.

        Returns:
            string_to_print (string):    Optional. String to be printed, in the case of the storage elsewhere.
        """
        def round_constants(expr, n=3):
            for a in sp.preorder_traversal(expr):
                if isinstance(a, sp.Float):
                    expr = expr.subs(a, round(a, n))
            return expr

        string_to_print = ""
        for i, iexpr in enumerate(self.expr):
            if self.valid:
                string_to_print += f"{self.lhs_vars[i]} = {round_constants(self.full_expr(self.params)[i], n=round_params)}\n"
            else:
                string_to_print += f"{self.lhs_vars[i]} = {self.expr[i]}\n"

        print(string_to_print)

        if return_string:
            return string_to_print

    def get_time(self):
        if "duration" in self.estimated:
            return self.estimated["duration"]
        else:
            return 0

    def check_initialization_input(self, expr, sym_vars, lhs_vars, sym_params):

        # transform expression to sympy expression if not already
        if isinstance(expr, str):
            expr = list([sp.sympify(ex) for ex in expr.split(",")])  # strip(" ()")
        elif isinstance(expr, (list, tuple)):
            expr = [sp.sympify(ex) for ex in expr]
        elif isinstance(expr, sp.Expr):
            expr = [expr]
        else:
            raise ValueError("Unknown type passed as expr of Model."
                  "Valid types: string, tuple or list of strings or sympy expression. Example: ['C0*x'].")

        # transform sym_vars to sympy expression if not already
        if isinstance(sym_vars, str):
            sym_vars = list([sp.symbols(ex) for ex in expr.split(",")])  # strip(" ()")
        elif isinstance(sym_vars, (list, tuple)):
            if not isinstance(sym_vars[0], sp.Symbol):
                sym_vars = sp.symbols(sym_vars)
        elif isinstance(sym_vars, sp.Expr):
            sym_vars = [sym_vars]
        else:
            raise ValueError("Unknown type passed as sym_vars of Model."
                  "Valid types: string, tuple or list of strings or sympy expression. Example: ['x', 'y'].")

        # transform lhs_vars to list of strings if not already
        if isinstance(lhs_vars, str):
            lhs_vars = list([ilhs for ilhs in lhs_vars.split(",")])  # strip(" ()")
        elif isinstance(lhs_vars, sp.Expr):
            lhs_vars = list([str(ilhs) for ilhs in lhs_vars])
        elif isinstance(lhs_vars, (list, tuple)):
            lhs_vars = lhs_vars
        else:
            raise ValueError("Unknown type passed as lhs_vars of Model."
                  "Valid types: string, tuple or list of strings or sympy expression. Example: ['x', 'y'].")

        # transform sym_params to tuple of sympy symbols
        if len(sym_params) == 0:
            pass
        elif isinstance(sym_params, (list, tuple)):
            if not isinstance(sym_params[0], sp.Symbol):
                sym_params = sp.symbols(sym_params)
        elif isinstance(sym_params, str):
            sym_params = tuple([sp.symbols(sym_params)])
        elif isinstance(sym_params, sp.Symbol):
            sym_params = tuple([sym_params])
        else:
            raise ValueError("Unknown type passed as sym_params input of Model."
                            "Valid types: sp.symbol, string, tuple or list of strings/sp.symbols. Example: ('C1', 'C2', 'C3').")


        return expr, sym_vars, lhs_vars, sym_params

    def __str__(self):
        if self.valid:
            return str(self.full_expr(self.params))
        else:
            return str(self.expr)
    
    def __repr__(self):
        return self.__str__()
    
    
    
if __name__ == '__main__':

    print("--- model.py examples ---")

    model0 = Model(expr="C0 * sin(x-y) - sin(x)",
                  sym_vars=["x", "y"],
                  lhs_vars=["x"],
                  sym_params=['C0'])
    print(f"--- model.py example 0 finished: {model0.nice_print()} ---")


    model1 = Model(expr=["C0 * sin(x-y) - sin(x)", "C1 * sin(x-y) - sin(y)"],
                  sym_vars=["x", "y"],
                  lhs_vars=["x", "y"],
                  sym_params=['C0', 'C1'])
    print(f"--- model.py example 1 finished: {model1.nice_print()} ---")


    from ProGED import ModelBox
    model2 = ModelBox()
    model2.add_model(["C * sin(x-y) - sin(x)", "C * sin(x-y) - sin(y)"],
                     symbols={"x": ["x", "y"], "const": "C"})
    print(f"--- model.py example 2 finished: {model2[(list(model2.models_dict.keys())[0])].nice_print()} ---")

    model3 = ModelBox()
    model3.add_model("C * sin(x-y) - sin(x)",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=['x'])
    print(f"--- model.py example 3 finished: {model3[(list(model3.models_dict.keys())[0])].nice_print()} ---")

    print("--- model.py test 4 ---")
    from nltk import PCFG
    grammar_str = "S -> 'c' '*' 'x' [1.0]"
    grammar = PCFG.fromstring(grammar_str)
    parse_tree_code = "0"
    expression_str = "c*x"
    probability = 1.0
    symbols_params = ["c"]
    symbols_variables = ["x"]
    
    print("Create the model instance.")
    model = Model(expr = expression_str, 
                  grammar = grammar, 
                  code = parse_tree_code, 
                  p = probability,
                  sym_params = symbols_params,
                  sym_vars = symbols_variables,
                  lhs_vars = symbols_variables)
    print(f"Printing model 4:")
    model.nice_print()
    assert str(model.expr[0]) == expression_str
    
    print("Try to print the model error before it thas been estimated."
          "The model returns the dummy value for an invalid model.")
    print(f"Dummy error: {model.get_error()}.")
    assert model.get_error() == 10**9
    
    print("Perform parameter estimation and add the results to the model.")
    result = {"x":[1.2], "fun":0.001}
    model.set_estimated(result)
    print(f"Printing model 4 with parameter values:")
    model.nice_print()
    assert str(model.full_expr(model.params)) == "[1.2*x]"
    
    print("Evaluate the model at points X:")
    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, model.params)
    print(y)
    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.array([0, 6.0]).reshape(2, 1))**2) < 1e-15
    
    print("--- model.py test 4 finished ---")

