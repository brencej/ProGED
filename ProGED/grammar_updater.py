# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from collections import defaultdict

import ProGED as pg

from ProGED.configs import settings
#settings["task_type"] = "algebraic"
#settings["lhs_vars"]= ["y"]


def get_prods_probs_from_grammar(grammar):
    """Returns a list of productions and a list of probabilities from a NLTK or ProGED grammar."""
    if isinstance(grammar, pg.GeneratorGrammar):
        grammar = grammar.grammar

    prods, probs = [], []
    lhs_index = []
    for prod in grammar.productions():
        lhs = str(prod).split(" -> ")[0].strip()
        rhs = str(prod).split(" -> ")[1].strip().split("[")[0].strip()
        prob = float(str(prod).split(" -> ")[1].strip().split("[")[1].strip().split("]")[0].strip())
        if lhs not in lhs_index:
            lhs_index += [lhs]
            prods += [[]]
            probs += [[]]
        prods[lhs_index.index(lhs)] += [f"{lhs} -> {rhs}"]
        probs[lhs_index.index(lhs)] += [prob]
    return prods, probs

def get_grammar_from_prods_probs(productions, probabilities):
    """Returns a string representation of a grammar from a list of productions and probabilities."""
    grammar = ""
    for i, prods in enumerate(productions):
        p_sum = sum(probabilities[i])
        for j, prod in enumerate(prods):
            grammar += f"{prod} [{probabilities[i][j]/p_sum:.15f}]\n"
    return grammar

def count_productions(tree, prod_dict):
    """Counts the number of times each production is used in a parse tree."""
    prod = str(tree[0]).split("[")[0].strip()
    prod_dict[prod] += 1

    if len(tree) == 1:
        return prod_dict
    
    for child_prod in tree[1:]:
        return count_productions(child_prod, prod_dict)

class GrammarUpdater:
    def __init__(self, 
                 grammar, 
                 variables, 
                 thr=1e-9, 
                 eps=1e-1, 
                 sample_size=10, 
                 prob_minimum=0.05, 
                 data=None, 
                 estimator=None, 
                 verbosity=1):
        """Initializes the GrammarUpdater object.
        Inputs:
            - grammar: a NLTK or ProGED grammar object
            - variables: list of variable names
            - thr: threshold for early stopping, set to negative to disable early stopping
            - eps: relative error threshold for the selection criterion
            - sample_size: number of models to generate in each iteration
            - prob_minimum: minimum probability for a production, used to avoid zero probabilities
            - data: data to fit the models
            - estimator: a ProGED.parameter_estimation.Estimator object; to change estimation provide custom Estimator or override fit_model
            - verbosity: level of verbosity in print statements

        Attributes:
            - vji: list of variable names
            - productions: list of productions
            - p_init: initial probability vector
            - thr: threshold for early stopping
            - eps: relative error threshold for the selection criterion
            - prob_minimum: minimum probability for a production
            - sample_size: number of models to generate in each iteration
            - verbosity: level of verbosity in print statements
            - evaled_models: cache of already fitted models
            - evaled_errors: list of errors of the models that have been fitted
            - evaled_ps: list of probability vectors that have been evaluated
            - best_err: best error found so far
            - best_model: best model found so far
            - data: data to fit the models
            - estimator: a ProGED.parameter_estimation.Estimator object
        """
        self.vji = variables
        self.productions, self.p_init = get_prods_probs_from_grammar(grammar)
        self.thr = thr
        self.eps = eps
        self.prob_minimum = prob_minimum
        self.sample_size = sample_size
        self.verbosity = verbosity

        self.evaled_models = {}
        self.evaled_errors = []
        self.evaled_ps = []
        self.best_err = np.inf
        self.best_model = None

        if data is None and estimator is None:
            raise ValueError("Either data or estimator object must be provided.")
        if data is not None:
            self.data = data
        if estimator is None:
            self.estimator = pg.parameter_estimation.Estimator(data=self.data, settings=settings)
        else:
            self.estimator = estimator

    def fit_model(self, model):
        """Fits a model and returns the error, while maintaining a cache of already fitted models."""
        if str(model) in self.evaled_models:
            return self.evaled_models[str(model)]
        else:
            fitted = self.estimator.fit_one(model)
            err = fitted.get_error()
            self.evaled_models[str(model)] = err
            return err
        
    def fit_model_simple(self, model):
        """Fits a model and returns the error."""
        return self.estimator.fit_one(model).get_error()
        
    def select_by_rel_eps(self, model, **kwargs):
        """Returns True if the model's error is within the relative error threshold, False otherwise."""
        if "eps" in kwargs:
            eps = kwargs["eps"]
        else:
            eps = self.eps

        if model.get_error() - self.best_err < eps*self.best_err:
            return True
        return False
    
    def evaluate_probs(self, p, selection_criterion = None):
        """Evaluates the probability vector p and returns the best model if a solution has been found, otherwise returns 
        the production counts of the models that fit the selection criterion.
        Inputs:
            - p: probability vector to evaluate
            - selection_criterion: function to select the best models in each iteration; default: select_by_rel_eps (relative error threshold)

        Returns:
            - success: True if a solution has been found, False otherwise
            - prod_counts: production counts of the models that fit the selection criterion
            - errs: errors of the models that fit the selection criterion
            - models: models that fit the selection criterion"""

        if not selection_criterion:
            select_f = self.select_by_rel_eps
        else:
            select_f = selection_criterion

        self.evaled_ps += [p]
        grammar = get_grammar_from_prods_probs(self.productions, p)
        try:
            grammar = pg.GeneratorGrammar(grammar, depth_limit=50, repeat_limit=1)
        except ValueError as e:
            if self.verbosity > 0:
                print("Encountered error with probability vector:", p)
            raise ValueError(e)

        models = pg.generate.generate_models(grammar, {"x":[f"'{v}'" for v in self.vji], "const":"C"}, strategy_settings = {"N":self.sample_size}, lhs_vars=self.estimator.settings["lhs_vars"])
        if len(models) == 0:
            return np.inf

        self.evaled_errors += [self.fit_model(m) for m in models.values()]
        
        for model in models:
            if model.get_error() < self.thr:
                self.best_model = model
                self.best_err = model.get_error()
                return True, None, None, None
            
            if model.get_error() < self.best_err:
                self.best_model = model
                self.best_err = model.get_error()

        prods_counts = []
        selected_models = []
        errs = []

        for model in models:
            if select_f(model):
                prods = defaultdict(int)
                for prob, tree in list(model.info["trees"].values()): 
                    prods = count_productions(tree[0], prods)
                prods_counts += [prods]
                selected_models += [model]
                errs += [model.get_error()]
        
        return False, prods_counts, errs, models
    
    def update_p(self, p, prods_counts, errs, models, update_fun=None, **kwargs):
        """Updates the probabilities based on the number of times each production is used 
            in selected expressions and the prior probabilities."""
        if update_fun is None:
            update_fun = self.m_estimate

        new_p = []
        for i,prods in enumerate(self.productions):
            set_counts = []
            for j,prod in enumerate(prods):
                sc = 0
                for prod_counts in prods_counts:
                    if prod in prod_counts:
                        sc += prod_counts[prod]
                set_counts += [sc]
            new_p += [[max([self.prob_minimum, pi]) for pi in update_fun(set_counts, p[i], **kwargs)]]

        return new_p
    
    def m_estimate(self, prod_counts, p_prior, **kwargs):
        """Returns the m-estimate of the posterior production probability."""
        total_count = sum(prod_counts)
        return [(prod_counts[i] + kwargs["m"]*p_prior[i])/(total_count + kwargs["m"]) for i in range(len(prod_counts))]

    def optimize(self, p_init=None, max_iter=10, update_p = None, **kwargs):
        """Optimizes the grammar probabilities using the specified update function.
        Inputs:
            - p_init: initial probability vector
            - max_iter: maximum number of iterations
            - update_p: function to update the probabilities; default: m_estimate
            - kwargs: additional arguments for the update function (e.g. m for m_estimate)

        Returns:
            - best_model: the best model found during the optimization
            - p: the optimized probability vector"""
        
        if p_init is None:
            p = self.p_init
        else:
            p = p_init

        if update_p is None:
            update_p = self.m_estimate
        else:
            update_p = update_p

        for i in range(max_iter):
            if self.verbosity > 0:
                print(f"Iteration {i}")
            
            success, productions, errors, models = self.evaluate_probs(p)
            if success:
                if self.verbosity > 0:
                    print("Solution found")
                return self.best_model, p #self.generate_grammar(p, vji=self.vji)
            
            self.evaled_ps += [p]
            
            if len(productions) > 0:
                p = self.update_p(p, productions, errors, models, **kwargs)
            else:
                if self.verbosity > 1:
                    print("No models found")
            
            if self.verbosity > 0:
                print(f"Best error: {self.best_err}", self.best_model, "p=", p)
        
        if self.verbosity > 0:
            print("Max iterations reached")
        success, productions, errors, models = self.evaluate_probs(p)
        self.evaled_ps += [p]
        return self.best_model, p
    
    def get_grammar(self, probabilities):
        """Returns a string representation of the grammar from a list of probabilities."""
        return get_grammar_from_prods_probs(self.productions, probabilities)
    
if __name__ == "__main__":
    # generate data
    def f (x):
        return x[:,0] - 3*x[:,1] - x[:,2] + x[:,4]
    N = 100
    x = np.random.uniform(-10,10,N*5).reshape((N,5))
    y = f(x)
    data = {f"x{i+1}": x[:,i] for i in range(5)}
    data.update({"y":y})
    data =  pd.DataFrame(data)

    # initialize grammar
    m = data.shape[1]-1
    vji = [f"x{i+1}" for i in range(m)]

    grammar = "E -> E '+' F [0.2] | E '-' F [0.2] | F [0.6]\n"
    grammar += "F -> F '*' T [0.2] | F '/' T [0.2] | T [0.6]\n"
    grammar += "T -> '(' E ')' [0.2] | V [0.6] | 'sin' '(' E ')' [0.2]\n"
    grammar += "\n".join([f"V -> '{v}' [{1/len(vji)}]" for v in vji])
    grammar = pg.GeneratorGrammar(grammar)

    # initialize grammar updater
    upt = GrammarUpdater(grammar, vji, thr=1e-9, eps=1e-1, sample_size=10, prob_minimum=0.05, data=data, verbosity=2)

    print("test of m_estimate")
    print(upt.m_estimate([1,2,3,4,5], [0.1,0.1,0.1,0.1,0.1], m=1))
    print("---"*20)

    print("test of evaluate_probs")
    success, prods_counts, errs, models = upt.evaluate_probs(upt.p_init)
    print(success, prods_counts, errs, models)
    print("---"*20)

    print("test of update_p")
    print(upt.update_p(upt.p_init, prods_counts, errs, models, m=1))
    print("---"*20)

    print("test of optimize")
    print(upt.optimize(max_iter=2, m=5))

