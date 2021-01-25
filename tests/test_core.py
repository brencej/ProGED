# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:37:52 2021

@author: Jure
"""

import numpy as np
from nltk import Nonterminal, PCFG

import sys
sys.path.append("../ProGED/")

from equation_discoverer import EqDisco
from generators.grammar import GeneratorGrammar
from generators.grammar_construction import grammar_from_template
from generate import generate_models
from model import Model
from model_box import ModelBox
from parameter_estimation import fit_models

def test_grammar_general():
    np.random.seed(0)
    
    txtgram = "S -> S '+' F [0.2] | F [0.8] \n"
    txtgram += "F -> 'x' [0.5] | 'y' [0.5]"
    grammar = GeneratorGrammar(txtgram)
    
    sample = grammar.generate_one()
    assert sample[0] == ['y'] and sample[1] == 0.4 and sample[2] == '11'
    
    assert grammar.count_trees(Nonterminal("S"), 5) == 30
    assert grammar.count_coverage(Nonterminal("S"), 2) == 0.8
    
    assert "".join(grammar.code_to_expression('0101')[0]) == "x+y"
    
def test_grammar_templates():
    np.random.seed(0)
   
    templates_to_test = ["polynomial", "trigonometric", "polytrig", "simplerational", "rational", "universal"]
    variables = ["'x'", "'y'", "'z'"]
    p_vars = [0.3, 0.3, 0.4]
    codes = ["1101", "011", "02", "12", "0110001221101", "0202111211"]
    
    grammars = [grammar_from_template(template_name, {"variables": variables, "p_vars": p_vars}) for template_name in templates_to_test]
    for i in range(len(grammars)):
        assert grammars[i].generate_one()[2] == codes[i]
        
def test_generate_models():
    np.random.seed(0)
    generator = grammar_from_template("polynomial", {"variables":["'x'", "'y'"], "p_vars":[0.3,0.7]})
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 3
    samples = ["C0*y", "C0*x*y**2", "C0*x**2 + C1"]
    
    models = generate_models(generator, symbols, strategy_settings = {"N":N})
    for i in range(len(models)):
        assert str(models[i]) == samples[i]
        
def test_model():
    grammar_str = "S -> 'c' '*' 'x' [1.0]"
    grammar = PCFG.fromstring(grammar_str)
    parse_tree_code = "0"
    expression_str = "c*x"
    probability = 1.0
    symbols_params = ["c"]
    symbols_variables = ["x"]
    
    model = Model(expr = expression_str, 
                  grammar = grammar, 
                  code = parse_tree_code, 
                  p = probability,
                  sym_params = symbols_params,
                  sym_vars = symbols_variables)

    assert str(model) == expression_str
    
    assert model.get_error() == 10**8
    
    result = {"x":[1.2], "fun":0.001}
    model.set_estimated(result)
    
    assert str(model.full_expr(*model.params)) == "1.2*x"
    
    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, *model.params)

    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.array([0, 6.0]))**2) < 1e-15
    
def test_model_box():
    grammar_str = "S -> 'c' '*' 'x' [0.5] | 'x' [0.5]"
    grammar = PCFG.fromstring(grammar_str)
    expr1_str = "x"
    expr2_str = "c*x"
    symbols = {"x":['x'], "const":"c", "start":"S"}
    
    models = ModelBox()
    models.add_model(expr1_str, symbols, grammar)
    assert len(models) == 1
    models.add_model(expr2_str, symbols, grammar, p=0.5, code="1")
    assert len(models) == 2
    assert str(models[1]) == str(models["c0*x"])
    assert str(models[1]) == "c0*x"
    assert models[1].p == 0.5
        
def test_parameter_estimation():
    np.random.seed(1)
    def f(x):
        return 2.0 * (x[:,0] + 0.3)
    X = np.linspace(-1, 1, 20).reshape(-1,1)
    Y = f(X).reshape(-1,1)
    data = np.hstack((X, Y))
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x'], "start":"S", "const":"C"}
    N = 2
    
    models = generate_models(grammar, symbols, strategy_settings = {"N":N})
    models = fit_models(models, data, target_variable_index = -1, equation_type="algebraic")
    
    assert np.abs(models[0].get_error() - 0.36) < 1e-6
    assert np.abs(models[1].get_error() - 1.4736842) < 1e-6
    
def test_parameter_estimation_2D():
    np.random.seed(0)
    def f(x):
        return 2.0 * (x[:,0]*x[:,1] + 0.3)
    
    r = np.linspace(-1, 1, 4)
    X = np.array([[[x,y] for x in r] for y in r]).reshape(-1,2)
    Y = f(X).reshape(-1,1)
    data = np.hstack((X, Y))
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 2
    
    models = generate_models(grammar, symbols, strategy_settings = {"N":N})
    models = fit_models(models, data, target_variable_index = -1, equation_type="algebraic")
    
    assert np.abs(models[0].get_error() - 0.36) < 1e-6
    assert np.abs(models[1].get_error() - 1.5945679) < 1e-6

def test_equation_discoverer():
    np.random.seed(0)
    def f(x):
        return 2.0 * (x[:,0] + 0.3)
	
    X = np.linspace(-1, 1, 20).reshape(-1,1)
    Y = f(X).reshape(-1,1)
    data = np.hstack((X,Y))
        
    ED = EqDisco(data = data,
                 task = None,
                 target_variable_index = -1,
                 sample_size = 2,
                 verbosity = 1)
    
    ED.generate_models()
    ED.fit_models()
    
    #print(ED.models[0].get_error())
    assert np.abs(ED.models[0].get_error() - 0.72842105) < 1e-6
    assert np.abs(ED.models[1].get_error() - 0.59163899) < 1e-6
    
# if __name__ == "__main__":
#     test_grammar_general()
#     test_grammar_templates()
#     test_generate_models()
#     test_model()
#     test_model_box()
#     test_parameter_estimation()
#     test_parameter_estimation_2D()    
#     test_equation_discoverer()