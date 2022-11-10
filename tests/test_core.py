# -*- coding: utf-8 -*-

import numpy as np
from nltk import Nonterminal, PCFG

from ProGED.equation_discoverer import EqDisco
from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.generate import generate_models
from ProGED.model import Model
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from utils.generate_data_ODE_systems import generate_ODE_data

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
    expr1_str = "x"
    expr2_str = "c*x"
    symbols = {"x":['x'], "const":"c", "start":"S"}
    
    models = ModelBox()
    models.add_model(expr1_str, symbols)
    assert len(models) == 1
    models.add_model(expr2_str, symbols, p=0.5, info={"code": "1"})
    assert len(models) == 2
    assert str(models[1]) == str(models["c0*x"])
    assert str(models[1]) == "c0*x"
    assert models[1].p == 0.5
        
def test_parameter_estimation():
    np.random.seed(1)
    def f(x):
        return 2.0 * (x[:, 0] + 0.3)
    X = np.linspace(-1, 1, 20).reshape(-1, 1)
    Y = f(X).reshape(-1, 1)
    data = np.hstack((X, Y))
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
                              
    symbols = {"x": ['x'], "start": "S", "const": "C"}
    models = generate_models(grammar, symbols, strategy_settings={"N": 2})

    estimation_settings = {"target_variable_index": -1,
                           "time_index": None,
                           "verbosity": 0}
    models = fit_models(models, data, task_type="algebraic", estimation_settings=estimation_settings)
    
    assert np.abs(models[0].get_error() - 0.36) < 1e-6
    assert np.abs(models[1].get_error() - 1.4736842) < 1e-6

def test_parameter_estimation_2D():
    np.random.seed(0)
    def f(x):
        return 2.0 * (x[:, 0]*x[:, 1] + 0.3)
    
    r = np.linspace(-1, 1, 4)
    X = np.array([[[x, y] for x in r] for y in r]).reshape(-1, 2)
    Y = f(X).reshape(-1, 1)
    data = np.hstack((X, Y))
    
    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x": ['x', 'y'], "start": "S", "const": "C"}
    models = generate_models(grammar, symbols, strategy_settings={"N": 2})

    estimation_settings = {"target_variable_index": -1,
                           "time_index": None,
                           "verbosity": 0}

    models = fit_models(models, data, task_type="algebraic", estimation_settings=estimation_settings)
    
    assert np.abs(models[0].get_error() - 0.36) < 1e-6
    assert np.abs(models[1].get_error() - 1.5945679) < 1e-6


def test_parameter_estimation_ODE():
    B = -2.56; a = 0.4; ts = np.linspace(0.45, 0.87, 5)
    ys = (ts+B)*np.exp(a*ts); xs = np.exp(a*ts)
    data = np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1), ys.reshape(-1, 1)))

    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                                T -> V [0.4] | 'C' "*" V [0.6]
                                V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x": ["x", "y"], "start": "S", "const": "C"}

    np.random.seed(0)
    models = generate_models(grammar, symbols, strategy_settings={"N": 3})

    estimation_settings = {"target_variable_index": 1,
                           "time_index": 0,
                           "objective_settings": {"use_jacobian": False},
                           "verbosity": 0}

    models = fit_models(models, data, task_type="differential", estimation_settings=estimation_settings)

    def assert_line(models, i, expr, error, tol=1e-9, n=100):
        #assert str(models[i].get_full_expr())[:n] == expr[:n]
        assert abs(models[i].get_error() - error) < tol
    assert_line(models, 0, "4.65716206980249*y", 4.60039764161529)
    assert_line(models, 1, "-10.0*x", 8.256188274283515)
    assert_line(models, 2, "y", 10.044322790817118, n=8)

    """    
        print("\n", models, "\n\nFinal score:")
        for m in models:
            print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")
    """

def test_parameter_estimation_ODE_system():
    generation_settings = {"simulation_time": 0.25}
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8], **generation_settings)

    system = ModelBox(observed=["x", "y"])
    system.add_system(["C*y", "C*y - C*x*x*y - C*x"], symbols={"x": ["x", "y"], "const": "C"})
    estimation_settings = {"target_variable_index": None,
                           "time_index": 0,
                           "max_iter": 1,
                           "pop_size": 1,
                           "objective_settings": {"use_jacobian": False},
                           "verbosity": 0}
    np.random.seed(0)
    system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
    assert system_out[0].get_error() < 1e-6
    # true params: [[1.], [-0.5., -1., 0.5]]


def test_parameter_estimation_ODE_system_partial_observability():
    generation_settings = {"simulation_time": 0.25}
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8], **generation_settings)
    data = data[:, (0, 1)]  # y, 2nd column, is not observed

    system = ModelBox(observed=["x"])
    system.add_system(["C*y", "C*y - C*x*x*y - C*x"], symbols={"x": ["x", "y"], "const": "C"})
    estimation_settings = {"target_variable_index": None,
                           "time_index": 0,
                           "max_iter": 1,
                           "pop_size": 1,
                           "objective_settings": {"use_jacobian": False},
                           "verbosity": 0}

    system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
    assert system_out[0].get_error() < 1e-5
    # true params: [[1.], [-0.5., -1., 0.5]]

def test_equation_discoverer():
    np.random.seed(0)
    def f(x):
        return 2.0 * (x[:, 0] + 0.3)
	
    X = np.linspace(-1, 1, 20).reshape(-1,1)
    Y = f(X).reshape(-1, 1)
    data = np.hstack((X, Y))
        
    ED = EqDisco(data=data,
                 task=None,
                 target_variable_index = -1,
                 sample_size = 5,
                 verbosity = 0)
    
    ED.generate_models()
    ED.fit_models()
    assert np.abs(ED.models[0].get_error() - 0.72842105) < 1e-6
    assert np.abs(ED.models[1].get_error() - 0.59163899) < 1e-6
    
def test_equation_discoverer_ODE():
    B = -2.56; a = 0.4; ts = np.linspace(0.45, 0.87, 5)
    ys = (ts+B)*np.exp(a*ts); xs = np.exp(a*ts)
    data = np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1), ys.reshape(-1, 1)))

    np.random.seed(20)
    ED = EqDisco(data = data,
                 task = None,
                 task_type = "differential",
                 time_index = 0,
                 target_variable_index = -1,
                 variable_names=["t", "x", "y"],
                 sample_size = 2,
                 verbosity = 1)
    ED.generate_models()
    ED.fit_models()

    def assert_line(models, i, expr, error, tol=1e-9, n=100):
        #assert str(models[i].get_full_expr())[:n] == expr[:n]
        assert abs(models[i].get_error() - error) < tol
    assert_line(ED.models, 0, "y", 12.70440146224583)
    assert_line(ED.models, 1, "0.400266188520229*x + y", 4.773528915588122, n=6)
    return


if __name__ == "__main__":

    # test_grammar_general()
    # test_grammar_templates()
    # test_generate_models()
    # test_model()
    # test_model_box()
    # test_parameter_estimation()
    # test_parameter_estimation_2D()
    # test_equation_discoverer()
    # test_parameter_estimation_ODE()
    # test_equation_discoverer_ODE()
    # test_parameter_estimation_ODE_system()
    test_parameter_estimation_ODE_system_partial_observability()
