# -*- coding: utf-8 -*-
import pandas as pd
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
from ProGED.configs import settings as settings_original
from copy import deepcopy

np.random.seed(0)


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
    generator = grammar_from_template("polynomial", {"variables": ["'x'", "'y'"], "p_vars": [0.3, 0.7]})
    symbols = {"x": ['x', 'y'], "start": "S", "const": "C"}
    N = 3
    samples = ["C0*y", "C0*x*y**2", "C0*x**2 + C1"]

    models = generate_models(generator, symbols, strategy_settings={"N": N})
    for i in range(len(models)):
        assert str(models[i])[1:-1] == samples[i]


def test_model():
    grammar_str = "S -> 'c' '*' 'x' [1.0]"
    grammar = PCFG.fromstring(grammar_str)
    parse_tree_code = "0"
    expression_str = "c*x"
    probability = 1.0
    symbols_params = ["c"]
    symbols_variables = ["x"]
    lhs_variables = ["y"]

    model = Model(expr=expression_str,
                  grammar=grammar,
                  code=parse_tree_code,
                  p=probability,
                  sym_params=symbols_params,
                  sym_vars=symbols_variables,
                  lhs_vars=lhs_variables)

    assert str(model).strip("[]") == expression_str

    assert model.get_error() == 10**9

    result = {"x": [1.2], "fun": 0.001}
    model.set_estimated(result)

    assert str(model.full_expr(model.params)).strip("[]") == "1.2*x"

    X = np.reshape(np.linspace(0, 5, 2), (2, 1))
    y = model.evaluate(X, model.params)

    assert isinstance(y, type(np.array([0])))
    assert sum((y - np.reshape([0, 6.0], (2, 1)))**2) < 1e-15

    model_nice_string = model.nice_print(return_string=True)
    assert model_nice_string == "y = 1.2*x\n"


def test_model_box():
    expr1_str = "x"
    expr2_str = "c*x"
    symbols = {"x": ['x'], "const": "c", "start": "S"}

    models = ModelBox()
    models.add_model(expr1_str, symbols)
    assert len(models) == 1
    models.add_model(expr2_str, symbols, p=0.5, info={"code": "1"})
    assert len(models) == 2
    assert str(models[1]) == str(models["[c0*x]"])
    assert str(models[1]) == "[c0*x]"
    assert models[1].p == 0.5


def test_parameter_estimation_algebraic_1D():
    # y = 2 (x + 0.3)

    np.random.seed(0)
    X = np.linspace(-1, 1, 5).reshape(-1, 1)
    Y = 2.0 * (X + 0.3)
    data = pd.DataFrame(np.hstack((X, Y)), columns=['x', 'y'])

    models = ModelBox()
    models.add_model("C*(x+C)",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=['y'])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'algebraic'

    models = fit_models(models, data, settings=settings)
    test_params = [1.99991279884777, 0.29999493846396297]
    assert np.abs(models[0].get_error() - 7.15435171733259e-05) < 1e-6
    for n, param in enumerate(list(models[0].params.values())):
        assert np.abs(param - test_params[n]) < 1e-6


def test_parameter_estimation_algebraic_2D():
    # y1 = 2 (x + 0.3)
    # y2 = 1.66 x

    np.random.seed(0)
    X = np.linspace(-1, 1, 5).reshape(-1, 1)
    Y1 = 2.0 * (X + 0.3)
    Y2 = 1.66 * X
    data = pd.DataFrame(np.hstack((X, Y1, Y2)), columns=['x', 'y1', 'y2'])

    models = ModelBox()
    models.add_model(["C*(x+C)", "C*x"],
                     symbols={"x": ["x"], "const": "C"},
                     lhs_vars=['y1', 'y2'])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'algebraic'

    models = fit_models(models, data, settings=settings)
    test_params = [1.9999283720986005, 0.29999396346942064, 1.6598867645944537]
    for n, param in enumerate(list(models[0].params.values())):
        assert np.abs(param - test_params[n]) < 1e-6
    assert np.abs(models[0].get_error() - 7.107301643897895e-05) < 1e-6


def test_parameter_estimation_ODE_1D():
    # model: dx = -2x

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    data = pd.DataFrame(np.vstack((t, x)).T, columns=['t', 'x'])

    models = ModelBox()
    models.add_model("C*x",
                     symbols={"x": ["x"], "const": "C"})

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 50

    models = fit_models(models, data, settings=settings)
    assert np.abs(models[0].get_error() - 8.60893804568542e-05) < 1e-6


def test_parameter_estimation_ODE_extra_interpolate():
    # model: dx = -2x + y

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    y = np.arange(-1, 1, 0.2)
    x = 3*np.exp(-2*t) + 1/2*y**2
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = ModelBox()
    models.add_model("C*x + y",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=["x"])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 10

    models = fit_models(models, data, settings=settings)
    assert abs(list(models[0].params.values())[0] - -1.9954050935) < 1e-6


def test_parameter_estimation_ODE_extra_vars():
    # model: dx = -2x

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    y = 5.1*np.exp(-1*t)
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = ModelBox()
    models.add_model("C*x + y",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=["x"])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 10

    models = fit_models(models, data, settings=settings)
    assert np.abs(models[0].get_error() - 0.099941324900947) < 1e-6


def test_parameter_estimation_ODE_2D():
    # model: dx = -2x
    #        dy = -1y (would have to check the value -1)

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    y = 5.1*np.exp(-1*t)
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"})

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'

    models = fit_models(models, data, settings=settings)

    test_params = [-2.0001969388280485, -1.000021388719896]
    for n, param in enumerate(models[0].params.values()):
        assert np.abs(param - test_params[n]) < 1e-6
    assert abs(models[0].get_error() - 7.524305872610019e-05) < 1e-6


def test_parameter_estimation_ODE_partial_observability():
    # model: dx = -2x
    #        dy = -2y

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    data = pd.DataFrame(np.vstack((t, x)).T, columns=['t', 'x'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"},
                     observed_vars=["x"])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'

    models = fit_models(models, data, settings=settings)
    assert models[0].estimated['x'][0] + 2 < 1e-2


def test_parameter_estimation_ODE_teacher_forcing():
    # model: dx = -2x
    #        dy = -2y

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    data = pd.DataFrame(np.vstack((t, x)).T, columns=['t', 'x'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"},
                     observed_vars=["x"])

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["objective_function"]["teacher_forcing"] = True

    models = fit_models(models, data, settings=settings)
    assert abs(models[0].get_error() - 5.7511694660763637e-05) < 1e-6


def test_parameter_estimation_simulate_separately():
    # model: dx = -2x
    #        dy = -1y (would have to check the value -1)

    np.random.seed(0)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    y = 5.1*np.exp(-1*t)
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"})

    settings = deepcopy(settings_original)
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["parameter_estimation"]["simulate_separately"] = True

    models = fit_models(models, data, settings=settings)

    test_params = [-2.0001969388280485, -1.000021388719896]
    for n, param in enumerate(models[0].params.values()):
        assert np.abs(param - test_params[n]) < 1e-6
    assert abs(models[0].get_error() - 7.524305872610019e-05) < 1e-6


def test_parameter_estimation_ODE_solved_as_algebraic():
    # model: dx = -2x

    np.random.seed(0)
    sim_step = 0.1
    t = np.arange(0, 10, sim_step)
    X = 3*np.exp(-2*t)
    dX = np.gradient(X, sim_step)
    data = pd.DataFrame(np.vstack((X, dX)).T, columns=['x', 'dx'])

    models = ModelBox()
    models.add_model(["C*x"],
                     symbols={"x": ["x"], "const": "C"},
                     lhs_vars=["dx"])

    settings = deepcopy(settings_original)
    settings['parameter_estimation']["task_type"] = 'algebraic'

    models = fit_models(models, data, settings=settings)
    assert abs(models[0].get_error() - 0.04928780981951337) < 1e-6


def test_equation_discoverer():
    # y = 2*x + 0.3

    np.random.seed(0)
    def f(x):
        return 2.0 * (x[:, 0] + 0.3)

    X = np.linspace(-1, 1, 20).reshape(-1,1)
    Y = f(X).reshape(-1, 1)
    data = pd.DataFrame(np.hstack((X, Y)), columns=["x", "y"])

    ED = EqDisco(data=data,
                 task=None,
                 rhs_vars=["x"],
                 lhs_vars=["y"],
                 sample_size=5,
                 verbosity=0)

    ED.generate_models()
    ED.fit_models()
    assert np.abs(list(ED.models[1].params.values())[0] - 2.550349722044572) < 1e-6
    assert np.abs(ED.models[0].get_error() - 0.853475865) < 1e-6

#
def test_equation_discoverer_ODE():
    # dy = x + 0.4y

    np.random.seed(0)
    B = -2.56; a = 0.4; ts = np.linspace(0.45, 0.87, 5)
    ys = (ts+B)*np.exp(a*ts); xs = np.exp(a*ts)
    data = pd.DataFrame(np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1), ys.reshape(-1, 1))), columns=['t', 'x', 'y'])

    settings = deepcopy(settings_original)
    settings['parameter_estimation']['task_type'] = 'differential'
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 1

    ED = EqDisco(data=data,
                 task=None,
                 task_type="differential",
                 rhs_vars=["x", "y"],
                 lhs_vars=["y"],
                 sample_size=4,
                 generator_template_name="polynomial",
                 verbosity=0)
    np.random.seed(95)
    ED.generate_models()
    ED.fit_models(settings=settings)

    test_params = [-9.667920152096535, -9.058465306540445, -9.317236898854375,]
    for n, param in enumerate(ED.models[1].params.values()):
        assert np.abs(param - test_params[n]) < 1e-6
    return


if __name__ == "__main__":

    # test_grammar_general()
    # test_grammar_templates()
    # test_generate_models()
    # test_model()
    # test_model_box()
    # test_parameter_estimation_algebraic_1D()
    # test_parameter_estimation_algebraic_2D()
    # test_parameter_estimation_ODE_1D()
    test_parameter_estimation_ODE_extra_interpolate()
    # test_parameter_estimation_ODE_extra_vars()
    # test_parameter_estimation_ODE_2D()
    # test_parameter_estimation_ODE_partial_observability()
    # test_parameter_estimation_ODE_teacher_forcing()
    # test_parameter_estimation_simulate_separately()
    # test_parameter_estimation_ODE_solved_as_algebraic()
    # test_equation_discoverer()
    test_equation_discoverer_ODE()

##

