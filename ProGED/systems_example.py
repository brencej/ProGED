import numpy as np
import sympy as sp

import ProGED as pg
from ProGED.examples.generate_data_ODE_systems import generate_ODE_data

if __name__ == "__main__":
    # Part 1: manually add a system to a ModelBox

    # example: two expression strings and the symbols dictionary
    ex1 = "C*C*x - C**2*x*y"
    ex2 = "C*x*y - sin(C*y) + C + C*x/x"
    symbols = {"x": ["x", "y"], "const": "C"}

    # create the model box, add the system by giving it a list of expression strings and the symbols dict
    models = pg.ModelBox()
    models.add_system([ex1, ex2], symbols=symbols)
    print(models)

    # using the model to compute derivatives
    # some fake data
    X = np.random.random((10, 2))
    # lambdify to get a callable function with current parameters
    f = models[0].lambdify()
    print(f(X))

    # Part 2: using a grammar to generate many systems

    # example: 3-dimensional system with variables x, y, z
    # let's use a  polynomial grammar with no special functions
    grammar = pg.grammar_from_template("polynomial", 
        generator_settings = {"variables": ["'x'", "'y'", "'z'"], "p_vars": [1/3, 1/3, 1/3], "functions": [], "p_F": []})
    symbols = {"x": ["x", "y", "z"], "const": "C"}
    # generate_models automatizes the monte-carlo generation and returns a ModelBox
    # we need to tell it the dimension of the system (default 1)
    models = pg.generate.generate_models(grammar, symbols, dimension=3)
    print(models)

    # Part 3: generate data of particular system of ODEs
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8])

    # Part 4: parameter estimation on models
    optimizer_settings_preset = {
        "lower_upper_bounds": (-3, 3),
        "default_error": 10 ** 9,
        "strategy": 'rand1bin',
        "f": 0.45,
        "cr": 0.88,
        "max_iter": 1000,
        "pop_size": 50,
        "atol": 0.01,
        "tol": 0.01
    }
    estimation_settings = {"optimizer": 'differential_evolution',
                           "optimizer_settings": optimizer_settings_preset}

    # without grammar
    ex1 = "C*y"
    ex2 = "C*y - C*x*x*y - C*x"
    symbols = {"x": ["x", "y"], "const": "C"}

    # create the model box, add the system by giving it a list of expression strings and the symbols dict
    system = pg.ModelBox()
    system.add_system([ex1, ex2], symbols=symbols)
    print(system)
    models_out = pg.fit_models(system, data, task_type='differential', time_index=0, estimation_settings=estimation_settings)

    # with grammar
    # models_out2 = pg.fit_models(models, data, task_type='differential', time_index=0, estimation_settings=estimation_settings)
