import numpy as np
import sympy as sp

import ProGED as pg

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