import numpy as np
from scipy.integrate import odeint
import ProGED as pg
from ProGED.examples.ds2022.generate_data_ODE_systems import generate_ODE_data
import matplotlib.pyplot as plt

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

    ###################################### Part 3: VPD ######################################

    # 3.1 generate data
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8])

    # 3.2 parameter estimation settings
    optimizer_settings = {
        "lower_upper_bounds": (-30, 30),
        "default_error": 10 ** 9,
        "strategy": 'rand1bin',
        "f": 0.45,
        "cr": 0.50,
        "max_iter": 1000,
        "pop_size": 50,
        "atol": 0.0001,
        "tol": 0.0001
    }
    estimation_settings = {"optimizer": 'differential_evolution',
                           "observed": ["x", "y"],
                           "optimizer_settings": optimizer_settings}

    # 3.3 define fully observed system
    ex1 = "C*y"
    ex2 = "C*y - C*x*x*y - C*x"
    symbols = {"x": ["x", "y"], "const": "C"}
    system = pg.ModelBox()
    system.add_system([ex1, ex2], symbols=symbols)
    models_out_full = pg.fit_models(system, data, task_type='differential', time_index=0,
                                    estimation_settings=estimation_settings)

    # 3.4 define partially observed system
    system = pg.ModelBox(observed=["y"])
    system.add_system([ex1, ex2], symbols=symbols)
    models_out_partial = pg.fit_models(system, data[:, (0, 2)], task_type='differential', time_index=0, estimation_settings=estimation_settings)

    # 3.5 with grammar
    # models_out2 = pg.fit_models(models, data, task_type='differential', time_index=0, estimation_settings=estimation_settings)

    ######################################## 4. Lorenz ############################################

    # 5.1 get data
    inits = [0, 1, 1.05]
    data = generate_ODE_data(system='lorenz', inits=inits)

    # 5.2 create fully observed lorenz model
    """
    ex1 = "C*y + C*x"
    ex2 = "C*x + C*x*z + C*y"
    ex3 = "C*x*y + C*z" 
    """

    ex1 = "C*(y-x)"
    ex2 = "x*(C-z)-y"
    ex3 = "x*y - C*z"

    symbols = {"x": ["x", "y", "z"], "const": "C"}
    system = pg.ModelBox()
    system.add_system([ex1, ex2, ex3], symbols=symbols)
    models_out = pg.fit_models(system, data, task_type='differential', time_index=0,
                               estimation_settings=estimation_settings)

    # 5.4 create partially observed model (no grammar)
    ex1 = "C*(y-x)"
    ex2 = "x*(C-z)-y"
    ex3 = "x*y - C*z"
    symbols = {"x": ["x", "y", "z"], "const": "C"}
    system = pg.ModelBox(observed=["z"])
    system.add_system([ex1, ex2, ex3], symbols=symbols)
    models_out = pg.fit_models(system, data[:,(0, 3)], task_type='differential', time_index=0, estimation_settings=estimation_settings)


    # 5.5: lorenz: check if the simulation works by inputing correct parameters
    system[0].params = [[-10., 10.], [-1., 28., -1.], [1., -2.667]]
    lamb_odes = system[0].lambdify(list=True)

    def lambdified_odes(t, x):
        return [lamb_odes[i](*x) for i in range(len(lamb_odes))]

    sol = odeint(lambdified_odes, inits, data[:, 0],
                  rtol=1e-12,
                  atol=1e-12,
                  tfirst=True)

    # plot the solution
    WIDTH, HEIGHT, DPI = 1000, 750, 100
    fig = plt.figure(facecolor='k', figsize=(WIDTH / DPI, HEIGHT / DPI))
    ax = fig.gca(projection='3d')
    ax.set_facecolor('k')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    s = 10
    cmap = plt.cm.winter
    for i in range(0, len(data)):
        ax.plot(sol[:, 0][i:i + s + 1], sol[:, 1][i:i + s + 1], sol[:, 2][i:i + s + 1], color=cmap(i / len(data)), alpha=0.4)

    ax.set_axis_off()
    plt.show()

