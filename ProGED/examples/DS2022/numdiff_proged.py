import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import ProGED as pg
from scipy.integrate import solve_ivp, odeint
import time

from ProGED.examples.DS2022.generate_data_ODE_systems import generate_ODE_data, lorenz

if __name__ == "__main__":
    np.random.seed(0)

    """ DATA """
    inits = [1,1,1]
    rho = 16
    lor = lambda t, x: lorenz(t, x, rho=rho)
    data = generate_ODE_data(lor, inits)
    T = data[:,0]
    deltaT = T[1]-T[0]
    X = data[:, 1:]
    """ NUMERIC DIFFERENTIATION """
    dX = np.array([np.gradient(Xi, deltaT) for Xi in X.T])

    """ TRUE MODEL """
    symbols = {"x": ["x","y","z"], "const":"C"}
    optimal_model = ["C*(y-x)", "x*(C-z)-y", "x*y-C*z"]
    optimal_model = ["C*x + C*y", "C*x + C*y + C*x*z", "C*x*y + C*z"]

    """ GRAMMAR """
    grammar_settings = {"variables": ["'x'", "'y'", "'z'"], 
                        "p_vars": [1/3, 1/3, 1/3], 
                        "functions": [], "p_F": []}
    grammar = pg.grammar_from_template("polynomial", grammar_settings)

    
    optimizer_settings = {"lower_upper_bounds": (-30,30),
                            "atol": 0.001, "tol": 0.001}
    estimation_settings = {"target_variable_index": 0,
                            "verbosity": 1,
                            "optimizer_settings": optimizer_settings,
                            "max_constants": 3}

    """ ITERATE THROUGH THE THREE DIMENSIONS """
    t1 = time.time()
    for i in range(3):
        """ SETUP THE MODELS """
        models = pg.generate.generate_models(grammar, symbols, strategy_settings={"N":2})
        #models = pg.ModelBox()
        #models.add_model(optimal_model[i], symbols)

        """ FIT PARAMETERS """
        dat = np.hstack((dX[i].reshape((-1,1)), X))
        models_fit = pg.fit_models(models, dat, task_type="algebraic", 
                                    estimation_settings=estimation_settings)
        
        #models_fit.dump("numdiff_lorenz_eq" + str(i) + "_models_fit.pg")
        print(models_fit)
    t2 = time.time()
    print("Time: ", t2-t1)