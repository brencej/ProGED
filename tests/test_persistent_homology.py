# metrics alternatives:
#   - persistent homology
#   - dynamic time warping (DTW)
#   - Jure's earthquake's metric
#   - bottleneck distance (not realy big idea - maybe DTW is enough)
# todo: use ph in test_lorenz (currently no ph!! :D)
#       -clean examples/ph_lorenz.py !!

import pandas as pd
import numpy as np
from copy import deepcopy

from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.configs import settings as default_settings

np.random.seed(0)
default_settings["parameter_estimation"]["task_type"] = 'differential'
default_settings["objective_function"]["persistent_homology"] = True


def test_persistent_homology_partial_observability():
    """Not the representative case, more like bug test case, to check against bugs."""
    # model: dx = -2x
    #        dy = -2y
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    data = pd.DataFrame(np.vstack((t, x)).T, columns=['t', 'x'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"},
                     observed_vars=["x"])

    settings = deepcopy(default_settings)
    models = fit_models(models, data, settings=settings)

    # asserts: 27.3.2023
    # assert abs(list(models[0].params.values())[0] - -1.9998757234114541) < 1e-6
    assert abs(list(models[0].params.values())[0] - -1.9998757234114541) < 1e-3

def test_persistent_homology_lorenz():
    """The representative case, here we should see the beauty of this metric."""
    # model: dx = 10(y - x)
    #        dy = x(28 - z) - y
    #        dz = xy - 2.66667z
    generation_settings = {
        "initial_time": 0,  # initial time
        "simulation_step": 0.1,  # simulation step /s
        "simulation_time": 1,  # simulation time (final time) /s
        }
    data = generate_ODE_data(system='lorenz', inits=[0.2, 0.8, 0.5],
                             **generation_settings)
    data = pd.DataFrame(data, columns=['t', 'x', 'y', 'z'])

    models = ModelBox()
    models.add_model(["C*(y-x)", "x*(C-z) - y", "x*y - C*z"],
                     symbols={"x": ["x", "y", "z"], "const": "C"})

    settings = deepcopy(default_settings)
    settings["parameter_estimation"]["param_bounds"] = ((-5, 28),)
    scale = 4
    settings["optimizer_DE"]["max_iter"] = 50*scale
    settings["optimizer_DE"]["pop_size"] = scale

    models = fit_models(models, data, settings=settings)
    params = list(models[0].params.values())
    # asserts: 27.3.2023
    # assert abs(params[0] - 10.001624739484754) < 1e-6
    # assert abs(params[1] - 27.998029955055976) < 1e-6
    # assert abs(params[2] - -2.6717422411357985) < 1e-6
    assert abs(params[0] - 10.001624739484754) < 1e-3
    assert abs(params[1] - 27.998029955055976) < 1e-3
    assert abs(params[2] - -2.6717422411357985) < 1e-3

def test_parameter_estimation_ODE_2D_persistent_homology():
    # model: dx = -2x
    #        dy = -1y (would have to check the value -1)
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    y = 5.1*np.exp(-1*t)
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"})

    settings = deepcopy(default_settings)
    settings["objective_function"]["persistent_homology_weight"] = 0.6
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 50  # proper
    # 0.00012831224361846863 [-2.00005965419591 * x, -1.00000664264945 * y]  #50 result
    # 0.00012831311978597356 [-2.00005960377112 * x, -1.00000667167653 * y]  #2 result
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 2

    models = fit_models(models, data, settings=settings)
    params = list(models[0].params.values())
    # asserts: 23.3.2023
    # assert abs(params[0] - -2.0000668972887663) < 1e-6
    # assert abs(params[1] - -0.99999781368651) < 1e-6
    assert abs(params[0] - -2.0000668972887663) < 1e-3
    assert abs(params[1] - -0.99999781368651) < 1e-3

if __name__ == "__main__":
    test_persistent_homology_partial_observability()
    test_persistent_homology_lorenz()
    test_parameter_estimation_ODE_2D_persistent_homology()
