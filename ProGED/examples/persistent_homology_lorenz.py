import time
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
from ProGED.configs import settings

np.random.seed(0)


# import numpy as np
#
# from ProGED.model_box import ModelBox
# from ProGED.parameter_estimation import fit_models
# from utils.generate_data_ODE_systems import generate_ODE_data

def test_persistent_homology_partial_observability():
    """Not the representative case, more like error case, to check against errors."""

    generation_settings = {"simulation_time": 0.25}
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8], **generation_settings)
    data = data[:, (0, 1)]  # y, 2nd column, is not observed
    system = ModelBox(observed=["x"])
    system.add_system(["C*y", "C*y - C*x*x*y - C*x"], symbols={"x": ["x", "y"], "const": "C"})
    estimation_settings = {"target_variable_index": None,
                           "time_index": 0,
                           "optimizer": 'DE_scipy',
                           "objective_settings": {"use_jacobian": False,
                                                  "persistent_homology": True,
                                                  },
                           "optimizer_settings": {"max_iter": 1,
                                                  "pop_size": 1},
                           "verbosity": 2,
                           }
    np.random.seed(0)
    system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
    print(f"All iters (as saved to system_model object): {system_out[0].ph_all_iters}")
    print(f"Iters when PH was used: {system_out[0].ph_used}")
    print(f"Iters when zero vs zero: {system_out[0].ph_zerovszero}")
    # print(abs(system_out[0].get_error()))
    assert abs(system_out[0].get_error()) < 1  # 3.2.2023
    # true params: [[1.], [-0.5., -1., 0.5]]

def test_parameter_estimation_persistent_homology_lorenz():
    """The representative case, here we should see the beauty of this metric."""
    # model: dx = 10(y - x)
    #        dy = x(28 - z) - y
    #        dz = xy - 2.66667z
    print("dx = 10(y - x) \n"
          "dy = x(28 - z) - y \n"
          "dz = xy - 2.66667z")
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

    # estimation_settings = {"target_variable_index": None,
    #                        "time_index": 0,
    #                        "optimizer": 'DE_scipy',
    #                        "optimizer_settings": {"max_iter": 1,
    #                                               "pop_size": 1,
    #                                               "lower_upper_bounds": (-28, 28),
    # "lower_upper_bounds": (-28, 28),

    # optimizer_DE = {
    #     "strategy": 'DE/best/1/bin',
    #     "max_iter": 1000,
    #     "pop_size": 20,
    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["parameter_estimation"]["param_bounds"] = ((-5, 28),)
    settings["objective_function"]["persistent_homology"] = True

    weight = 0.70
    weight = 0.60
    # weight = 0.55
    # weight = 0.57
    # weight = 0.56
    # weight = 0.59
    settings["objective_function"]["persistent_homology_weight"] = weight
    # settings["objective_function"]["persistent_homology_weight"] = 0.99
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 50  # proper
    # 0.00012831224361846863 [-2.00005965419591 * x, -1.00000664264945 * y]  #50 result
    # 0.00012831311978597356 [-2.00005960377112 * x, -1.00000667167653 * y]  #2 result
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 2
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 1
    scale = 20
    scale = 5
    scale = 4
    # scale = 3

    settings["optimizer_DE"]["max_iter"] = 50*scale
    settings["optimizer_DE"]["pop_size"] = scale
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 3
    settings["optimizer_DE"]["verbose"] = True
    # settings["experiment"]["verbosity"] = 2

    # [10.033734086410286, 27.934671707772303, -2.627103944792303] no homo
    # 0.59
    # homo
    # x = -10.03 * x + 10.03 * y
    # y = x * (27.93 - z) - y
    # z = x * y - 2.63 * z
    #
    # 1679585841.0645025
    # 5.244409084320068

    # [10.001624739484754, 27.998029955055976, -2.6717422411357985]
    # 0.6
    # homo
    # x = -10.0 * x + 10.0 * y
    # y = x * (28.0 - z) - y
    # z = x * y - 2.67 * z
    #
    # 1679585906.0587845
    # 5.394258499145508

    start = time.time()
    models = fit_models(models, data, settings=settings)
    duration = time.time()-start
    print(weight)
    print([10, 28, -2.66667])
    print([10.033734086410286, 27.934671707772303, -2.627103944792303], 'no homo')
    # print([10.033734086410286, 27.934671707772303, -2.627103944792303], 'no homo')
    print([18.171839049595256, 27.759853358369256, -3.648575896478916], '0.55 homo')
    print([18.171839049595256, 27.759853358369256, -3.648575896478916], '0.56 homo')
    print([10.001624739484754, 27.998029955055976, -2.671923906124296], '0.59 homo')
    print([10.033734086410286, 27.934671707772303, -2.627103944792303], '0.7 homo')
    print([10.001624739484754, 27.998029955055976, -2.6717422411357985], '0.60 homo')
    print([10.001624739484754, 27.998029955055976, -2.6720044673329273], '0.57 homo')
    print([10.001624739484754, 27.998029955055976, -2.6720044673329273], '0.57 homo')

    params = list(models[0].params.values())
    print(params, f'{weight} homo')
    # assert abs(params[0] - -2.0000604802785835) < 1e-6
    # assert abs(params[1] - -1.0000035839874042) < 1e-6
    # assert abs(params[0] - -2.0000604802785835) < 1e-3
    # assert abs(params[1] - -1.0000035839874042) < 1e-3
    models[0].nice_print()
    print(start)
    print(duration)

    # print(f"All iters (as saved to system_model object): {system_out[0].ph_all_iters}")
    # print(f"Iters when PH was used: {system_out[0].ph_used}")
    # print(f"Iters when zero vs zero: {system_out[0].ph_zerovszero}")
    # print(abs(system_out[0].get_error()))
    # assert abs(system_out[0].get_error()) < 1.0  # 3.2.2023

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

    settings["parameter_estimation"]["task_type"] = 'differential'
    settings["objective_function"]["persistent_homology"] = True
    settings["objective_function"]["persistent_homology_weight"] = 0.70
    # settings["optimizer_DE"]["termination_after_nochange_iters"] = 50  # proper
    # 0.00012831224361846863 [-2.00005965419591 * x, -1.00000664264945 * y]  #50 result
    # 0.00012831311978597356 [-2.00005960377112 * x, -1.00000667167653 * y]  #2 result
    settings["optimizer_DE"]["termination_after_nochange_iters"] = 2
    # settings["optimizer_DE"]["verbose"] = True
    # settings["experiment"]["verbosity"] = 2

    models = fit_models(models, data, settings=settings)
    params = list(models[0].params.values())
    # assert abs(params[0] - -2.0000604802785835) < 1e-6
    # assert abs(params[1] - -1.0000035839874042) < 1e-6
    assert abs(params[0] - -2.0000604802785835) < 1e-3
    assert abs(params[1] - -1.0000035839874042) < 1e-3

if __name__ == "__main__":

    # test_persistent_homology_partial_observability()
    # test_parameter_estimation_ODE_2D_persistent_homology()
    test_parameter_estimation_persistent_homology_lorenz()
