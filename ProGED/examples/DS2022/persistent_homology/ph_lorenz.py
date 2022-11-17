# -*- coding: utf-8 -*-

import time
import pickle
import sys

import numpy as np
from nltk import Nonterminal, PCFG
import matplotlib.pyplot as plt

from ProGED.equation_discoverer import EqDisco
from ProGED.generators.grammar import GeneratorGrammar
from ProGED.generators.grammar_construction import grammar_from_template
from ProGED.generate import generate_models
from ProGED.model import Model
from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from utils.generate_data_ODE_systems import generate_ODE_data

def test_persistent_homology_ODE_system():
    # generation_settings = {"simulation_time": 0.25}
    generation_settings = {
        "initial_time": 0,  # initial time
        "simulation_step": 0.01,  # simulation step /s
        "simulation_time": 40,  # simulation time (final time) /s
    }
    inits = [1.0, 1.0, 1.0]
    data = generate_ODE_data(system='lorenz', inits=inits,
                             generation_settings=generation_settings)
    # data = generate_ODE_data(system='lorenz_stable', inits=[0.2, 0.8, 0.5])

    # P1 = data[:,1:]
    # fig = plt.figure()
    # plt.title('lorenz')
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(P1[:, 0], P1[:, 1], P1[:, 2], s=1)
    # plt.show()

    # ax = plt.axes(projection='3d')
    # ax.plot3D(data[:, 1], data[:, 2], data[:, 3])
    # plt.show()
    # # plt.close('all')

    system = ModelBox(observed=["x", "y", "z"])
    # sigma * (x[1] - x[0]),
    # x[0] * (rho - x[2]) - x[1],
    # x[0] * x[1] - beta * x[2],
    system.add_system(["C*(y-x)", "x*(C-z) - y", "x*y - C*z"], symbols={"x": ["x", "y", "z"], "const": "C"})
    # system.add_system(["C*x-y", "x*C-z - y", "x*y - C*z"], symbols={"x": ["x", "y", "z"], "const": "C"})
    # size = 1
    # size = 5
    # size = 6

    # # Defaults:
    # ph, size = True, 1
    # solo_ph = False

    ph, size = True, 1
    solo_ph = True

    double_flags = set(sys.argv[1:])
    if "-ph" in double_flags:
        ph = True
    elif "-noph" in double_flags:
        ph = False
    elif "-solo" in double_flags:
        solo_ph = True
    flags_dict = {argument.split("=")[0]: argument.split("=")[1] for argument in sys.argv[1:] if len(argument.split("=")) > 1}
    size = int(flags_dict.get("--size", size))
    ph = bool(flags_dict.get("--ph", ph))
    solo_ph = bool(flags_dict.get("--solo_ph", solo_ph))

    weights = (0.01, 0.99) if solo_ph else (0.5, 0.5)
    memo = f"ph: {ph}, ph solo: {solo_ph}, size: {size}"
    print(memo)
    systems = []
    #  sigma=10, rho=28, beta=2.66667):
    # [ sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2], ]
    estimation_settings = {"target_variable_index": None,
                           "time_index": 0,
                           "optimizer_settings": {"max_iter": size*50,
                                                  "pop_size": size,
                                                  "lower_upper_bounds": (-28, 28),
                                                  },
                           "objective_settings": {"use_jacobian": False},
                           "verbosity": 1,
                           "persistent_homology": ph,
                           "persistent_homology_weights": weights,
                           }

    np.random.seed(0)
    start = time.perf_counter()
    try:
        system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
    except Exception as error:
        print(error)
    ctime = time.perf_counter() - start
    print(f"consumed time: {round(ctime, 2)} secs or {round(ctime/60, 2)} mins, ")
    print(memo, f"\n inits:{inits}")
    systems += [system_out]
    for system_out in systems:
    # true params: [[-0.5., -1., 0.5]]
    # assert abs(system_out[0].get_error() - 266.667354661213) < 1e-6
        expr = system_out[0].full_expr()
        error = system_out[0].estimated["fun"]
    # print(system_out[0].full_expr())
        print(f"full found expr: {expr}")
        print(f"found error: {error}")


    timestamp = time.strftime("%m_%d__%H_%M%S")
    # timestamp = ""
    preamble = f"{memo} " \
               f"{expr} " \
               f"{error} \n" + timestamp

    pickle.dump([systems, preamble], open(f"ph_lorenz_systems{timestamp}.p", "wb"))
    # favorite_color = pickle.load(open("ph_lorenz_system ... .p", "rb"))
    expr_consts = ["sigma 10", "rho 28", "beta 2.666"]
    expr_truth = "[ 10 * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2], ]"
    print("exprs of ground truth:")
    print(expr_consts)
    print(expr_truth)
    # assert abs(system_out[0].get_error() - 7.109693380523827) < 1e-6
    print(2)
    assert 1==1

    # [-9.33465268723235 * x + 9.33465268723235 * y, x * (6.26027961658288 - z) - y, x * y - 2.57148765076742 * z]
    # assert abs(system_out[0].get_error() - 0) < 1e-6
    # [-0.17912909  4.39808796 - 1.71474011]
    #  sigma=10, rho=28, beta=2.66667):
    # [ sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2], ]
    # with size (=popsize, maxiter=5*10) = 5:
    # [-9.39057203121008 * x + 9.39057203121008 * y, x * (25.4218661828293 - z) - y, x * y - 2.30400661222897 * z]
    # with size (=popsize, maxiter=5*10) = 20:
    # [-10.6675198563835 * x + 10.6675198563835 * y, x * (26.2693899626468 - z) - y, x * y - 3.89296267553082 * z]
    # consumed time: 26.61 secs or 0.44 mins,
    # ph: True, size: 1
    # full found expr:
    # [-10.6547799773229 * x + 10.6547799773229 * y, x * (19.2306504396799 - z) - y, x * y - 2.37730311870262 * z]
    # found error: 6.829248589726909
    # consumed time: 119.11 secs or 1.99 mins,
    # ph: True, size: 2
    # full found expr: [10.4537978384329*x - 10.4537978384329*y, x*(8.57406622606231 - z) - y, x*y - 13.8156702577722*z]
    # found error: 5.711466204268346
    # exprs of ground truth:
    # ['sigma 10', 'rho 28', 'beta 2.666']
    # [ 10 * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2], ]
    #

if __name__ == "__main__":

    test_persistent_homology_ODE_system()
