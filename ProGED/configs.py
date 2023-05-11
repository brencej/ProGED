# todo: write documentation for this file (e.g. this are settings for par. esti. only.

import numpy as np


experiment = {
    "seed": 0,
    "verbosity": 0,
}

parameter_estimation = {
    "task_type": 'algebraic',
    "dataset": "./data.csv",
    "observed_vars": None,
    "lhs_vars": None,
    "optimizer": 'DE',  # default is pymoo's DE
    # "simulate_separately": True,
    "simulate_separately": False,
    # set sepa to default!
    "max_constants": 15,
    "param_bounds": ((-10, 10),),
    "default_error": 10 ** 9,
    "timeout": np.inf,
}

optimizer_DE = {
    "strategy": 'DE/best/1/bin',
    "max_iter": 1000,
    "pop_size": 20,
    "mutation": 0.5,
    "cr": 0.5,
    "tol": 0.001,
    "atol": 0.001,
    "termination_threshold_error": 10 ** (-4),
    "termination_after_nochange_iters": 200,
    "termination_after_nochange_tolerance": 10 ** (-6),
    "verbose": False,
    "save_history": False,
}

optimizer_hyperopt = {
    "a": 1,
}

objective_function = {
    "use_jacobian": False,
    "teacher_forcing": False,
    "atol": 10 ** (-6),
    "rtol": 10 ** (-4),
    "max_step": 10 ** 3,
    "default_error": 10 ** 9,
    "persistent_homology": False,
    "persistent_homology_size": 200,
    "persistent_homology_weight": 0.6,
}

settings = {
    "experiment": experiment,
    "parameter_estimation": parameter_estimation,
    "optimizer_DE": optimizer_DE,
    "optimizer_hyperopt": optimizer_hyperopt,
    "objective_function": objective_function,
}
