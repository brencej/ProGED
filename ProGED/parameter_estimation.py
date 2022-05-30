# -*- coding: utf-8 -*-
"""Methods for estimating model parameters. Currently implemented:
        differential evolution, hyperopt.

Methods:
    fit_models: Performs parameter estimation on given models. Main interface to the module.
"""

import os
import sys
import time

import numpy as np
import sympy
from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint
import sympy as sp
# from sklearn import ensemble #, tree  # Left for gitch_doctor metamodel
from _io import TextIOWrapper as stdout_type
import ProGED.mute_so as mt
from ProGED.examples.tee_so import Tee
from ProGED.model_box import ModelBox
from ProGED.task import TASK_TYPES


# from ProGED.optimizers import DE_fit, DE_fit_metamodel, hyperopt_fit, min_fit
# glitch-doctor downloaded from github:
# from ProGED.glitch_doctor.metamodel import Metamodel
# import ProGED.glitch_doctor.metamodel.Metamodel
# import ProGED.glitch_doctor.model.Model

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
warnings.filterwarnings("ignore", message="invalid value encountered in power")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="overflow encountered in square")
warnings.filterwarnings("ignore", message="overflow encountered in double_scalars")


def model_error (params, model, X, Y, _T=None, estimation_settings=None):
    """Defines mean squared error as the error metric."""
    try:
        verbosity = estimation_settings['verbosity']

        testY = model.evaluate(X, *params)
        res = np.mean((Y-testY)**2)
        if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            if verbosity >= 3:
                print("isnan, isinf, isreal =", np.isnan(res),
                        np.isinf(res), not np.isreal(res))
                print(model.expr, model.params, model.sym_params, model.sym_vars)
            return estimation_settings['default_error']
        if verbosity >= 3:
            print("Function model_error did not encounter any "
                "errors, the output *square error/loss* is legit.")
        return res
    except Exception as error:
        if verbosity >= 2:
            print("model_error: Params at error:", params,
                  f"and {type(error)} with message:", error)
        if verbosity >= 1:
            print(f"Program is returning default_error:"
                    f"{estimation_settings['default_error']}")
        return estimation_settings['default_error']


def model_error_general (params, model, X, Y, T, **estimation_settings):
    """Calculate error of model with given parameters in general with
    type of error given.
        Input = TODO:
    - X are columns without features that are derived.
    - Y are columns of features that are derived via ode fitting.
    - T is column of times at which samples in X and Y happen.
    - estimation_settings: look description of fit_models()
    """
    task_type = estimation_settings["task_type"]
    if task_type in ("algebraic", "integer-algebraic"):
        return model_error(params, model, X, Y, _T=None,
                            estimation_settings=estimation_settings)
    elif task_type == "differential":
        return model_ode_error(params, model, X, Y, T, estimation_settings)
    else:
        types_string = "\", \"".join(TASK_TYPES)
        raise ValueError("Variable task_type has unsupported value "
                f"\"{task_type}\", while list of possible values: "
                f"\"{types_string}\".")


def ode(models_list, params_matrix, T, X_data, y0, **estimation_settings):
    """Solve system of ODEs defined by equations in models_list.

    Raise error if input is incompatible.
        Input:
    models_list -- list (not dictionary) of models that e.g.
        generate_models() generates.
    params_matrix -- list of lists or ndarrays of parameters for
        corresponding models.
    y0 -- array (1-dim) of initial value of vector function y(t)
        i.e. y0 = y(T[0]) = [y1(T[0]), y2(T[0]), y3(T[0]),...].
    X_data -- 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
    T -- (1-dim) array, i.e. of shape (N,)
    max_ode_steps -- maximal number of steps inside ODE solver to
        determine the minimal step size inside ODE solver.
        Output:
    Solution of ODE evaluated at times T.
    """

    # 1. Check input
    if not (isinstance(models_list, list)
            and isinstance(params_matrix, list)
            and len(params_matrix) > 0
            and isinstance(params_matrix[0], (list, np.ndarray))
            and X_data.ndim == 2
            and y0.ndim == 1):
        message = str(type(params_matrix[0])) + "\n"
        info = (isinstance(models_list, list),
                isinstance(params_matrix, list),
                len(params_matrix)>0,
                isinstance(params_matrix[0], (list, np.ndarray)),
                X_data.ndim == 2)
        print(message, info)
        print("Function ode's defined error: Input arguments are not in the required form!")
        raise TypeError(f"Function ode's defined error: Input arguments are not in required form!"
                        + f"\n{message, info}")
    elif not T.shape[0] == X_data.shape[0]:
        print("Number of samples in T and X does not match.")
        raise IndexError("Number of samples in T and X does not match.")
    elif not (y0.shape[0] == X_data.shape[1]):
        print("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")
        raise IndexError("Number of symbols in models and combination of "
                        + "number of equations and dimensions of input data"
                        + " does not match.")

    # 2. Simulate

    # Set min_step via prescribing maximum number of steps:
    if "max_ode_steps" in estimation_settings:
        max_steps = estimation_settings["max_ode_steps"]
    else:
        # max_steps = 10**6  # On laptop, this would need less than 3 seconds.
        max_steps = T.shape[0] * 10 ** 3  # Set to |timepoints|*1000.
    # Convert max_steps to min_step:
    min_step_from_max_steps = abs(T[-1] - T[0]) / max_steps
    # The minimal min_step to avoid min step error in LSODA:
    min_step_error = 10 ** (-15)
    min_step = max(min_step_from_max_steps, min_step_error)  # Force them both.

    # simulate
    lamb_odes = [model.lambdify(*params) for model, params in zip(models_list, params_matrix)]

    def lambdified_odes(t, x):
        x, y = x
        return [lamb_odes[i](x, y) for i in range(len(lamb_odes))]

    Yode = odeint(lambdified_odes, y0, T,
                  rtol=estimation_settings['odeint_rtol'],
                  atol=estimation_settings['odeint_atol'],
                  hmin=min_step,
                  tfirst=True)

    return Yode

def model_ode_error(params, model_list, X, Y, T, estimation_settings):
    """Defines mean squared error of solution to differential equation
    as the error metric.

        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns without features that are derived.
        - Y are columns of features that are derived via ode fitting.
    """

    # split the complete set of parameters in subsets for each equation based on the num of parameters each eq has
    params_matrix = [list(params[s:e]) for s, e in estimation_settings['param_split_indices']]
    #print('Iter ' + str(estimation_settings["iter"]))
    estimation_settings["iter"] += 1
    #print(params_matrix)

    try:
        # Next few lines strongly suppress any warnning messages
        # produced by LSODA solver, called by ode() function.
        # Suppression further complicates if making log files (Tee):
        change_std2tee = False  # Normaly no need for this mess.
        if isinstance(sys.stdout, Tee):
            # In this case the real standard output (sys.stdout) is not
            # saved in original location sys.stdout. We have to obtain
            # it inside of Tee object (look module tee_so).
            tee_object = sys.stdout  # obtain Tee object that has sys.stdout
            std_output = tee_object.stdout  # Obtain sys.stdout.
            sys.stdout = std_output  # Change fake stdout to real stdout.
            change_std2tee = True  # Remember to change it back.
        def run_ode():
            return ode(model_list, params_matrix, T, X, y0=X[0,:],
                       **estimation_settings)  # Y[:1] if _ or Y[0] if |
        # Next line works only when sys.stdout is real. Thats why above.
        if isinstance(sys.stdout, stdout_type):
            with open(os.devnull, 'w') as f, mt.stdout_redirected(f):
                try:
                    simX = run_ode()
                except Exception as error:
                    if estimation_settings["verbosity"] >= 1:
                        print("Inside ode(), preventing tee/IO error. Params at error:",
                            params_matrix, f"and {type(error)} with message:", error)
        else:
            simX = run_ode()
        if change_std2tee:
            sys.stdout = tee_object  # Change it back to fake stdout (tee).

        try:
            res = np.mean((X-simX)**2)
            if estimation_settings["verbosity"] >= 4:
                print("succesfully returning now inside model_ode_error")
            if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            # #    print(model.expr, model.params, model.sym_params, model.sym_vars)
                return estimation_settings['default_error']
            return res

        except Exception as error:
            if estimation_settings["verbosity"] >= 1:
                print("Error in ode() in mean(Y-odeY): Params at error:",
                    params_matrix, f"and {type(error)} with message:", error)
            return estimation_settings['default_error']

    except Exception as error:
        if estimation_settings["verbosity"] >= 1:
            print("Excepted an error inside ode() of model_ode_error.")
            print("Params at error:", params_matrix,
                    f"and {type(error)} with message:", error)
            print("Returning default error. All is well.")
        return estimation_settings['default_error']

def hyperopt_fit (model, X, Y, T, p0, **estimation_settings):
    """Calls Hyperopt.
    Exists to make passing arguments to the objective function easier.

    Arguments:
        model, X, Y, T, p0, estimation_settings: Just like other
            optimizers, see DE_fit.
        estimation_settings (dict): Optional. Arguments to be passed
                to the parameter estimation. See the documentation of
                ProGED.fit_models for details about more generally
                available options (keys).
            Options specific for hyperopt_fit only (See Hyperopt's
                    documentation for more details.):
                hyperopt_algo (function): The search algorithom used
                    by Hyperopt. See 'algo' argument in hyperopt.fmin.
                    Defult: rand.suggest.
                hyperopt_max_evals (int): The maximum number of
                    evaluations of the objective function.
                    See 'max_evals' argument in hyperopt.fmin.
                    Default: 500.
                hyperopt_space_fn (function): Function used in
                    search space expression.
                    Read below for more info. Default: hp.uniform.
                hyperopt_space_args (hyperopt.pyll.base.Apply):
                    Arguments used in conjunction with
                    hyperopt_space_fn function call when specifying
                    the search space.
                    Default: (lower_bound, upper_bound).
                hyperopt_space_kwargs (hyperopt.pyll.base.Apply): Same
                    as hyperopt_space_args except that it is dictionary
                    for optional arguments. Default: {}.

    In context to ProGED, I currently see possible personal
    configuration in one dimension only. This is because user cannot
    predict how many and which parameters will the random generator
    generate.

    One-dimensional configuration of search space is here specified
    by function called in the stochastic space parameter expression
    and by this expression's (also optional) arguments as described at:
    https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions
    at 0b49cde7c0.
    The function is specified via the `hyperopt_space_fn` setting and
    (optional) arguments are similarly specified via
    the `hyperopt_space_(kw)args` settings. As a result, this is
    called:
        hyperopt_space_fn('label_i-th_dim', *hyperopt_space_args,
                        **hyperopt_space_kwargs)
    to produce the 1-D search space which is then copied a few times
    into a n-dim list with distinct labels to avoid hyperopt error.
        E.g. if passing settings:
      estimation_settings["hyperopt_space_fn"]=hp.randint and
      estimation_settings["hyperopt_space_args"]=(lower_bound, upper_bound)
    in case of p0=(2.45, 6.543, 6.5),
    the search space will be:
    [hp.randint('C0', upper_bound),
     hp.randint('C1', upper_bound),
     hp.randint('C2', upper_bound)], since p0 is 3-D.

    When omitting arguments in stochastic expression, it defaults
    to arguments=(lower_bound, upper_bound), which are allways
    present inside of estimation_settings input. This default
    behaviour is prone to errors in combination of unknown space
    functions.

    List of space functions behaving well without specifying arguments:
        - hp.randint
        - hp.uniform (default)
        - hp.loguniform

    Defaults:
        If search space function or arguments are unspecified, then
        the 1-D space:
            hp.uniform('Ci', lower_bound, upper_bound)
        is used.
    """

    from hyperopt import hp, fmin, rand, pyll, Trials
    import hyperopt.pyll.stochastic
    verbosity = estimation_settings["verbosity"]
    lu_bounds = estimation_settings["lower_upper_bounds"]
    lower_bound, upper_bound = lu_bounds[0]+1e-30, lu_bounds[1]+1e-30

    space_fn = estimation_settings.get("hyperopt_space_fn", hp.uniform)
    if space_fn not in {hp.randint, hp.uniform, hp.loguniform}:
        if verbosity >= 1:
            print(
                f"hyperopt_fit's warnings: "
                f"Input estimation_settings[\"hyperopt_space_fn\"]={space_fn} "
                f"should be used carefully, since it is not recognized as the"
                f" member of the default configuration of the form"
                f" space_fn('label', low, high).\n"
                f"Therefore make sure the function is compatible with search"
                f" space arguments ( hyperopt_space_(kw)args ).\n"
                f"In doubt use one of:\n  - hp.randint\n  - hp.uniform\n"
                f"  - hp.loguniform")

    # User can specify one dimensional search space, which is then replicated.
    args = estimation_settings.get("hyperopt_space_args", ())
    kwargs = estimation_settings.get("hyperopt_space_kwargs", {})
    if args != () or kwargs != {}:
        space = [space_fn('C'+str(i), *args, **kwargs) for i in range(len(p0))]
    else:
        space = [space_fn('C'+str(i), lower_bound, upper_bound) for i in range(len(p0))]

    def objective(params):
        # First way for solution:
        params = [float(i) for i in params]  # Use float instead of np.int32.
        return estimation_settings["objective_function"](
            params, model, X, Y, T, estimation_settings)
    # Use user's hyperopt specifications or use the default ones:
    algo = estimation_settings.get("hyperopt_algo", rand.suggest)
    max_evals = estimation_settings.get("hyperopt_max_evals", 500)
    timeout = estimation_settings["timeout"]

    # My testing code. Delete this block:
    # if str(model.expr) == "C0*exp(C1*n)":
    #     estimation_settings["timeout"] = estimation_settings["timeout_privilege"]
    #     max_evals = max_evals*10
    #     print("This model is privileged.")

    if verbosity >= 3:
        print(f"Hyperopt will run with specs:\n"
              f"  - search space:\n" + "".join([str(i)+"\n" for i in space])
              # + f"  - algorithm: {algo}\n"
              + f"  - timeout: {timeout}\n  - max_evals: {max_evals}")
        print("A few points generated from the space specified:")
        for i in range(10):
            print(hyperopt.pyll.stochastic.sample(space))

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=algo,
        trials=trials,
        timeout=timeout,
        max_evals=max_evals,
        rstate=np.random,
        verbose=(verbosity >= 1),
        )
    params = list(best.values())
    result = {"x": params, "fun": min(trials.losses())}
    if verbosity >= 3:
        print(result)
    return result

def DE_fit (model, X, Y, T, p0, **estimation_settings):
    """Calls scipy.optimize.differential_evolution.
    Exists to make passing arguments to the objective function easier."""


    lu_bounds = estimation_settings["lower_upper_bounds"]
    lower_bound, upper_bound = lu_bounds[0]+1e-30, lu_bounds[1]+1e-30
    bounds = [[lower_bound, upper_bound] for i in range(len(p0))]

    start = time.perf_counter()
    def diff_evol_timeout(x=0, convergence=0):
        now = time.perf_counter()
        if (now-start) > estimation_settings["timeout"]:
            if estimation_settings['verbosity'] >= 1:
                print("Time out!!!")
            return True
        else:
            return False

    return differential_evolution(estimation_settings["objective_function"],
                                  bounds=bounds,
                                  args=[model, X, Y, T, estimation_settings],
                                  maxiter=estimation_settings["DE_max_iter"],
                                  strategy=estimation_settings["DE_strategy"],
                                  popsize=estimation_settings["DE_pop_size"],
                                  mutation=(estimation_settings["DE_f"], 1),
                                  recombination=estimation_settings["DE_cr"],
                                  tol=estimation_settings["DE_tol"],
                                  atol=estimation_settings["DE_atol"],
                                  callback=diff_evol_timeout)

def min_fit (model, X, Y):
    """Calls scipy.optimize.minimize. Exists to make passing arguments to the objective function easier."""
    return minimize(optimization_wrapper, model.params, args=(model, X, Y))

class ParameterEstimator:
    """Wraps the entire parameter estimation, so that we can pass the map function in fit_models
        a callable with only a single argument.
        Also checks some basic requirements, suich as minimum and maximum number of parameters.

        TODO:
            add inputs to make requirements flexible
            add verbosity input
        Input:
            estimation_settings: Dictionary with multiple parameters
                that determine estimation process more specifically.
    """
    def __init__(self, data, time_index, estimation_settings):

        task_type = estimation_settings["task_type"]
        var_mask = np.ones(data.shape[-1], bool)

        if task_type == "differential":
            var_mask[time_index] = False
            self.X = data[:, var_mask]
            self.Y = None
            self.T = data[:, time_index]
            estimation_settings["objective_function"] = model_ode_error

        elif task_type == "algebraic" or task_type == "integer-algebraic":
            target_index = estimation_settings["target_variable_index"]
            self.X = data[:, var_mask]
            self.Y = data[:, target_index]
            self.T = None
            estimation_settings["objective_function"] = model_error

        else:
            types_string = "\", \"".join(TASK_TYPES)
            raise ValueError("Variable task_type has unsupported value "
                             f"\"{task_type}\", while list of possible values: "
                             f"\"{types_string}\".")

        if task_type == "integer-algebraic":
            estimation_settings["objective_function"] = (
                lambda params, model, X, Y, _T, estimation_settings:
                model_error(np.round(params), model, X, Y, _T=None,
                            estimation_settings=estimation_settings))

        self.estimation_settings = estimation_settings


    def fit_one (self, model):
        optimizer_library = {"differential_evolution": DE_fit, "hyperopt": hyperopt_fit, "minimize": min_fit}

        if self.estimation_settings["verbosity"] > 0:
            print("Estimating model " + str(model.expr))
        try:
            if len(model.params) > self.estimation_settings["max_constants"]:
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun": model_error_general(
                    [], model, self.X, self.Y, self.T, **self.estimation_settings)})
            else:
                optimizer = optimizer_library[self.estimation_settings['optimizer']]
                res = optimizer(model, self.X, self.Y, self.T, p0=model.params, **self.estimation_settings)
                model.set_estimated(res)
                if self.estimation_settings["verbosity"] >= 3:
                    print(res, type(res["x"]), type(res["x"][0]))

        except Exception as error:
            if self.estimation_settings["verbosity"] >= 1:
                print((f"Excepted an error inside fit_one: Of type "
                        f"{type(error)} and message:{error}!! \nModel:"), model)
            model.set_estimated({}, valid=False)

        if self.estimation_settings["verbosity"] > 0:
            print(f"model: {str(model.get_full_expr()):<70}; "
                    + f"p: {model.p:<23}; "
                    + f"error: {model.get_error()}")

        return model

    def fit_systemODE(self, system):

        # next 10 lines added by nina
        # split the complete set of parameters in subsets for each equation based on the num of paramaters each eq has
        param_split_indices = []
        for ii in range(len(system)):
            numParams = len(system[ii].sym_params)
            if ii == 0:
                param_split_indices.append((0, numParams))
            else:
                param_split_indices.append((param_split_indices[-1][1], param_split_indices[-1][1] + numParams))

        self.estimation_settings['param_split_indices'] = param_split_indices

        optimizer_library = {"differential_evolution": DE_fit, "hyperopt": hyperopt_fit, "minimize": min_fit}
        system_size = len(system)
        system_params = [item for sublist in system for item in sublist.params]

        if self.estimation_settings["verbosity"] > 0:
            print('Estimating system: ')
            print([str(system[i].expr) for i in range(system_size)])
        try:
            optimizer = optimizer_library[self.estimation_settings['optimizer']]
            res = optimizer(system, self.X, self.Y, self.T, p0=system_params, **self.estimation_settings)
            # added by Nina, temporarly, until set_estimate gets fixed for systems of odes.
            if res.success:
                for i in range(len(system)):
                    system[i].set_estimated(res)
                    system[i].params = list(res["x"][param_split_indices[i][0]:param_split_indices[i][1]])
                    system[i].valid = True
            else:
                print('Issue error! Optimization wasnt successful')

            if self.estimation_settings["verbosity"] >= 3:
                print(res, type(res["x"]), type(res["x"][0]))

        except Exception as error:
            if self.estimation_settings["verbosity"] >= 1:
                print((f"Excepted an error inside fit_systemODE: Of type "
                        f"{type(error)} and message:{error}!! \nModel:"), system)
            [system[i].set_estimated({}, valid=False) for i in range(system_size)]

        #if self.estimation_settings["verbosity"] > 0:
        #    print(f"model: {str(system.get_full_expr()):<70}; "
        #            + f"p: {system.p:<23}; "
        #            + f"error: {system.get_error()}")

        return system

def fit_models (models, data, time_index=None, pool_map=map, verbosity=0,
                task_type="algebraic", estimation_settings={}):

    """Performs parameter estimation on given models. Main interface to the module.

    Supports parallelization by passing it a pooled map callable.

    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted.
        data (numpy.array): Input data of shape N x M, where N is the number of samples
            and M is the number of variables.
        target_variable_index (int): Index of column in data that belongs to the target variable.
        time_index (int): Index of column in data that belongs to measurement of time.
                Required for differential equations, None otherwise.
        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, data, -1, pool_map = pool.map)
        verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
        task_type (str): Type of equations, e.g. "algebraic" or "differential", that
            equation discovery algorithm tries to discover.
        estimation_settings (dict): Dictionary where majority of optional arguments is stored
                and where additional optional arguments can be passed to lower level parts of
                equation discovery.
            arguments that can be passed via estimation_settings dictionary:
                max_constants (int): Maximum number of free constants allowed. For the sake of computational
                    efficiency, models exceeding this constraint are ignored. Default: 5.
                timeout (float): Maximum time in seconds consumed for whole
                    minimization optimization process, e.g. for differential evolution, that
                    is performed for each model.
                lower_upper_bounds (tuple[float]): Pair, i.e. tuple of lower and upper
                    bound used to specify the boundaries of optimization, e.g. of
                    differential evolution.
                max_ode_steps (int): Maximum number of steps used in one run of LSODA solver.
    """
    estimation_settings_preset = {
        "task_type": task_type,
        "target_variable_index": 1,
        "timeout": np.inf,
        "max_constants": 5,
        "lower_upper_bounds": (-30, 30),
        "default_error": 10**9,
        "optimizer": 'differential_evolution',
        "DE_f": 0.45,
        "DE_cr": 0.88,
        "DE_max_iter": 1000,
        "DE_pop_size": 100,
        "DE_atol": 0.01,
        "DE_tol": 0.01,
        "DE_strategy": 'rand1bin',
        "odeint_atol": 10 ** (-6),
        "odeint_rtol": 10 ** (-4),
        "odeint_max_step": 10 ** 3,
        "param_split_indices": [],
        "verbosity": verbosity,
        }
    estimation_settings_preset.update(estimation_settings)
    estimation_settings = estimation_settings_preset
    estimator = ParameterEstimator(data, time_index, estimation_settings)

    if estimation_settings["task_type"] == 'differential':
        fitted_models = list(pool_map(estimator.fit_systemODE, models))
        return fitted_models
    else:
        fitted_models = list(pool_map(estimator.fit_one, models.values()))
        fitted_models_dict = dict(zip(models.keys(), fitted_models))
        return ModelBox(fitted_models_dict)



if __name__ == "__main__":

    # 1. test (older)
    print("--- parameter_estimation.py test --- ")
    np.random.seed(2)

    from pyDOE import lhs
    from generators.grammar import GeneratorGrammar
    from generate import generate_models

    def testf (x):
        return 3*x[:,0]*x[:,1]**2 + 0.5

    X = lhs(2, 10)*5
    X = X.reshape(-1, 2)
    y = testf(X).reshape(-1,1)
    data = np.hstack((X,y))

    grammar = GeneratorGrammar("""S -> S '+' T [0.4] | T [0.6]
                              T -> 'C' [0.6] | T "*" V [0.4]
                              V -> 'x' [0.5] | 'y' [0.5]""")
    symbols = {"x":['x', 'y'], "start":"S", "const":"C"}
    N = 10

    models = generate_models(grammar, symbols, strategy_settings={"N":10})
    models = fit_models(models, data, task_type="algebraic")
    print(models)



