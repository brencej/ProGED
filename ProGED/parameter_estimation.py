# -*- coding: utf-8 -*-

##################################################################################################
##                               EQUATION PARAMETER ESTIMATION
##################################################################################################

"""
Methods for estimating model parameters.
Possible optimization methods: a) differential evolution from pymoo,
                               b) differential evolution from scipy,
                               c) hyperopt
Possible objective functions:  a) trajectory rmse for ode
                               b) mse for algebraic equations (or numerical differentiation of ode)
                               c) persistent homology calculation (for ode)
Methods:
    fit_models: Performs parameter estimation on given models. Main interface to the module.
"""

import os
import sys
import time
import math
from typing import List
import numpy as np
import sympy as sp

from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint

# from sklearn import ensemble #, tree  # Left for gitch_doctor metamodel
from _io import TextIOWrapper as stdout_type
import ProGED.mute_so as mt
from ProGED.examples.tee_so import Tee
from ProGED.model_box import ModelBox
from ProGED.task import TASK_TYPES

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

def fit_models(models, data, task_type="algebraic", pool_map=map, estimation_settings={}):
    """
    Performs parameter estimation on given models. Main interface to the module.
    Supports parallelization by passing it a pooled map callable.

    Arguments:
        models (ModelBox): Instance of ModelBox, containing the models to be fitted.
        data (numpy.array): Input data of shape N x M, where N is the number of samples
            and M is the number of variables.
        task_type (str): Type of equations, e.g. "algebraic" or "differential", that
            equation discovery algorithm tries to discover.

        pool_map (function): Map function for parallelization. Example use with 8 workers:
                from multiprocessing import Pool
                pool = Pool(8)
                fit_models (models, data, -1, pool_map = pool.map)

        estimation_settings (dict): Dictionary where majority of optional arguments is stored
                and where additional optional arguments can be passed to lower level parts of
                equation discovery.

            arguments that can be passed via estimation_settings dictionary:
                target_variable_index (int): Index of column in data that belongs to the target variable. If None,
                         all variables are considered targets (e.g. for systems ODE).
                time_index (int): Index of column in data that belongs to measurement of time.
                        Required for differential equations, None otherwise.
                max_constants (int): Maximum number of free constants allowed. For the sake of computational
                    efficiency, models exceeding this constraint are ignored. Default: 5.
                timeout (float): Maximum time in seconds consumed for whole
                    minimization optimization process, e.g. for differential evolution, that
                    is performed for each model.
                lower_upper_bounds (tuple[float]): Pair, i.e. tuple of lower and upper
                    bound used to specify the boundaries of optimization, e.g. of
                    differential evolution.
                default_error: if the error for objective function could not be calculated, use this default.
                objective_settings: settings for odeint (scipy.integrate documentation)
                optimizer_settings: settings for differential_evolution (scipy.optimize documentation)
                verbosity (int): Level of printout desired. 0: none, 1: info, 2+: debug.
    """

    objective_settings_preset = {
        "atol": 10 ** (-6),
        "rtol": 10 ** (-4),
        "max_step": 10 ** 3,
        "use_jacobian": False,
        "teacher_forcing": False,
        "simulate_separately": False,
        "persistent_homology": False,
        "persistent_homology_size": 200,
        "persistent_homology_weight": 0.5,
    }

    optimizer_settings_preset = {
        "lower_upper_bounds": (-10, 10),
        "default_error": 10 ** 9,
        "strategy": 'best1bin',
        "f": 0.5,
        "cr": 0.7,
        "max_iter": 1000,
        "pop_size": 20,
        "atol": 0.001,
        "tol": 0.001,
        "pymoo_min_f": 10 ** (-6),
        "hyperopt_seed": None,
    }

    estimation_settings_preset = {
        "target_variable_index": None,
        "time_index": None,
        "max_constants": 15,
        "optimizer": 'DE_pymoo',
        "observed": models.observed,
        "optimizer_settings": optimizer_settings_preset,
        "objective_settings": objective_settings_preset,
        "default_error": 10 ** 9,
        "timeout": np.inf,
        "verbosity": 1,
        }

    estimation_settings_preset.update(estimation_settings)

    if "objective_settings" in estimation_settings:
        objective_settings_preset.update(estimation_settings["objective_settings"])
        estimation_settings_preset["objective_settings"] = dict(objective_settings_preset)
    if "optimizer_settings" in estimation_settings:
        optimizer_settings_preset.update(estimation_settings["optimizer_settings"])
        estimation_settings_preset["optimizer_settings"] = dict(optimizer_settings_preset)

    estimation_settings = dict(estimation_settings_preset)
    estimation_settings["objective_settings"]["verbosity"] = estimation_settings["verbosity"]
    estimation_settings["task_type"] = task_type
    estimator = ParameterEstimator(data, task_type, estimation_settings)

    return ModelBox(dict(zip(models.keys(), list(pool_map(estimator.fit_one, models.values())))))


class ParameterEstimator:
    """ Wraps the entire parameter estimation, so that we can pass the map function in fit_models
        a callable with only a single argument.
        Also checks some basic requirements, such as minimum and maximum number of parameters.

        TODO:
            add inputs to make requirements flexible
            add verbosity input
    """

    def __init__(self, data, task_type, estimation_settings):

        TASK_TYPES = ["algebraic", "differential", "integer-algebraic"]
        target_index = estimation_settings["target_variable_index"]
        time_index = estimation_settings["time_index"]

        # setup of the mask for the data columns
        var_mask = np.ones(data.shape[-1], bool)

        # case of disabled persistent homology
        self.persistent_diagram = None

        ## a. set parameter estimation for differential equations
        if task_type == "differential":

            # check if all required variables are defined
            if time_index is None:
                raise ValueError("\nVariable time_index has to be defined when task_type = differential.\n")
            if target_index is None and estimation_settings["objective_settings"]["simulate_separately"]:
                raise ValueError("\nVariable target_variable_index has to be defined when simulate_separately = true.\n")

            # set parameter estimator values
            estimation_settings["objective_function"] = model_ode_error
            if target_index is not None:
                var_mask[[time_index, target_index]] = False
            else:
                var_mask[time_index] = False

            self.X = data[:, var_mask]
            self.T = data[:, time_index]
            if estimation_settings["objective_settings"]["simulate_separately"] or estimation_settings["target_variable_index"]:
                self.Y = data[:, estimation_settings["target_variable_index"]]
            else:
                self.Y = None

            # take care of persistent homology case, i.e. if using topological distance
            if estimation_settings["objective_settings"]["persistent_homology"]:
                weight = estimation_settings["objective_settings"]["persistent_homology_weight"]
                if (not isinstance(weight, (float, int, np.float))) or weight < 0 or weight > 1:
                    raise TypeError("ERROR: Persistent homology weight should be of type float and in range [0,1]!")
                size = estimation_settings["objective_settings"]["persistent_homology_size"]
                trajectory = np.vstack(np.vstack((self.X, self.Y))) if self.Y is not None else self.X
                try:
                    self.persistent_diagram = ph_diag(trajectory, size=size)
                    if (self.persistent_diagram[1].shape == (0, 2)) and estimation_settings["verbosity"] >= 1:
                        print("INFO: persistent diagram of the ground truth is trivial (empty), "
                              "i.e. no interesting 2D-property is present.")
                except Exception as error:
                    if estimation_settings["verbosity"] >= 1:
                        print(f"WARNNING: Excepted an error when constructing ph_diagram of the original dataset "
                              f"of type {type(error)} and message:{error}!")
                    self.persistent_diagram = None

        ## b. set parameter estimation for algebraic and integer-algebraic equations
        elif task_type == "algebraic" or task_type == "integer-algebraic":

            # check if all required variables are defined
            if estimation_settings["target_variable_index"] is None:
                raise ValueError("\nVariable target_variable_index has to be defined when task_type = algebraic.\n")

            # set parameter estimator values
            var_mask[target_index] = False
            self.X = data[:, var_mask]
            self.Y = data[:, target_index]
            self.T = None
            estimation_settings["objective_function"] = model_error

        ## c. raise error if the task_type doesnt exist
        else:
            types_string = "\", \"".join(TASK_TYPES)
            raise ValueError("Variable task_type has unsupported value "
                             f"\"{task_type}\", while list of possible values: "
                             f"\"{types_string}\".")

        ## additional settings for integer-algebraic equations
        if task_type == "integer-algebraic":
            estimation_settings["objective_function"] = (
                lambda params, model, X, Y, _T, estimation_settings:
                    model_error(np.round(params), model, X, Y, _T=None, estimation_settings=estimation_settings))

        ##
        self.estimation_settings = estimation_settings
        self.verbosity = estimation_settings["verbosity"]


    def fit_one (self, model):
        # fits one model within the models dictionary

        OPTIMIZER_TYPES = {"DE_pymoo":DE_pymoo, "DE_scipy": DE_scipy, "hyperopt": hyperopt_fit}

        if self.verbosity > 0:
            print("Estimating model: " + str(model.expr))
        try:

            ## a. check requirements
            if len(model.params) > self.estimation_settings["max_constants"]:
                print("Model skipped. More model parameters than allowed (check max constant).")
                pass
            elif len(model.params) < 1:
                model.set_estimated({"x":[], "fun": model_error_general(
                    [], model, self.X, self.Y, self.T, self.persistent_diagram,
                    **self.estimation_settings)})

            ## b. estimate parameters
            else:
                try:
                    optimizer = OPTIMIZER_TYPES[self.estimation_settings['optimizer']]
                except:
                    print(f"*** Warning: \nOptimization method not recognized. Choose among: {list(OPTIMIZER_TYPES.keys())}.\n"
                                  f"The default method was used: DE_pymoo. ***")
                    optimizer = OPTIMIZER_TYPES["DE_pymoo"]

                # include all parameters, including unknown initial values in the partially observed scenarios.
                obs_idx = [model.sym_vars.index(sp.symbols(model.observed[i])) for i in range(len(model.observed))]
                model.obs_mask[obs_idx] = True
                if sp.symbols('t') in model.sym_vars:
                    model.obs_mask = model.obs_mask[:-1]
                if np.any(model.obs_mask == False):
                    model_params = model.get_params(flatten=True) + list(model.initials[~model.obs_mask])
                else:
                    model_params = model.get_params(flatten=True)

                # parameter estimation
                t1 = time.time()
                res = optimizer(model, self.X, self.Y, self.T, p0=model_params,
                                ph_diagram=self.persistent_diagram,
                                **self.estimation_settings)
                t2 = time.time()
                res["duration"] = t2-t1
                model.set_estimated(res)
                if self.estimation_settings["verbosity"] >= 2:
                    print(res, type(res["x"]), type(res["x"][0]))

        except Exception as error:
            if self.estimation_settings["verbosity"] >= 1:
                print(f"Excepted an error inside fit_one: Of type {type(error)} and message:{error}! \nModel:", model)
            model.set_estimated({}, valid=False)

        if self.estimation_settings["verbosity"] > 0:
            print(f"model: {str(model.full_expr()):<70}; p: {model.p:<23}; error: {model.get_error()}")

        return model


##################################################################################################
##                               OPTIMIZATION ALGORITHMS
##################################################################################################


#---------------- 1. DIFFERENTIAL EVOLUTION BASED ON SCIPY LIBRARY -------------------------------

def DE_scipy(model, X, Y, T, p0, ph_diagram, **estimation_settings):
    """Calls scipy.optimize.differential_evolution."""

    lu_bounds = estimation_settings['optimizer_settings']['lower_upper_bounds']
    lower_bound, upper_bound = lu_bounds[0]+1e-30, lu_bounds[1]+1e-30
    bounds = [[lower_bound, upper_bound] for i in range(len(p0))]

    start = time.perf_counter()
    def diff_evol_timeout(x=0, convergence=0):
        now = time.perf_counter()
        if (now-start) > estimation_settings["timeout"]:
            if estimation_settings['verbosity'] >= 1:
                print("Time out!")
            return True
        else:
            return False

    output = differential_evolution(func=estimation_settings["objective_function"],
                                  bounds=bounds,
                                  callback=diff_evol_timeout,
                                  args=[model, X, Y, T, ph_diagram, estimation_settings],
                                  maxiter=estimation_settings["optimizer_settings"]["max_iter"],
                                  strategy=estimation_settings["optimizer_settings"]["strategy"],
                                  popsize=estimation_settings["optimizer_settings"]["pop_size"],
                                  mutation=estimation_settings["optimizer_settings"]["f"],
                                  recombination=estimation_settings["optimizer_settings"]["cr"],
                                  tol=estimation_settings["optimizer_settings"]["tol"],
                                  atol=estimation_settings["optimizer_settings"]["atol"])

    return {"x": output.x, "fun": output.fun, "all_results": output}

#---------------- 2. DIFFERENTIAL EVOLUTION BASED ON PYMOO LIBRARY -------------------------------

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.core.termination import Termination
from pymoo.termination.default import MaximumGenerationTermination
class PymooProblem(Problem):

    def __init__(self, params, model, data, target, time, ph_diagram, estimation_settings):

        xl = np.full(len(params), estimation_settings['optimizer_settings']['lower_upper_bounds'][0])
        xu = np.full(len(params), estimation_settings['optimizer_settings']['lower_upper_bounds'][1])
        super().__init__(n_var=len(params), n_obj=1, n_constr=0, xl=xl, xu=xu)
        self.params = params
        self.model = model
        self.data = data
        self.target = target
        self.time = time
        self.ph_diagram = ph_diagram
        self.estimation_settings = estimation_settings
        self.best_f = estimation_settings['optimizer_settings']['default_error']
        self.opt_curve = []

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.asarray(
            [self.estimation_settings["objective_function"](
                x[i,:],
                self.model,
                self.data,
                self.target,
                self.time,
                self.ph_diagram,
                self.estimation_settings) for i in range(len(x))]
        )
        self.best_f = np.min(out["F"])
        self.opt_curve.append(self.best_f)

class BestTermination(Termination):
    def __init__(self, min_f=1e-3, n_max_gen=500) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)
    def _update(self, algorithm):
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        return self.max_gen.update(algorithm)

def DE_pymoo(model, X, Y, T, p0, ph_diagram, **estimation_settings):

    pymoo_problem = PymooProblem(p0, model, X, Y, T, ph_diagram, estimation_settings)

    strategy = "DE/best/1/bin" if estimation_settings["optimizer_settings"]["strategy"] == 'best1bin' else "DE/rand/1/bin"

    algorithm = DE(
        pop_size=estimation_settings["optimizer_settings"]["pop_size"],
        sampling=LHS(),
        variant=strategy,
        CR=estimation_settings["optimizer_settings"]["cr"],
        F=estimation_settings["optimizer_settings"]["f"],
        dither="vector",
        jitter=False
    )

    # Defaul termination
    # termination = DefaultSingleObjectiveTermination(
    #     xtol=1e-8,
    #     cvtol=1e-6,
    #     ftol=1e-6,
    #     period=20,
    #     n_max_gen=1000, #estimation_settings["optimizer_settings"]["max_iter"],
    #     n_max_evals=100000
    # )

    termination = BestTermination(min_f = estimation_settings["optimizer_settings"]["pymoo_min_f"],
                                  n_max_gen = estimation_settings["optimizer_settings"]["max_iter"])

    output = minimize(pymoo_problem,
                      algorithm,
                      termination,
                      seed=1,
                      verbose=False,
                      save_history=False)

    return {"x": output.X, "fun": output.F[0], "all_results": output}

#----------------------------- 3. HYPEROPT  -------------------------------

def hyperopt_fit(model, X, Y, T, p0, ph_diagram, **estimation_settings):
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

    """

    from hyperopt import hp, fmin, rand, pyll, Trials
    import hyperopt.pyll.stochastic
    verbosity = estimation_settings["verbosity"]
    lu_bounds = estimation_settings["optimizer_settings"]["lower_upper_bounds"]
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
        return estimation_settings["objective_function"](params, model, X, Y, T, ph_diagram, estimation_settings)

    # Use user's hyperopt specifications or use the default ones:
    algo = estimation_settings.get("hyperopt_algo", rand.suggest)
    max_evals = estimation_settings.get("hyperopt_max_evals", 500)
    timeout = estimation_settings["timeout"]

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
        rstate = np.random.default_rng(estimation_settings["optimizer_settings"]["hyperopt_seed"]),
        verbose=(verbosity >= 1),
        )
    params = list(best.values())

    return {"x": params, "fun": min(trials.losses()), "all_results": trials}

########################################################################################################
##                                          OBJECTIVE FUNCTIONS
########################################################################################################

## -------------------------------------- 1. ODE RMSE ----------------------------------------------------
def model_ode_error(params, model, X, Y, T, ph_diagram, estimation_settings):
    """Defines mean squared error of solution to differential equation as the error metric.

        Input:
        - T is column of times at which samples in X and Y happen.
        - X are columns of data (without potential target variables/columns).
        - Y are columns of data that correspond to target variables: features that are derived via ode fitting.
    """
    try:
        # simulate
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
            return ode(model, params, T, X, Y, **estimation_settings["objective_settings"])

        # Next line works only when sys.stdout is real. Thats why above.
        if isinstance(sys.stdout, stdout_type) and (estimation_settings["verbosity"] < 2):
            with open(os.devnull, 'w') as f, mt.stdout_redirected(f):
                try:
                    simX = run_ode()
                except Exception as error:
                    # Next error message should be theoretically impossible to witness, since all output is
                    # forwarded into fake output (black hole).
                    if estimation_settings["verbosity"] >= 1:
                        print("Inside ode(), preventing tee/IO error. Params at error:",
                              params, f"and {type(error)} with message:", error)
        else:
            simX = run_ode()
        if change_std2tee:
            sys.stdout = tee_object  # Change it back to fake stdout (tee).

        # b. calculate the objective (MSE of the fit)
        if estimation_settings["objective_settings"]["simulate_separately"]:
            res = np.sqrt(np.mean((Y - simX.reshape(-1))**2))
        else:
            res = np.sqrt(np.mean((X - simX)**2))

        # c. calculate the persistent_diagram of simulated trajectory
        if estimation_settings["objective_settings"]["persistent_homology"] and ph_diagram is not None:
            if estimation_settings["verbosity"] >= 2:
                print(f'iters vs. ph_used till now: {model.all_iters}, ph_iters:{model.ph_used}')
            weight = estimation_settings["objective_settings"]["persistent_homology_weight"]
            size = estimation_settings["objective_settings"]["persistent_homology_size"]
            if estimation_settings["objective_settings"]["simulate_separately"]:
                trajectory = np.vstack((X, simX))
            else:
                trajectory = simX
            try:
                persistent_homology_error = ph_error(trajectory, ph_diagram, size, estimation_settings["verbosity"])
                res = math.tan(math.atan(res) * weight + math.atan(persistent_homology_error) * (1-weight))
                model.ph_used += 1
            except Exception as error:
                if estimation_settings["verbosity"] >= 2:
                    print("\nError from Persistent Homology metric when calculating"
                          " bottleneck distance.\n", error)

        if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            if estimation_settings["verbosity"] > 1:
                print("Objective error is nan, inf or unreal. Returning default error.")
            return estimation_settings['default_error']

    except Exception as error:
        print("\nError within model_ode_error().\n", error)

    model.all_iters += 1
    return res


def ode(model, params, T, X, Y, **objective_settings):
    """Solve system of ODEs defined by equations in models_list.

    Raise error if input is incompatible.
        Input:
            model -- list (not dictionary) of models that e.g. generate_models() generates.
            params -- list of lists or ndarrays of parameters for corresponding models.
            X_data -- 2-dim array (matrix) i.e. X = [X[0,:], X[1,:],...].
            T -- (1-dim) array, i.e. of shape (N,)
            max_ode_steps -- maximal number of steps inside ODE solver to determine the minimal step size inside ODE solver.
        Outputs:
            Solution of ODE evaluated at times T.
    """

    # a. Check input
    if not (X.ndim == 2):
        raise TypeError("Function ode's defined error: Input arguments are not in required form! \n"
                        "data is a 2D matrix: {}, \n".format(str(X.ndim == 2)))
    elif not T.shape[0] == X.shape[0]:
        raise IndexError("Number of samples in time column and data matrix does not match.")

    # Update params
    model.set_params(params, split=True)

    # Set jacobian
    Jf = None
    if objective_settings["use_jacobian"]:
        J = model.lambdify_jacobian()
        def Jf(t, x):
            return J(*x)

    # Set model function
    if objective_settings["simulate_separately"]:
        # model_func = model.lambdify(list=True)[0]
        model_func = model.lambdify(list=True)
    else:
        model_func = model.lambdify(list=True)

    # Set initial values
    if Y is None:
        model.initials[model.obs_mask] = X[0]
        model.initials[~model.obs_mask] = params[model.total_eq_params:]
    else:
        model.obs_mask = np.full(Y.ndim, True, dtype=bool)
        model.initials = Y[0]

    # Interpolate data matrix
    # X_interp = interp1d(T, X, axi[s=0, kind='cubic', fill_value="extrapolate")
    X_interp = interp1d(T, X, axis=0, kind='cubic', fill_value="extrapolate") if X.shape[1] != 0 else (lambda t: np.array([]))

    # Set derivative functions either for or without teacher forcing
    if objective_settings["teacher_forcing"]:

        def dxdt(t, x):
            b = np.empty(len(model.sym_vars))
            b[~model.obs_mask] = x[~model.obs_mask]
            b[model.obs_mask] = X_interp(t)
            return [model_func[i](*b) for i in range(len(model_func))]

    else:
        if Y is None:           # hasattr(model.expr, '__len__')?
            def dxdt(t, x):
                if sp.symbols("t") in model.sym_vars:
                    return [model_func[i](*x, t) for i in range(len(model_func))]
                else:
                    return [model_func[i](*x) for i in range(len(model_func))]
        else:
            def dxdt(t, x):
                b = np.concatenate((x, X_interp(t)))
                return [model_func(*b)]

    # Simulate
    sim = odeint(dxdt, model.initials, T,
                rtol=objective_settings['rtol'], atol=objective_settings['atol'],
                Dfun=Jf, tfirst=True)

    # Return simulation only for the observed state variables
    if not objective_settings["simulate_separately"]:
        sim = sim[:, model.obs_mask]

    return sim

## -------------------------------------- 2. ALGEBRAIC MSE ----------------------------------------------------
def model_error(params, model, X, Y, _T=None, _ph_metric=None, estimation_settings=None):
    """Defines mean squared error as the error metric."""

    try:
        verbosity = estimation_settings['verbosity']

        testY = model.evaluate(X, *params)
        res = np.sqrt(np.mean((Y-testY)**2))

        if np.isnan(res) or np.isinf(res) or not np.isreal(res):
            if verbosity > 1:
                print("isnan, isinf, isreal =", np.isnan(res), np.isinf(res), not np.isreal(res))
                print(model.expr, model.params, model.sym_params, model.sym_vars)
            return estimation_settings['default_error']
        return res

    except Exception as error:
        if verbosity > 1:
            print("model_error: Params at error:", params, f"and {type(error)} with message:", error)
        if verbosity > 0:
            print(f"Program is returning default_error: {estimation_settings['default_error']}")
        return estimation_settings['default_error']

## ------------------------ 3. GENERAL - CHOOSES BETWEEN ODE OR ALGEBRAIC --------------------------
def model_error_general(params, model, X, Y, T, ph_diagram, **estimation_settings):
    """Calculate error of model with given parameters in general with type of error given.
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
        return model_ode_error(params, model, X, Y, T, ph_diagram,
                               estimation_settings)
    else:
        types_string = "\", \"".join(TASK_TYPES)
        raise ValueError("Variable task_type has unsupported value "
                f"\"{task_type}\", while list of possible values: "
                f"\"{types_string}\".")

## ------------------------ 4. PERSISTENT HOMOLOGY ERROR (for ODE)  --------------------------

def ph_error(trajectory: np.ndarray, diagram_truth: List[np.ndarray], size: int, verbosity: int) -> float:
    """Calculates persistent homology metric between given trajectory
    and ground truth trajectory based on topological properties of both.
    See ph_test.py in  examples/DS2022/persistent_homology.

    Inputs:
        - trajectory: of shape (many, few), i.e. many time points of few dimensions.
        - diagram_truth: persistence diagram of ED dataset, i.e. ground
        truth trajectory. To speed up costly computation of persistent
        diagram, we can calculate it once at the beginning of ED and
        then always reuse the already calculated one.
    Output:
        - float: bottleneck distance between the two diagrams
    """

    # for persistent homology:  # pip scikit-tda
    import persim

    # size = diagram_truth[0].shape[0]
    diagram = ph_diag(trajectory, size)
    if diagram[1].shape == (0, 2) and diagram_truth[1].shape == (0, 2):
        if verbosity >= 2:
            print("Both ground truth and candidate trajectory have trivial persistence diagram of dim 1")
        return 0
    # try:
    else:
        distance_bottleneck = persim.bottleneck(diagram[1], diagram_truth[1])
    # except IndexError(" index -1 is out of bounds for axis 0 with size 0") as error:
    #     distance_bottleneck = 0
    return distance_bottleneck

def ph_diag(trajectory: np.ndarray, size: int) -> List[np.ndarray]:
    """Returns persistent diagram of given trajectory. See ph_test.py in examples.

    Inputs:
        - trajectory: of shape (many, few), i.e. many time points of few dimensions.
        - size: Number of point clouds taken into the account when
        calculating persistent diagram. I.e. trajectory is
        down-sampled by averaging to get to desired number of time
        points. Rule of thumb of time complexity: 200 points ~ 0.02 seconds
    Output:
        - diagram [list of length 2]: as output of
        ripser.ripser(*trajectory*)['dgms']
    """

    # for persistent homology:  # pip scikit-tda
    import ripser

    def downsample(lorenz: np.ndarray) -> np.ndarray:
        m = int(lorenz.shape[0] / size)
        lorenz = lorenz[:(m * size), :]
        def aggregate(array: np.ndarray) -> np.ndarray:
            return array.reshape(-1, m).mean(axis=1)
        lor = np.apply_along_axis(aggregate, 0, lorenz)
        return lor

    P1 = downsample(trajectory) if size < trajectory.shape[0] else trajectory
    diagrams = ripser.ripser(P1)['dgms']
    return diagrams

