import os
import sys
import time
import math
import numpy as np
import sympy as sp
import pandas as pd

from scipy.optimize import differential_evolution, minimize
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp, odeint

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.termination import Termination
from pymoo.termination.default import MaximumGenerationTermination

import ProGED as pg
from ProGED.model_box import ModelBox
from ProGED.configs import settings
from ProGED.external.persistent_homology import ph_init, ph_after

## FIRST FUNCTION
def fit_models(models, data=None, settings=settings, pool_map=map):
    """
    Main function that accepts the models, checks the settings and fits them one by one (parallelization possible).

    Arguments:
        models    (ProGED ModelBox)   a dictionary of Model objects. See ModelBox and Model
        data      (pandas DataFrame)  each column should have a variable name and represent data of that variable
        settings  (dict)              settings for the fitting
        pool_map  (function)          a function that allows to process (fit) the models in parrallel.
                                        Default is a map() function from Pool library.
    Returns:
        ModelBox of fitted models

    """

    check_inputs(settings)
    estimator = Estimator(data=data, settings=settings)
    fitted = list(pool_map(estimator.fit_one, models.values()))
    return pg.ModelBox(dict(zip(models.keys(), fitted)))

def check_inputs(settings):
    """Checks if correct settings were set. TODO: Should be extended."""

    task_types = ["algebraic", "differential"]
    if settings['parameter_estimation']['task_type'] not in task_types:
        raise ValueError(f"The setting 'task_type' has unsupported value. Allowed options: {task_types}.")

    optimizers = ["DE", "DE_scipy", "hyperopt"]
    if settings['parameter_estimation']['optimizer'] not in optimizers:
        raise ValueError(f"The setting 'optimizer' has unsupported value. Allowed options: {optimizers}.")


## ESTIMATOR CLASS ##
class Estimator():
    """
    Estimator class contains all the relevant information needed for fitting, except the model. It is created
    to ease the parallelization process. Also does some validity checks.

    Arguments:
        data      (pandas DataFrame)  each column should have a variable name and represent data of that variable.
                                      If data is not present, it can be alternatively loaded, if path is provided
                                      in the settings dictionary.
        settings  (dict)              Settings for the fitting. See configs.py for a detailed explanation.
    Methods:
        fit_one:                fits one model.
        check observability:    checks if data are only partial. If so, adds additional parameters (initial states) to
                                the vector of unknown parameters that will be optimized.

    """

    def __init__(self, data=None, settings=settings):

        # set data, either from DataFrame or, alternatively, load it if path is provided
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.read_csv(settings['parameter_estimation']['data'])

        # set settings
        self.settings = settings

        # set objective function based on settings and task type
        if self.settings['parameter_estimation']['task_type'] == 'algebraic':
            settings['parameter_estimation']['objective_function'] = objective_algebraic
        else:
            settings['parameter_estimation']['objective_function'] = objective_differential

            # if task is differential, check if the time is included in the data.
            if 't' not in self.data.columns:
                if 'time' in self.data.columns:
                    self.data.rename(columns={"time": "t"})
                else:
                    raise ValueError("Differential task needs to include the time in one of the columns, named"
                                     "either 't' or 'time'.")

    def fit_one(self, model):

        # check number of parameters
        # if there is too many parameters, skip:
        if len(model.params) > self.settings["parameter_estimation"]["max_constants"]:
            print("Model skipped. More model parameters than allowed (check max constant).")
            pass

        #if there is no parameters:
        elif len(model.params) == 0:
            model.set_estimated({"x": [],
                                 "fun": directly_calculate_objective([], model, self)})

        #else do parameter estimation
        else:
            model = self.check_observability(model)
            if self.settings["objective_function"]["persistent_homology"]:
                ph_init(self, model)
            optmizers_dict = {"DE": DEwrapper, "hyperopt": "hyperopt_fit"}
            optimizer = optmizers_dict[settings["parameter_estimation"]["optimizer"]]
            t1 = time.time()
            if settings["parameter_estimation"]["simulate_separately"]:
                result = {'x': [], 'fun': 0}
                model_splits = model.split()
                for model_split in model_splits:
                    result_split = optimizer(self, model_split)
                    result['x'].append(result_split['x'][0])
                    result['fun'] += result_split['fun']
            else:
                result = optimizer(self, model)

            result["duration"] = time.time() - t1
            model.set_estimated(result)

        return model

    def check_observability(self, model):
        if self.settings["parameter_estimation"]["observed_vars"] and model.observed_vars != self.settings["parameter_estimation"]["observed_vars"]:
            raise ValueError("Observed variables in the model (model.observed_vars) do not match the observed variables "
                             "in parameter estimation settings. Correct accordingly.")

        unobserved_vars = [str(item) for item in model.rhs_vars if str(item) not in model.observed_vars]
        if unobserved_vars:
            model.unobserved_vars = unobserved_vars
            for ivar in unobserved_vars:
                if ivar in list(model.initials.keys()):
                    model.params[ivar] = model.initials[ivar]
                else:
                    raise ValueError("Unobserved variables need to have inital state. Otherwise not possible to solve.")
        return model


### OPTIMIZER ###

def DEwrapper(estimator, model):
    """
    Function that wrappes the Pymoo's differential evolution function (DE). It sets up the Pymoo's Problem object,
    an algorithm DE, the Termination object. Then, it optimizes the unknown parameters using minimize function.

    Returns:
        result (dict): Results of parameter estimation. Includes:
                            "x": solution of optimization, i.e. optimal parameter values (list of floats)
                            "fun": value of optimization function, i.e. error of model (float).
                            "complete_results": complete output of optimization algorithm
    """

    pymoo_problem = PymooProblem(estimator, model)

    algorithm = DE(
        pop_size=estimator.settings["optimizer_DE"]["pop_size"],
        variant=estimator.settings["optimizer_DE"]["strategy"],
        CR=estimator.settings["optimizer_DE"]["cr"],
        F=estimator.settings["optimizer_DE"]["mutation"],
        sampling=LHS(),
        dither="vector",
        jitter=False
    )

    termination = BestTermination(min_f = estimator.settings["optimizer_DE"]["termination_threshold_error"],
                                  n_max_gen = estimator.settings["optimizer_DE"]["max_iter"],
                                  terminate_if_no_change=estimator.settings["optimizer_DE"]["termination_after_nochange_iters"],
                                  terminate_if_no_change_tolerance=estimator.settings["optimizer_DE"]["termination_after_nochange_tolerance"],
                                  )

    output = minimize(pymoo_problem,
                      algorithm,
                      termination,
                      seed=estimator.settings["experiment"]["seed"],
                      verbose=estimator.settings["optimizer_DE"]["verbose"],
                      save_history=estimator.settings["optimizer_DE"]["save_history"])

    model.set_params(output.X, extra_params=True)
    model.set_initials(output.X, dict(estimator.data.iloc[0, :]))

    return {"x": output.X, "fun": output.F[0], "complete_results": output}


class PymooProblem(Problem):
    """Implements a method evaluating a set of solutions
        Arguments:
            params (list of floats): unknown parameter values that are optimized over iterations.
            model  (ProGED object Model): see class Model in Model.py
            estimator (object Estimator): see class Estimator above
            best_f  (float): current best error (is updated over iterations)
            optimization_curve (list): saves the best error over iterations (this is manually added, as pymoo's history
                                        is very heavy)
            xl (list of floats): lower bounds of each of the parameter
            xu (list of floats): upper bounds of each of the parameter

        Methods:
            evaluate: runs objective function
    """

    def __init__(self, estimator, model):

        params = list(model.params.values())
        xl = [l[0] for l in estimator.settings['parameter_estimation']['param_bounds']]
        xu = [u[1] for u in estimator.settings['parameter_estimation']['param_bounds']]
        super().__init__(n_var=len(params), n_obj=1, n_constr=0, xl=xl, xu=xu)

        self.params = params
        self.model = model
        self.estimator = estimator
        self.best_f = estimator.settings['parameter_estimation']['default_error']
        self.optimization_curve = []

    def _evaluate(self, x, out, *args, **kwargs):
        objective = self.estimator.settings["parameter_estimation"]["objective_function"]
        out["F"] = np.asarray([objective(
                                        x[i,:],
                                        self.model,
                                        self.estimator) for i in range(len(x))])
        self.best_f = np.min(out["F"])
        self.optimization_curve.append(self.best_f)


class BestTermination(Termination):
    """Terminates optimization under certain conditions.
        Arguments:
            min_f (float):              if the objective functions reaches this error (min_f), optimization stops.
                                        In the settings, min_f is set as "termination_threshold_error".
            max_gen (int):              Maximum number of generations to be created. After that, optimization stops.
                                        In the settings, max_gen is set as "max_iter".
            terminate_if_no_change (int): Maximum number of iterations without the meaningful change in min_f. After that,
                                        optimization stops. In the settings, terminate_if_no_change is set
                                        as "termination_after_nochange_iters". By meaningful change we mean change
                                         outside the epsilon-neighbourhood of 0, i.e. absolute change being bigger than
                                         certain tolerance specified by `terminate_if_no_change_tolerance`.
            terminate_if_no_change_tolerance (float): Relative tolerance that ignores small changes. It is used in
                                        conjunction with the `terminate_if_no_change` variable (see above).

        Methods:
            update: checks at every iteration if termination critera are met
    """
    def __init__(self, min_f=1e-3, n_max_gen=500, terminate_if_no_change=200, terminate_if_no_change_tolerance=10 ** (-6)) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)
        self.terminate_if_no_change = terminate_if_no_change
        self.terminate_if_no_change_tolerance = terminate_if_no_change_tolerance

    def _update(self, algorithm):
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        elif (len(algorithm.problem.optimization_curve) > self.terminate_if_no_change + 2) and \
             abs(algorithm.problem.optimization_curve[-1] -
                 algorithm.problem.optimization_curve[-self.terminate_if_no_change]) \
                < self.terminate_if_no_change_tolerance:
            self.terminate()
        return self.max_gen.update(algorithm)


### OBJECTIVE FUNCTIONS ###

def objective_algebraic(params, model, estimator):
    """
    Objective function for algebraic models.

    Returns:
        error (float):  if successfull, returns the root-mean-square error between the true trajectories (X) and
                            simulated trajectories (X_hat), else it returns the dummy error (10**9).
    """

    # set the newely estimated parameters
    model.set_params(params)

    # get appropriate data points
    rhs_vars = [str(i) for i in model.rhs_vars]
    if model.lhs_vars == rhs_vars:
        raise ValueError("Left and right hand side variables are the same. This should not happen in algebraic models.")
    X = np.array(estimator.data[rhs_vars])
    Y = np.array(estimator.data[model.lhs_vars])

    # estimate the model and calculate the error
    Y_hat = model.evaluate(X)
    error = np.sqrt(np.mean((Y - Y_hat) ** 2))

    if np.isnan(error) or np.isinf(error) or not np.isreal(error):
        return estimator.settings['parameter_estimation']['default_error']
    else:
        return error

def objective_differential(params, model, estimator):
    """
    Objective function for differential models. Simulated trajectories are calculated in the function 'simulate_ode'.

    Returns:
        error (float):  if successfull, returns the root-mean-square error between the true trajectories (X) and
                            simulated trajectories (X_hat), else it returns the dummy error (10**9).
    """

    # set the newly estimated parameters and initial states
    model.set_params(params, extra_params=True)
    model.set_initials(params, dict(estimator.data.iloc[0, :]))

    # get appropriate data points
    X = np.array(estimator.data[[str(i) for i in model.observed_vars if i in model.lhs_vars]])

    # estimate the model
    X_hat = simulate_ode(estimator, model)

    # if successful, calculate the error and optionally extend error with persistent homology calculation.
    # if not successful or error is not float, return dummy error.
    if X_hat is not None:
        error = np.sqrt(np.mean((X - X_hat) ** 2))

        if estimator.settings['objective_function']["persistent_homology"]:
            error = ph_after(estimator, model, error, X_hat)

        if np.isnan(error) or np.isinf(error) or not np.isreal(error):
            return estimator.settings['parameter_estimation']['default_error']
        else:
            return error
    else:
        return estimator.settings['parameter_estimation']['default_error']


def simulate_ode(estimator, model):

    # Make model function
    model_function = model.lambdify(add_time=True, list=True)

    # Interpolate the data of extra variables
    X_extras = interp1d(estimator.data['t'], estimator.data[model.extra_vars], axis=0, kind='cubic',
                        fill_value="extrapolate") if model.extra_vars != [] else (lambda t: np.array([]))

    # Set jacobian
    Jf = None
    if estimator.settings['objective_function']["use_jacobian"]:
        J = model.lambdify_jacobian()
        def Jf(t, x):
            return J(*x)

    # Make function 'rhs' either with or without teacher forcing
    observability_mask = np.array([item in model.observed_vars for item in model.lhs_vars])

    if estimator.settings['objective_function']["teacher_forcing"]:
        X_interp = interp1d(estimator.data['t'], estimator.data.loc[:, estimator.data.columns != 't'],
                            axis=0, kind='cubic', fill_value="extrapolate") if estimator.data.shape[1] != 0 else (lambda t: np.array([]))

        def rhs(t, x):
            b = np.empty(len(model.rhs_vars))
            b[observability_mask] = X_interp(t)
            b[~observability_mask] = x[~observability_mask]
            return [model_function[i](t, *b) for i in range(len(model_function))]
    else:
        def rhs(t, x):
            b = np.concatenate((x, X_extras(t)))
            return [model_function[i](t, *b) for i in range(len(model_function))]

    # Simulate
    simulation, full_output = odeint(rhs,
                                     list(model.initials.values()),  # initial states
                                     estimator.data['t'],            # time vector
                                     rtol=estimator.settings['objective_function']['rtol'],
                                     atol=estimator.settings['objective_function']['atol'],
                                     Dfun=Jf,
                                     tfirst=True,
                                     full_output=True)

    # Return only trajectories of observed variables, if simulation is successful, else return dummy error
    if 'successful' in full_output['message']:
        return simulation[:, observability_mask]
    else:
        return None

def directly_calculate_objective(params, model, estimator):
    """
    Directly calculates objective function. It is only possible if no parameters need to be estimated.

    Returns:
        error (float):  if successful, returns the root-mean-square error between the true data and
                            estimated model data, else it returns the dummy error (10**9).
    """
    if estimator.settings['parameter_estimation']['task_type'] in ("algebraic", "integer-algebraic"):
        return objective_algebraic(params, model, estimator)
    elif estimator.settings['parameter_estimation']['task_type'] == "differential":
        return objective_differential(params, model, estimator)


##
if __name__ == '__main__':
    print("--- more parameter_estimation.py examples are included in the folder 'tests' ---")

    # test algebraic model with one equation
    X = np.linspace(-1, 1, 5).reshape(-1, 1)
    Y = 2.0 * (X + 0.3)
    data = pd.DataFrame(np.hstack((X, Y)), columns=['x', 'y'])

    models = pg.ModelBox()
    models.add_model("C*(x+C)",
                     symbols={"x": ["x", "y"], "const": "C"},
                     lhs_vars=['y'])

    settings["parameter_estimation"]["task_type"] = 'algebraic'

    models_fitted0 = pg.fit_models(models, data, settings=settings)
    print("models_fitted0 error: " + str(models_fitted0[0].get_error()))
    print("--- parameter_estimation.py test algebraic model with one equation successfully finished ---")


    # differential model with two equations and two variables
    t = np.arange(0, 1, 0.1)
    x = 3*np.exp(-2*t)
    y = 5.1*np.exp(-1*t)
    data = pd.DataFrame(np.vstack((t, x, y)).T, columns=['t', 'x', 'y'])

    models = pg.ModelBox()
    models.add_model(["C*x", "C*y"],
                     symbols={"x": ["x", "y"], "const": "C"})

    settings["parameter_estimation"]["task_type"] = 'differential'

    models_fitted3 = pg.fit_models(models, data, settings=settings)
    print("models_fitted3 error: " + str(models_fitted3[0].get_error()))
    print("--- parameter_estimation.py test differential model with two equations finished ---")
