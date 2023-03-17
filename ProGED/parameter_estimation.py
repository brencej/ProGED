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

##
def fit_models(models, data=None, settings=settings, pool_map=map):

    check_inputs(settings)
    estimator = Estimator(data=data, settings=settings)
    fitted = list(pool_map(estimator.fit_one, models.values()))
    return pg.ModelBox(dict(zip(models.keys(), fitted)))

def check_inputs(settings):

    task_types = ["algebraic", "differential"]
    if settings['parameter_estimation']['task_type'] not in task_types:
        raise ValueError(f"The setting 'task_type' has unsupported value. Allowed options: {task_types}.")

    optimizers = ["DE", "DE_scipy", "hyperopt"]
    if settings['parameter_estimation']['optimizer'] not in optimizers:
        raise ValueError(f"The setting 'optimizer' has unsupported value. Allowed options: {optimizers}.")


### ESTIMATOR CLASS ###

class Estimator():

    def __init__(self, data=None, settings=settings):

        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            self.data = pd.read_csv(settings['parameter_estimation']['data'])

        self.settings = settings

        if self.settings['parameter_estimation']['task_type'] == 'algebraic':
            settings['parameter_estimation']['objective_function'] = objective_algebraic
        else:
            settings['parameter_estimation']['objective_function'] = objective_differential

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

        # if there is no parameters:
        # elif len(model.params) == 0:
        #     if
        #         model.set_estimated({"x": [],
        #                              "fun": objective_algebraic([], model, self)})
        #
        # else do parameter estimation
        else:
            model = self.check_observability(model)
            optmizers_dict = {"DE": DEwrapper, "hyperopt": "hyperopt_fit"}
            optimizer = optmizers_dict[settings["parameter_estimation"]["optimizer"]]
            t1 = time.time()
            result = optimizer(self, model)
            result["duration"] = time.time() - t1

            model.set_estimated(result)

        return model

    def check_observability(self, model):
        unobserved_vars = [str(item) for item in model.sym_vars if str(item) not in model.observed_vars]
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
                                  n_max_gen = estimator.settings["optimizer_DE"]["max_iter"])

    output = minimize(pymoo_problem,
                      algorithm,
                      termination,
                      seed=estimator.settings["experiment"]["seed"],
                      verbose=estimator.settings["optimizer_DE"]["verbose"],
                      save_history=estimator.settings["optimizer_DE"]["save_history"])

    model.set_params(output.X, with_initials=True)
    model.set_initials(output.X, dict(estimator.data.iloc[0, :]))

    return {"x": output.X, "fun": output.F[0], "complete_results": output}


class PymooProblem(Problem):

    def __init__(self, estimator, model):

        params = list(model.params.values())
        xl = [l[0] for l in estimator.settings['parameter_estimation']['param_bounds']]
        xu = [u[1] for u in estimator.settings['parameter_estimation']['param_bounds']]
        super().__init__(n_var=len(params), n_obj=1, n_constr=0, xl=xl, xu=xu)

        self.params = params
        self.model = model
        self.estimator = estimator
        self.best_f = estimator.settings['parameter_estimation']['default_error']
        self.opt_curve = []

    def _evaluate(self, x, out, *args, **kwargs):
        objective = self.estimator.settings["parameter_estimation"]["objective_function"]
        out["F"] = np.asarray([objective(
                                        x[i,:],
                                        self.model,
                                        self.estimator) for i in range(len(x))])
        self.best_f = np.min(out["F"])
        self.opt_curve.append(self.best_f)


class BestTermination(Termination):

    def __init__(self, min_f=1e-3, n_max_gen=500) -> None:
        super().__init__()
        self.min_f = min_f
        self.max_gen = MaximumGenerationTermination(n_max_gen)

    def _update(self, algorithm):
        terminate_on_const = algorithm.problem.estimator.settings["optimizer_DE"]["termination_after_nochange_iters"]
        if algorithm.problem.best_f < self.min_f:
            self.terminate()
        elif (len(algorithm.problem.opt_curve) > terminate_on_const+2) and (algorithm.problem.opt_curve[-1] == algorithm.problem.opt_curve[-terminate_on_const]):
            self.terminate()
        return self.max_gen.update(algorithm)


### OBJECTIVES ###

def objective_algebraic(params, model, estimator):

    model.set_params(params)
    lhs = [str(i) for i in model.lhs_vars]
    rhs = [str(i) for i in model.sym_vars]
    if lhs == rhs:
        raise ValueError("Left and right hand side variables are the same. This should not happen in algebraic models.")
    X = np.array(estimator.data[rhs])
    Y = np.array(estimator.data[lhs])

    Y_hat = model.evaluate(X)
    error = np.sqrt(np.mean((Y - Y_hat) ** 2))

    if np.isnan(error) or np.isinf(error) or not np.isreal(error):
        return estimator.settings['parameter_estimation']['default_error']
    else:
        return error

def objective_differential(params, model, estimator):

    model.set_params(params, with_initials=True)
    model.set_initials(params, dict(estimator.data.iloc[0, :]))

    X = np.array(estimator.data[[str(i) for i in model.observed_vars]])
    X_hat = simulate_ode(estimator, model)

    if estimator.settings['objective_function']["simulate_separately"]:
        error = np.sqrt(np.mean((X - X_hat.reshape(-1)) ** 2))
    else:
        error = np.sqrt(np.mean((X - X_hat) ** 2))

    if estimator.settings['objective_function']["persistent_homology"]:
        error = 1

    if np.isnan(error) or np.isinf(error) or not np.isreal(error):
        return estimator.settings['parameter_estimation']['default_error']
    else:
        return error


def simulate_ode(estimator, model):

    # Make model function
    model_function = model.lambdify(list=True, add_time=True)

    # Set jacobian
    Jf = None
    if estimator.settings['objective_function']["use_jacobian"]:
        J = model.lambdify_jacobian()
        def Jf(t, x):
            return J(*x)

    # Make function 'rhs' either with or without teacher forcing
    if estimator.settings['objective_function']["teacher_forcing"]:
        observability_mask = np.array([item in model.observed_vars for item in model.lhs_vars])
        X_interp = interp1d(estimator.data['t'], estimator.data.loc[:, estimator.data.columns != 't'],
                    axis=0, kind='cubic', fill_value="extrapolate") if estimator.data.shape[1] != 0 else (lambda t: np.array([]))
        def rhs(t, x):
            b = np.empty(len(model.sym_vars))
            b[observability_mask] = X_interp(t)
            b[~observability_mask] = x[~observability_mask]
            return [model_function[i](t, *b) for i in range(len(model_function))]
    else:
        def rhs(t, x):
            return [model_function[i](t, *x) for i in range(len(model_function))]

    # Simulate
    simulation = odeint(rhs,
                         list(model.initials.values()), # initial states
                         estimator.data['t'],           # time vector
                         rtol=estimator.settings['objective_function']['rtol'],
                         atol=estimator.settings['objective_function']['atol'],
                         Dfun=Jf,
                         tfirst=True)

    return simulation



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
