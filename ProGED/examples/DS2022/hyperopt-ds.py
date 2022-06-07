
import numpy as np
from hyperopt import hp, fmin, rand, pyll, Trials
import hyperopt.pyll.stochastic
from ProGED.examples.DS2022.hyperopt_obj import Estimation

# verbosity = estimation_settings["verbosity"]
# lu_bounds = estimation_settings["lower_upper_bounds"]
# lower_bound, upper_bound = lu_bounds[0] + 1e-30, lu_bounds[1] + 1e-30
#
# space_fn = estimation_settings.get("hyperopt_space_fn", hp.uniform)
# if space_fn not in {hp.randint, hp.uniform, hp.loguniform}:
#     # if verbosity >= 1:
#         print(
#             f"hyperopt_fit's warnings: "
#             f"Input estimation_settings[\"hyperopt_space_fn\"]={space_fn} "
#             f"should be used carefully, since it is not recognized as the"
#             f" member of the default configuration of the form"
#             f" space_fn('label', low, high).\n"
#             f"Therefore make sure the function is compatible with search"
#             f" space arguments ( hyperopt_space_(kw)args ).\n"
#             f"In doubt use one of:\n  - hp.randint\n  - hp.uniform\n"
#             f"  - hp.loguniform")
#
# # User can specify one dimensional search space, which is then replicated.
# args = estimation_settings.get("hyperopt_space_args", ())
# kwargs = estimation_settings.get("hyperopt_space_kwargs", {})
# if args != () or kwargs != {}:
#     space = [space_fn('C' + str(i), *args, **kwargs) for i in range(len(p0))]
# else:
#     space = [space_fn('C' + str(i), lower_bound, upper_bound) for i in range(len(p0))]


# hyperparameters:
# - recombination (cr) [0, 1] or [0.5, 1]
# - mutation (f) [0, 2]
# - pop_size [50, 300]
# - maxiter [100, 15000]
space = [hp.uniform('hp_f', 0, 1),
         hp.uniform('hp_cr', 0, 2),
         hp.randint('hp_pop_size', 50, 300),
         hp.randint('hp_max_iter', 100, 15000)
         ]
#          "f": hyperparams[0],
#               "cr": hyperparams[1],
# "pop_size": hyperparams[2],
# "max_iter": hyperparams[3],

est = Estimation("lorenz")
print(est.models)

# res, t = est.fit([0.5, 0.9, 3, 2])
# print(res, t)


def objective(params):
    # First way for solution:
    # params = [float(i) for i in params]  # Use float instead of np.int32.
    # return estimation_settings["objective_function"](
    #     params, model, X, Y, T, estimation_settings)
    res, t = est.fit(params)
    return res

# Use user's hyperopt specifications or use the default ones:
algo = rand.suggest
max_evals = 50
timeout = 100


# if verbosity >= 3:
#     print(f"Hyperopt will run with specs:\n"
#           f"  - search space:\n" + "".join([str(i) + "\n" for i in space])
#           # + f"  - algorithm: {algo}\n"
#           + f"  - timeout: {timeout}\n  - max_evals: {max_evals}")
#     print("A few points generated from the space specified:")
#     for i in range(10):
#         print(hyperopt.pyll.stochastic.sample(space))
#
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=algo,
    trials=trials,
    timeout=timeout,
    max_evals=max_evals,
    rstate=np.random,
    verbose=True,
)
params = list(best.values())
result = {"x": params, "fun": min(trials.losses())}
# if verbosity >= 3:
#     print(result)
# return result
print(result)
