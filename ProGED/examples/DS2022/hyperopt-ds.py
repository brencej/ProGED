
import numpy as np
from hyperopt import hp, fmin, rand, pyll, Trials
import hyperopt.pyll.stochastic
from ProGED.examples.DS2022.hyperopt_obj import Estimation

# hyperparameters:
# - recombination (cr) [0, 1] or [0.5, 1]
# - mutation (f) [0, 2]
# - pop_size [50, 300]
# - maxiter [100, 15000]
space = [hp.uniform('hp_f', 0, 1),
         hp.uniform('hp_cr', 0, 2),
         # hp.quniform('hp_pop_size', 2, 3, 25),
         hp.quniform('hp_pop_size', 50, 300, 25),
         # hp.quniform('hp_max_iter', 4, 5)
         hp.qloguniform('hp_max_iter', np.log(100), np.log(15000), 100)
         ]

est = Estimation("lorenz")
print(est.models)

expr = est.models[0].full_expr()


def objective(params):
    print('\nMy params: ', params, '\n')
    res, t = est.fit(params)
    return res

# Use user's hyperopt specifications or use the default ones:
algo = rand.suggest
max_evals = 1000000000000000000
timeout = 1*60*60
# timeout = 130
# max_evals = 1
# timeout = 1

print('whoami')


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
print(result)


#
#
# [ 0.1638615  -0.80748112  0.06462776]
# 175.87690633927954
#   0%|          | 5/1000000000000000000 [02:36<8713837054040697:44:32, 31.37s/trial, best loss: 166.99465453229976]
# {'x': [1.5000006944654507, 0.35669813748640533], 'fun': 166.99465453229976}

#
# [ 0.1638615  -0.80748112  0.06462776]
# 175.87690633927954
#   0%|          | 5/1000000000000000000 [02:36<8713837054040697:44:32, 31.37s/trial, best loss: 166.99465453229976]
# {'x': [1.5000006944654507, 0.35669813748640533], 'fun': 166.99465453229976}


# Iter 176
# [0.31572701 0.80446935 0.00874652]
# 243.52571666994555
#   0%|          | 19/1000000000000000000 [10:10<8921890056621261:56:16, 32.12s/trial, best loss: 166.13451348103706]
# {'x': [1.1320020705987923, 0.23067128499491618], 'fun': 166.13451348103706}


