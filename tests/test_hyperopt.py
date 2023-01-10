import numpy as np
from hyperopt import hp
from ProGED.equation_discoverer import EqDisco


def test_equation_discoverer_hyperopt():
    B = -2.56; a = 0.4; ts = np.linspace(0.45, 0.87, 5)
    ys = (ts+B)*np.exp(a*ts); xs = np.exp(a*ts)
    data = np.hstack((ts.reshape(-1, 1), xs.reshape(-1, 1), ys.reshape(-1, 1)))

    np.random.seed(20)
    ED = EqDisco(data = data,
                    task = None,
                    task_type = "differential",
                    time_index = 0,
                    target_variable_index = -1,
                    variable_names=["t", "x", "y"],
                    sample_size = 2,
                    verbosity = 1,
                    estimation_settings={
                        "time_index": 0,
                        "optimizer": 'hyperopt',
                        "hyperopt_space_fn": hp.qnormal,
                        "hyperopt_space_args": (0.4, 0.5, 1/1000),
                        "hyperopt_max_evals": 100,
                        "optimizer_settings": {
                            "hyperopt_seed": 0
                        }
                    }
                    )
    ED.generate_models()
    ED.fit_models()

    def assert_line(models, i, expr, error, tol=1e-7, n=100):
        #assert str(ED.models[i].get_full_expr())[:n] == expr[:n]
        assert abs(ED.models[i].get_error() - error) < tol
    assert_line(ED.models, 0, "y", 3.564323422789496)
    assert_line(ED.models, 1, "-0.839*x + y", 3.221875534700383, n=6)
    # print(ED.models)
    return