import numpy as np

from ProGED.model_box import ModelBox
from ProGED.parameter_estimation import fit_models
from utils.generate_data_ODE_systems import generate_ODE_data

def test_persistent_homology_partial_observability():
    """Not the representanive case, more like error case, to check against errors."""

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
    print(f"All iters (as saved to system_model object): {system_out[0].all_iters}")
    print(f"Iters when PH was used: {system_out[0].ph_used}")
    print(f"Iters when zero vs zero: {system_out[0].zerovszero}")
    # print(abs(system_out[0].get_error()))
    assert abs(system_out[0].get_error()) < 1  # 3.2.2023
    # true params: [[1.], [-0.5., -1., 0.5]]

def test_persistent_homology_ODE_system():
    """The representative case, here we should see the beauty of this metric."""
    data = generate_ODE_data(system='lorenz', inits=[0.2, 0.8, 0.5])

    system = ModelBox(observed=["x", "y", "z"])
    system.add_system(["C*(y-x)", "x*(C-z) - y", "x*y - C*z"], symbols={"x": ["x", "y", "z"], "const": "C"})
    estimation_settings = {"target_variable_index": None,
                           "time_index": 0,
                           "optimizer": 'DE_scipy',
                           "optimizer_settings": {"max_iter": 1,
                                                  "pop_size": 1,
                                                  "lower_upper_bounds": (-28, 28),
                                                  },
                           "objective_settings": {"use_jacobian": False,
                                                  "persistent_homology": True,
                                                  },
                           "verbosity": 2,
                           }

    np.random.seed(0)
    system_out = fit_models(system, data, task_type='differential', estimation_settings=estimation_settings)
    print(f"All iters (as saved to system_model object): {system_out[0].all_iters}")
    print(f"Iters when PH was used: {system_out[0].ph_used}")
    print(f"Iters when zero vs zero: {system_out[0].zerovszero}")
    # print(abs(system_out[0].get_error()))
    assert abs(system_out[0].get_error()) < 1.0  # 3.2.2023


if __name__ == "__main__":

    # test_persistent_homology_partial_observability()
    test_persistent_homology_ODE_system()
