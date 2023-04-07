## ------------------------ 4. PERSISTENT HOMOLOGY ERROR (for ODE)  --------------------------

import math
import numpy as np
import persim
# for persistent homology:  # pip scikit-tda
import ripser
from typing import List

def ph_error(trajectory: np.ndarray, diagrams_truth: List[np.ndarray], diagram_dimension: int,
             size: int, verbosity: int, model) -> float:
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

    # size = diagram_truth[0].shape[0]
    diagrams = ph_diag(trajectory, size)
    if diagrams_truth[diagram_dimension].shape == (0, 2) and diagrams[diagram_dimension].shape == (0, 2):

        # for debugging add this code into model.py, __init__
        # # number of successful persistent homology comparisons vs. pure rmse (should be limited to ODEs?)
        # self.ph_all_iters = 0
        # self.ph_used = 0
        # self.ph_zerovszero = 0

        # model.ph_zerovszero += 1
        if verbosity >= 2:
            print(f"Both ground truth and candidate trajectory have trivial persistence diagram "
                  f"of dim {diagram_dimension}.")
        return 0
    # try:
    else:
        distance_bottleneck = persim.bottleneck(diagrams_truth[diagram_dimension], diagrams[diagram_dimension])
        if verbosity >= 3:
            try:
                import matplotlib.pyplot as plt
                persim.plot_diagrams(diagrams, show=True, title=f"candidate: {model.full_expr()}")
                distance_bottleneck, matching = persim.bottleneck(diagrams_truth[diagram_dimension],
                                                                  diagrams[diagram_dimension], matching=True)
                print(f'bottleneck distance: {distance_bottleneck}')
                plt.close()
                persim.bottleneck_matching(diagrams_truth[diagram_dimension],
                                           diagrams[diagram_dimension],
                                           # matching,
                                           matching=matching,
                                           labels=['orignal; bottleneck distance:', f'candidate; {distance_bottleneck:.3f}', ],
                                           )
                plt.show()
            except Exception as error:
                print(f"Error when PLOTTING of type {type(error)} and message:{error}!")
            print("Both ground truth and candidate trajectory have trivial persistence diagram of dim 1")
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


def ph_init(estimator, model):
    """Calculate the persistent_diagram of ground_truth data in initialization of parameter_estimation."""

    # take care of persistent homology case, i.e. if using topological distance
    weight = estimator.settings["objective_function"]["persistent_homology_weight"]
    if (not isinstance(weight, (float, int, np.float64))) or weight < 0 or weight > 1:
        raise TypeError("ERROR: Persistent homology weight should be of type float and in range [0,1]!")
    size = estimator.settings["objective_function"]["persistent_homology_size"]
    verbosity = estimator.settings["experiment"]["verbosity"]

    # # X = np.array(estimator.data[[str(i) for i in model.observed_vars]])
    # X = np.array(estimator.data[[str(i) for i in model.observed_vars if i in model.lhs_vars]])
    # trajectory = np.vstack((X, estimator.data[model.extra_vars])) if model.extra_vars != [] else X  # ugly, b/c = observed
    trajectory = np.array(estimator.data[[str(i) for i in model.observed_vars]])  # seems good
    # trajectory = np.vstack(np.vstack((self.X, self.Y))) if self.Y is not None else self.X

    try:
        persistent_diagrams = ph_diag(trajectory, size)
        if persistent_diagrams[1].shape == (0, 2) or trajectory.shape[1] == 1:
            if verbosity >= 1:
                print("INFO: Dimensionality of the trajectory is trivially 1 or the "
                      "persistent diagram (of 2D?) of the ground truth is trivially empty, "
                      "i.e. no interesting 2D-property is present. Therefore, I will use diagrams regarding "
                      "persistent homology dimension 0, i.e. H_0 homology group.")
                # "and marbuse maximum size, since its fast.")
            estimator.persistent_diagrams = (persistent_diagrams, 0)
        else:
            estimator.persistent_diagrams = (persistent_diagrams, 1)
        if verbosity >= 3:
            print("verbosity is forcing plotting:")
            try:
                import matplotlib.pyplot as plt

                dim = estimator.persistent_diagrams[1]
                persim.plot_diagrams(persistent_diagrams[dim], show=True, title="init")
            except Exception as error:
                print(f"Error when PLOTTING of type {type(error)} and message:{error}!")
    except Exception as error:
        if verbosity >= 1:
            print(f"WARNING: Excepted an error when constructing ph_diagram of the original dataset "
                  f"of type {type(error)} and message:{error}!")
        estimator.persistent_diagrams = "ph Failed"

def ph_after(estimator, model, error, X_hat):
    """Calculate the persistent_diagram of simulated trajectory in
    the initialization of parameter_estimation."""

    # c. calculate the persistent_diagram of simulated trajectory
    # if estimation_settings["objective_settings"]["persistent_homology"] and ph_diagrams is not None:
    ph_diagrams = estimator.persistent_diagrams
    if ph_diagrams == "ph Failed":
        return error

    verbosity = estimator.settings["experiment"]["verbosity"]
    weight = estimator.settings["objective_function"]["persistent_homology_weight"]
    size = estimator.settings["objective_function"]["persistent_homology_size"]
    # if verbosity >= 2:
    #     print(f'iters vs. ph_used till now: {model.ph_all_iters}, ph_iters:{model.ph_used}')

    trajectory = np.vstack((X_hat, estimator.data[model.extra_vars])) if model.extra_vars != [] else X_hat  # seems good
    try:
        truth_diags = ph_diagrams[0]
        dim = ph_diagrams[1]
        persistent_homology_error = ph_error(trajectory, truth_diags, dim, size, verbosity,
                                             model)
        error = math.tan(math.atan(error) * weight + math.atan(persistent_homology_error) * (1 - weight))
        # model.ph_used += 1
    except Exception as err:
        if verbosity >= 2:
            print("\nError from Persistent Homology metric when calculating"
                  " bottleneck distance.\n", err)

    return error
