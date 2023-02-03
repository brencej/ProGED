import matplotlib.pyplot as plt
import persim

from ph_test import traj2diag, plot3d, big_loada

if __name__ == "__main__":

    # size = 100000 # 4-8sec
    size = 1000  # 4-8sec
    size = 500  # 4-8sec
    size = 200  # 4-8sec

    # ###  Use case example:  ####
    # lorenzs = big_loada()
    # traj1 = lorenzs[0]
    # traj2 = lorenzs[1]
    # traj2 = lorenzs[2]
    #
    #
    # print("input trajectory:")
    # plot3d(traj1)
    # diagrams1, P1 = traj2diag(traj1, size)
    # print("downsampled trajectory:")
    # plot3d(P1)
    # print("diagram:")
    # persim.plot_diagrams(diagrams1, show=True)
    #
    # print("second trajectory:")
    # plot3d(traj2)
    # diagrams2, P2 = traj2diag(traj2, size)
    # print("downsampled second trajectory:")
    # plot3d(P2)
    # print("second diagram:")
    # persim.plot_diagrams(diagrams2, show=True)
    #
    # print("bottleneck distance diagram:")
    # distance_bottleneck, matching = persim.bottleneck(diagrams1[1], diagrams2[1], matching=True)
    # print("bottleneck distance = ", distance_bottleneck)
    # persim.bottleneck_matching(diagrams1[1], diagrams2[1], matching=matching, labels=['traj1', 'second trajectory'])
    # plt.show()
    #
    # print('end')
    # #### End of use case example:  ####

    import sys
    import os
    import pickle as pkl
    import pandas as pd
    import numpy as np
    import ProGED as pg
    from ProGED.model_box import ModelBox
    import itertools
    from src.generate_data.systems_collection import strogatz, mysystems
    from proged.helper_functions import get_fit_settings

    np.random.seed(1)

    def get_settings(iinput, systems, snrs, inits, set_obs):
        sys_names = list(systems.keys())
        combinations = []
        for sys_name in sys_names:
            combinations.append(list(itertools.product([sys_name], systems[sys_name].get_obs(set_obs), inits, snrs)))
        combinations = [item for sublist in combinations for item in sublist]
        return combinations[iinput-1]

    # iinput = int(sys.argv[1])
    iinput = 1
    systems = {**strogatz, **mysystems}
    exp_version = "e3_ph"
    data_version = "allong"
    structure_version = "s0"
    set_obs = "all"  # either full, part or all
    snrs = ["inf", 30, 20, 13, 10, 7]
    inits = np.arange(0, 4)
    data_length = 1000
    sys_name, iobs, iinit, snr = get_settings(iinput, systems, snrs, inits, set_obs)

    path_main = "D:\\Experiments\\MLJ23"
    path_base_out = f"{path_main}{os.sep}results{os.sep}proged{os.sep}parestim_sim{os.sep}{exp_version}{os.sep}"

    # ----- Get data (without der) -------
    path_data_in = f"{path_main}{os.sep}data{os.sep}{data_version}{os.sep}{sys_name}{os.sep}"
    data_filename = f"data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init{iinit}.csv"
    # data_orig = pd.read_csv(path_data_in + data_filename)
    path_main = "src/data/"
    data_orig = pd.read_csv(path_main + data_filename)

    # prepare data

    # a. mlj paper:
    data = np.array(pd.concat([data_orig.iloc[:, 0], data_orig[iobs]], axis=1))
    # b. lorenz2 tests ph:
    from utils.generate_data_ODE_systems import generate_ODE_data
    data = generate_ODE_data(system='lorenz', inits=[0.2, 0.8, 0.5])
    # c. lorenz1 tests ph
    generation_settings = {"simulation_time": 0.25}
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8], **generation_settings)
    data = data[:, (0, 1)]  # y, 2nd column, is not observed

    print('pure data')
    print(data[:10, :10])

    print(data.shape)

    traj1 = data[:, 1:]
    print(traj1.shape)
    print('spacial data')
    print(traj1[:10, :10])
    print("input trajectory:")
    plot3d(traj1, "first traj, spacial only")
    diagrams1, P1 = traj2diag(traj1, size)
    print("downsampled trajectory:")
    plot3d(P1, "downsampled")
    print("diagram:")
    persim.plot_diagrams(diagrams1, show=True, title="first traj")




