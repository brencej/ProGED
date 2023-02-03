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
# data = np.array(pd.concat([data_orig.iloc[:, 0], data_orig[iobs]], axis=1))

iobs_name = ''.join(iobs)
print(f"{sys_name} | snr: {snr} | obs: {iobs_name} | init: {iinit}")


cols_dx = ['x', 'y', 'dx']
cols_dy = ['x', 'y', 'dy']

datas = []
for choice in [cols_dx, cols_dy]:
    x = np.array(data_orig[choice[:-1]])
    y = np.array(data_orig[choice[-1]])
    datas += [(x, y)]

# fit

# a) fit ete

import time
from ProGED.examples.DS2022.trans.mlj import ete
start = time.perf_counter()

for x, y in datas:
    eq = ete(x, y)
    print(eq)
    end = time.perf_counter()
    print(f'ete needed {end-start} seconds, i.e. {(end-start)/60} minutes.')
    start = end



# b) fit proged

# estimation_settings = get_fit_settings(obs=iobs)
# estimation_settings["optimizer_settings"]["lower_upper_bounds"] = systems[sys_name].bounds
# estimation_settings["verbosity"] = 2
#
# systemBox = pg.ModelBox(observed=iobs)
# systemBox = ModelBox(observed=iobs)
# systemBox.add_system(systems[sys_name].sym_structure, symbols={"x": systems[sys_name].sys_vars, "const": "C"})
# print(estimation_settings)
# 1/0
# systemBox_fitted = pg.fit_models(systemBox, data, task_type='differential', estimation_settings=estimation_settings)
# systemBox_fitted.observed = iobs
# print(len(systemBox_fitted))
# print(systemBox_fitted[0].get_error())
# print(systemBox_fitted[0].all_iters)
# print(systemBox_fitted[0].ph_used)
#
# # save the fitted models and the settings file
# path_out = f"{path_base_out}{sys_name}{os.sep}"
# # os.makedirs(path_out, exist_ok=True)
# out_filename = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}" \
#                f"_snr{snr}_init{iinit}_obs{iobs_name}_fitted.pg"
# # systemBox_fitted.dump(path_out + out_filename)
#
# # # save settings of this exp
# # if iinput == 1:
# #
# #     settings_filename = f"estimation_settings_{data_version}_{structure_version}_{exp_version}"
# #
# #     with open(path_base_out + settings_filename + ".pkl", "wb") as set_file:
# #         pkl.dump(estimation_settings, set_file)
# #
# #     fo = open(path_base_out + settings_filename + ".txt", "w")
# #     for k, v in estimation_settings.items():
# #         fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
# #     fo.close()
# #
# #     systems_filename = f"systems_settings_{data_version}_{structure_version}_{exp_version}.txt"
# #     fo = open(path_base_out + systems_filename, "w")
# #     for k, v in systems.items():
# #         fo.write(str(k) + ' >>> ' + str(systems[str(k)].__dict__) + '\n\n')
# #     fo.close()
