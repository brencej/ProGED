import sys
import os
import time
import pickle
import ProGED as pg
from ProGED.examples.DS2022.generate_data_ODE_systems import generate_ODE_data
# sys.path.append(os.getcwd()+"/source")

if __name__ == '__main__':

    #batch_idx = sys.argv[1]
    batch_idx = '1'
    job_version = '1'
    system = 'VDP'
    inits = [-0.2, -0.8]
    models_path = os.path.join("jobs", "{}".format(system), "v{}".format(job_version))
    models_file = "job_{}_v{}_batch{}.pg".format(system, job_version, batch_idx)

    # 1. get data
    data = generate_ODE_data(system=system, inits=inits)

    # 2. param estimation settings
    optimizer_settings = {
        "lower_upper_bounds": (-10, 10),
        "default_error": 10 ** 9,
        "strategy": 'rand1bin',
        "f": 0.45,
        "cr": 0.88,
        "max_iter": 1000,
        "pop_size": 100,
        "atol": 0.001,
        "tol": 0.001
    }

    estimation_settings = {"optimizer": 'differential_evolution',
                           "observed": ["x", "y"],
                           "optimizer_settings": optimizer_settings}

    # 3. estimate parameters
    start_time = time.time()

    print("--Loading models")
    with open(os.path.join(models_path, models_file), "rb") as file:
        models = pickle.load(file)

    print("--Fitting models")
    models_out = pg.fit_models(models,
                               data=data,
                               task_type='differential',
                               time_index=0,
                               estimation_settings=estimation_settings)

    print("--Exporting results")
    with open(models_path + os.sep + models_file.split(".")[0] + "_fit.pg", "wb") as file:
        pickle.dump(models, file)

    print("--End time in seconds: " + str(time.time() - start_time))

