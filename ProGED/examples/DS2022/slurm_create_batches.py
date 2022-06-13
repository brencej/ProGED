import os
import io
import pickle
import pandas as pd
import numpy as np
import ProGED as pg
from ProGED.generate import generate_models
from ProGED.examples.DS2022.generate_data_ODE_systems import generate_ODE_data
from ProGED.generators.grammar_construction import construct_production

def create_sh_file(**batch_settings):
    sys_name = batch_settings["system"]
    job_vers = batch_settings["job_version"]
    nbatches = batch_settings["n_batches"]
    pyfile_name = "slurm_run_batch_{}_v{}.py".format(sys_name, job_vers)

    title = "slurm_run_batch_{}_v{}.sh".format(sys_name, job_vers)
    f = io.open(os.path.join(batch_settings["path_main"], title), "w", newline='\n')
    f.write("#!/bin/bash\n"
            "#SBATCH --job-name={}v{}\n".format(sys_name, job_vers))
    f.write("#SBATCH --time=2-00:00:00\n"
            "#SBATCH --mem-per-cpu=2GB\n")
    f.write("#SBATCH --array=0-{}\n".format(str(nbatches-1)))
    f.write("#SBATCH --cpus-per-task=1\n")
    f.write("#SBATCH --output=jobs/{}/v{}/nbatches{}/slurm_output_%A_%a.out\n".format(sys_name, job_vers, str(nbatches)))
    f.write("\nsingularity exec proged_container.sif python3.7 " + pyfile_name + " ${SLURM_ARRAY_TASK_ID}")
    f.close()


def create_batches(**batch_settings):

    # generate grammar
    if batch_settings["grammar"] == "polynomial":
        grammar = pg.grammar_from_template("polynomial",
                                           generator_settings={"variables": batch_settings["variables"],
                                                               "p_vars": batch_settings["p_vars"], "functions": [],
                                                               "p_F": []})
    else:
        grammarstr = construct_production("E", ["E '+' F", "E '-' F", "F"], [0.15, 0.15, 0.7])
        grammarstr += construct_production("F", ["F '*' T", "T"], [0.2, 0.8])
        grammarstr += construct_production("T", ["'(' E ')'", "V", "'C'"], [0.2, 0.55, 0.25])
        grammarstr += construct_production("V", ["'x'", "'y'", "'z'"], [1 / 3, 1 / 3, 1 / 3])
        grammar = pg.GeneratorGrammar(grammarstr)

    symbols = {"x": batch_settings["variables"], "const": "C"}
    # generate models from grammar
    np.random.seed(0)
    models = generate_models(grammar,
                             symbols,
                             strategy_settings={"N": batch_settings["num_samples"],
                                                "max_repeat": 100},
                             dimension=len(batch_settings["variables"]))

    model_batches = models.split(n_batches=batch_settings["n_batches"])

    # save batches
    path_jobs = os.path.join(batch_settings["path_main"],
                                 "jobs",
                                 "{}".format(batch_settings["system"]),
                                 "v{}".format(batch_settings["job_version"]),
                                 "nbatches{}".format(str(batch_settings["n_batches"])))
    os.makedirs(path_jobs, exist_ok=True)

    for ib in range(batch_settings["n_batches"]):
        file_name = os.path.join(path_jobs, "job_{}_v{}_batch{}.pg".format(batch_settings["system"], batch_settings["job_version"], str(ib)))
        with open(file_name, "wb") as file:
            pickle.dump(model_batches[ib], file)

    # do manually
    if batch_settings["manual"]:
        symbols = {"x": ["x", "y"], "const": "C"}
        models = pg.ModelBox()
        models.add_system(["C*y", "C*y + C*x*x*y + C*x"], symbols=symbols)
        models.add_system(["C*x", "C*y"], symbols=symbols)
        file_name = os.path.join(path_jobs, "job_{}_v{}_batchM.pg".format(batch_settings["system"], batch_settings["job_version"]))
        with open(file_name, "wb") as file:
            pickle.dump(models, file)


    # create shell file
    create_sh_file(**batch_settings)

if __name__ == '__main__':

    # settings
    batch_settings = {
        "system": 'lorenz_stable',
        "job_version": '14',
        "variables": ["'x'", "'y'", "'z'"],
        "grammar": "custom",
        "p_vars": [1 / 3, 1 / 3, 1 / 3],
        "num_samples": 10000,
        "n_batches": 1,
        "path_main": os.path.join("D:\\", "Experiments", "DS2022"),
        "manual": False
    }

    create_batches(**batch_settings)

    # create data
    idx_init = '0'
    data_path = batch_settings["path_main"] + "\\data\\{}\\v{}\\".format(batch_settings["system"], batch_settings["job_version"])
    os.makedirs(data_path, exist_ok=True)
    data_file = "data_{}_v{}_init{}.csv".format(batch_settings["system"], batch_settings["job_version"],  idx_init)

    data = generate_ODE_data('lorenz_stable', [1., 1., 1.])
    pd.DataFrame(data).to_csv(data_path + data_file, header=False, index=False)





