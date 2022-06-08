import os
import io
import pickle
import ProGED as pg
from ProGED.generate import generate_models

def create_sh_file(**batch_settings):
    sys_name = batch_settings["system_name"]
    job_vers = batch_settings["job_version"]
    pyfile_name = "slurm_run_batch_{}_v{}.py".format(sys_name, job_vers)

    title = "slurm_run_jobs_{}_v{}.sh".format(sys_name, job_vers)
    f = io.open(os.path.join(batch_settings["path_out"], title), "w", newline='\n')
    f.write("#!/bin/bash\n"
            "#SBATCH --job-name={}v{}\n".format(sys_name, job_vers))
    f.write("#SBATCH --time=2-00:00:00\n"
            "#SBATCH --mem-per-cpu=2GB\n")
    f.write("#SBATCH --array=0-{}\n".format(str(batch_settings["n_batches"]-1)))
    f.write("#SBATCH --cpus-per-task=1\n")
    f.write("#SBATCH --output=jobs/{}/v{}/slurm_output_%A_%a.out\n".format(sys_name, job_vers))
    f.write("\nsingularity exec proged_container.sif python3.7 " + pyfile_name + " ${SLURM_ARRAY_TASK_ID}")
    f.close()


def create_batches(**batch_settings):

    # generate grammar
    grammar = pg.grammar_from_template("polynomial",
                                       generator_settings={"variables": batch_settings["variables"],
                                                           "p_vars": batch_settings["p_vars"], "functions": [], "p_F": []})
    symbols = {"x": batch_settings["variables"], "const": "C"}

    # generate models from grammar
    models = generate_models(grammar,
                             symbols,
                             strategy_settings={"N": batch_settings["num_samples"]},
                             dimension=len(batch_settings["variables"]))

    model_batches = models.split(n_batches=batch_settings["n_batches"])

    # save batches
    path_for_jobs = os.path.join(batch_settings["path_out"],
                                 "jobs",
                                 "{}".format(batch_settings["system_name"]),
                                 "v{}".format(batch_settings["job_version"]))
    os.makedirs(path_for_jobs, exist_ok=True)
    for ib in range(batch_settings["n_batches"]):
        file_name = os.path.join(path_for_jobs, "job_{}_v{}_batch{}.pg".format(batch_settings["system_name"], batch_settings["job_version"], str(ib)))
        with open(file_name, "wb") as file:
            pickle.dump(model_batches[ib], file)

    # create shell file
    create_sh_file(**batch_settings)

if __name__ == '__main__':

    # settings
    batch_settings = {
        "system_name": 'VDP',
        "job_version": '1',
        "variables": ["'x'", "'y'"],
        "p_vars": [1 / 2, 1 / 2],
        "num_samples": 100,
        "n_batches": 10,
        "path_out": os.path.join("D:", "ED_paper")
    }

    create_batches(**batch_settings)



