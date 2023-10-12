import glob
import sys

import ProGED as pg

if __name__ == "__main__":
    name = sys.argv[1]

    with open("results_summary.csv", "w") as f:
        f.write("eqN, N, success\n")
    
    s = 0
    for eqN in range(100):
        filename = name + "_eq" + str(eqN) + "_"
        print(filename)
        files = glob.glob(filename + "*")
        #print(files)

        if len(files) > 0:
            models = pg.ModelBox()
            for file in files:
                models.load(file)
            
            best_models = models.retrieve_best_models(N=1)
            success = 0
            if len(best_models) > 0:
                if best_models[0].get_error() < 1e-9:
                    success = 1
                    s += 1
            N = len(models)

        else:
            success = 0
            N = 0

        txt = str(eqN) + ", " + str(N) + ", " + str(success)
        print("-->", txt)
        with open("results_summary.csv", "a") as f:
            f.write(txt + "\n")

    print(s)