import pickle
import time
import ProGED as pg
from utils.generate_data_ODE_systems import generate_ODE_data
from ProGED.examples.DS2022.lorenz_simulation import plot_results


if __name__ == "__main__":
    #sys.stdout = open("D:\\Experiments\\DS2022\\trials\\lorenz_stable_point3_console.txt", "w")
    start_time = time.time()
    params = [[10], [16], [-8 / 3]]
    inits = [1., 1., 1.]
    data = generate_ODE_data(system='lorenz_stable', inits=inits)

    objective_settings = {
        "atol": 10 ** (-6), "rtol": 10 ** (-6),
        "use_jacobian": False, "simulate_separately": False}

    optimizer_settings = {
        "lower_upper_bounds": (-30, 30),
        "default_error": 10 ** 9,
        "strategy": 'rand1bin',
        "f": 0.45, "cr": 0.88,
        "max_iter": 200, "pop_size": 50,
        "atol": 0.01, "tol": 0.01}

    estimation_settings = {"optimizer": 'differential_evolution',
                           "optimizer_settings": optimizer_settings,
                           "objective_settings": objective_settings,
                           "verbosity": 2}

    symbols = {"x": ["x", "y", "z"], "const": "C"}
    system = pg.ModelBox(observed=["x", "y"])
    system.add_system(["C*(y-x)", "x*(C-z)-y", "x*y - C*z"], symbols=symbols)
    #system.add_system(["C*y + C*x", "C*x + C*x*z + C*y", "C*x*y + C*z"], symbols=symbols)
    #system.add_system(["C*y", "C*y + C", "C*x"], symbols=symbols)
    #system.add_system(["C*x*y*z", "C*z", "C*y"], symbols=symbols)

    models_out = pg.fit_models(system, data[:, (0, 1, 2)], task_type='differential', time_index=0, estimation_settings=estimation_settings)

    print("--End time in seconds: " + str(time.time() - start_time))

    with open("D:\\Experiments\\DS2022\\trials\\lorenz_stable_point46_fit.pg", "wb") as file:
        pickle.dump(models_out, file)

    with open("D:\\Experiments\\DS2022\\trials\\lorenz_stable_point32_settings.pg", "wb") as file:
        pickle.dump(estimation_settings, file)

    #sys.stdout.close()
    plot_results('lorenz_stable', models_out[2], inits)