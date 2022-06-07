import ProGED as pg
from ProGED.examples.ds2022.generate_data_ODE_systems import generate_ODE_data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 1. get data
    data = generate_ODE_data(system='VDP', inits=[-0.2, -0.8])

    # 2. generate grammar
    grammar = pg.grammar_from_template("polynomial",
                                       generator_settings={"variables": ["'x'", "'y'", "'z'"],
                                                           "p_vars": [1 / 3, 1 / 3, 1 / 3], "functions": [], "p_F": []})
    symbols = {"x": ["x", "y", "z"], "const": "C"}
    models = pg.generate.generate_models(grammar, symbols, dimension=3)

    # 2.2. generate model box manually
    ex1 = "C*y"
    ex2 = "C*y - C*x*x*y - C*x"
    symbols = {"x": ["x", "y"], "const": "C"}
    models = pg.ModelBox()
    models.add_system([ex1, ex2], symbols=symbols)

    # 3. param estimation settings
    optimizer_settings = {
        # "lower_upper_bounds": (-10, 10),
        "lower_upper_bounds": (-1, 1),
        "default_error": 10 ** 9,
        "strategy": 'rand1bin',
        "f": 0.45,
        "cr": 0.50,
        # "max_iter": 1000,
        "max_iter": 1,
        "pop_size": 5,
        "atol": 0.0001,
        "tol": 0.0001
    }
    estimation_settings = {"optimizer": 'differential_evolution',
                           "observed": ["x", "y"],
                           "optimizer_settings": optimizer_settings}

    # 4. estimate parameters
    models_out = pg.fit_models(models,
                               data=data,
                               task_type='differential',
                               time_index=0,
                               estimation_settings=estimation_settings)

    models_out[0].params
    models_out[0].get_full_expr()
    # for m in models_out.models:
    #     print(f"model: {str(m.get_full_expr()):<30}; error: {m.get_error():<15}")

