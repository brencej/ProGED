import numpy as np
import matplotlib.pyplot as plt

from ProGED.bayesian_search import BayesianSearch
from ProGED.generators.hvae_generator import GeneratorHVAE, SymType, HVAE, Encoder, Decoder, GRU122, GRU221

universal_symbols = [{"symbol": 'x', "type": SymType.Var, "precedence": 5},
                     {"symbol": 'c', "type": SymType.Const, "precedence": 5},
                     {"symbol": '+', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '-', "type": SymType.Operator, "precedence": 0},
                     {"symbol": '*', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '/', "type": SymType.Operator, "precedence": 1},
                     {"symbol": '^', "type": SymType.Operator, "precedence": 2}]
                     # {"symbol": 'sin', "type": SymType.Fun, "precedence": 5},
                     # {"symbol": 'cos', "type": SymType.Fun, "precedence": 5},
                     # {"symbol": 'sqrt', "type": SymType.Fun, "precedence": 5}]


def read_eq_data(filename):
    x = []
    with open(filename, "r") as file:
        for row in file:
            line = [float(t) for t in row.strip().split(" ")]
            x.append(line)
    part = np.random.permutation(len(x))
    return np.array(x)[part[:10000], :]


def generate_data(function):
    x = np.linspace(0, 10, num=10000)[:, None]+1e-5
    return np.concatenate((x, function(x)), axis=1)


def read_equations(filename):
    eqs = []
    with open(filename, "r") as file:
        for l in file:
            eqs.append(l.strip().split(" "))
    return eqs


# Read equations used for training
eqs = read_equations("data/eqs_5_4k.txt")

# Read/Create data of observations of the function
# data = read_eq_data("..data/Feynman_with_units/III.15.12")
data = generate_data(lambda z: 1/3 + z + z*z/2)

# Train/Load the model
generator = GeneratorHVAE.train_and_init(eqs, ["x"], universal_symbols, epochs=20,
                                         hidden_size=64, representation_size=64,
                                         model_path="./parameters/test1.pt")
generator = GeneratorHVAE("./parameters/test1.pt", ["x"], universal_symbols)

# Run Bayesian search and print out the results
bs = BayesianSearch(generator=generator, initial_samples=512)
x, y, best_x, best_y = bs.search(data, iterations=64, eqs_per_iter=16)
plt.plot(best_y)
plt.show()
print(best_y)
