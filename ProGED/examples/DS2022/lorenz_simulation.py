import matplotlib; matplotlib.use('Qt5Agg')
import numpy as np
from scipy.integrate import odeint
import ProGED as pg
from utils.generate_data_ODE_systems import generate_ODE_data
import matplotlib.pyplot as plt



def plot_results(system, model, inits):

    data = generate_ODE_data(system=system, inits=inits)
    t = data[:, 0]

    lamb_odes = model.lambdify(list=True)

    def lambdified_odes(t, x):
        return [lamb_odes[i](*x) for i in range(len(lamb_odes))]

    sol = odeint(lambdified_odes, inits, t,
                 rtol=1e-12,
                 atol=1e-12,
                 tfirst=True)

    rmse = np.mean((data[:, 1:] - sol) ** 2)

    WIDTH, HEIGHT, DPI = 1000, 750, 100
    fig = plt.figure(facecolor='k', figsize=(WIDTH / DPI, HEIGHT / DPI))
    ax = fig.gca(projection='3d')
    ax.set_facecolor('k')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    s = 10
    for i in range(0, len(data)):
        ax.plot(sol[:, 0][i:i + s + 1], sol[:, 1][i:i + s + 1], sol[:, 2][i:i + s + 1], color='red', alpha=0.4)
        ax.plot(data[:, 1][i:i + s + 1], data[:, 2][i:i + s + 1], data[:, 3][i:i + s + 1], color='blue', alpha=0.4)

    ax.set_axis_off()

    fig, ax = plt.subplots(3)
    ax[0].plot(data[:, 0], data[:, 1], color='blue')
    ax[0].plot(data[:, 0], sol[:, 0], color='red')
    ax[0].set(xlabel='time', ylabel='X')
    ax[1].plot(data[:, 0], data[:, 2], color='blue')
    ax[1].plot(data[:, 0], sol[:, 1], color='red')
    ax[1].set(xlabel='time', ylabel='Y')
    ax[2].plot(data[:, 0], data[:, 3], color='blue')
    ax[2].plot(data[:, 0], sol[:, 2], color='red')
    ax[2].set(xlabel='time', ylabel='Z')
    plt.suptitle("Best possible reconstruction; RMSE = " + str(rmse))
    plt.show()

if __name__ == "__main__":

    # 4.1 get data
    # inits = [0, 1, 1.05]
    # data = generate_ODE_data(system='lorenz_chaotic', inits=inits)

    params_stable = [[10], [16], [-8/3]]
    params_chaotic = [[10.], [28.], [-2.667]]
    inits = [1., 1., 1.]
    data = generate_ODE_data(system='lorenz_stable', inits=inits)

    # 4.2 create lorenz model
    """
    ex1 = "C*y + C*x"
    ex2 = "C*x + C*x*z + C*y"
    ex3 = "C*x*y + C*z"
    """

    ex1 = "C*(y-x)"
    ex2 = "x*(C-z)-y"
    ex3 = "x*y - C*z"

    symbols = {"x": ["x", "y", "z"], "const": "C"}
    system = pg.ModelBox(observed=["x", "y", "z"])
    system.add_system([ex1, ex2, ex3], symbols=symbols)

    # 4.5: lorenz: check if the simulation works by inputing correct parameters
    #system[0].params = [[-10., 10.], [-1., 28., -1.], [1., -2.667]]
    #system[0].params = [[10.], [28.], [-2.667]]

